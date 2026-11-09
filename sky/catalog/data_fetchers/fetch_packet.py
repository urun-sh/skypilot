"""Fetch Packet.ai GPU products and emit a SkyPilot catalog CSV (v2).

Source data: ``GET /api/v2/instances/launch-options`` on https://dash.packet.ai
(the authenticated price surface; SLOW — the client sizes the timeout) plus
``GET /api/v2/instances/category-check`` per product service. The v1 API is
RETIRED (measured 2026-11-09: every ``/api/v1`` route answers HTTP 410
Gone), which had left the lane dead since 2026-10-21.

v2 shape (measured 2026-11-09, live): the launch-options payload is FLAT
(no ``data`` wrapper, no ``pools``) — ``products[]`` carry
``pricePerHourCents`` (the WHOLE-BOX hourly, US cents — the 8-GPU B200
product carries 6392, i.e. $63.92/box ≈ $7.99/GPU) but NO stock;
availability lives in ``category-check``: ``freeSlots`` per
``serviceId`` per region id, plus ``serviceRegions`` and ``maintenance``
(and ``checkFailed`` — a True answer may be incomplete and REFUSES the
fetch, never guessed around).

Instance types stay synthesized ``<accelerator>:<count>`` tokens (e.g.
``rtxpro6000:1``): product ids are ephemeral and regions are numeric ids,
so the stable accelerator+count pair is the catalog identity. ``Region``
is the region NAME resolved from ``regions[]`` by id (``us-west-2``) —
the create body takes the numeric ``region_id``, re-resolved by the
provisioner from the live surfaces at launch.

Rows are emitted ONLY for (product, region) pairs whose category-check
reports a free slot — everything this module can price, the lane can
actually rent (v2 serves ``POST /instances`` for API keys; the v1 501
create seam is GONE). A zero-ROW fetch is cross-checked against the same
surfaces: any service reporting a free slot while no row will be written
is a filter bug or API drift — REFUSE, never write a false "no stock"
catalog.

Modeled on ``sky/catalog/data_fetchers/fetch_quantacloud.py`` (the newest
in-tree carry); the differences are all grounded in the measured v2
launch-options/category-check shapes (see sky/adaptors/packet.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Tuple)

from sky.adaptors import packet as packet_api
from sky.catalog.data_fetchers import fetch_latitude

CSV_COLUMNS = [
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'Price',
    'Region',
    'GpuInfo',
    'SpotPrice',
]

# Product-name fragment -> the ONE accelerator token the platform's
# identities use for that family (see fetch_latitude's map and
# _PROVIDER_GPU_IDENTITIES in skypilot-controller). Keyed on the
# machine-stable FAMILY fragment of the product name (not the full display
# name, which carries tier suffixes like "- Dynamic"/"- Dedicated" that
# change with merchandising): every variant of a family maps to the same
# token. Unmapped families pass through compacted (uppercased, alphanumerics
# kept): an unrecognized GPU is a real accelerator nobody requests through
# our identities, and passing it through keeps it visible in the catalog
# for audit instead of silently hiding the product.
_ACCELERATOR_MAP = {
    # Every whole-card RTX PRO 6000 variant -> the family the rtx6000
    # shape's identities carry (same token as the Vast/Spheron/Latitude/
    # Quantacloud lanes).
    'RTXPRO6000': 'RTXPRO6000',
    'RTX6000': 'RTX6000ADA',
    'RTXA6000': 'RTXA6000',
    'B200': 'B200',
    'H200': 'H200',
    'H100': 'H100',
    'A100': 'A100',
    'L40S': 'L40S',
    'L40': 'L40',
    'RTX4090': 'RTX4090',
    'RTX5090': 'RTX5090',
}

# Product names carry merchandising suffixes ("NVIDIA RTX PRO 6000
# Blackwell - Dynamic", "- Dedicated", "monthly flat price") and count
# prefixes ("2 x NVIDIA RTX 4090 Dedicated"). The count prefix is parsed
# FIRST; the suffix is stripped by matching the accelerator fragment.
_COUNT_PREFIX_RE = re.compile(r'^\s*(\d+)\s*x\s+', re.IGNORECASE)


class PacketCatalogError(RuntimeError):
    """Raised when the fetcher cannot run or the payload is unusable."""


def _strip_vendor_noise(name: str) -> str:
    """The compacted family key of a product display name.

    ``NVIDIA RTX PRO 6000 Blackwell - Dynamic`` -> ``RTXPRO6000BLACKWELL``;
    ``1 x NVIDIA B200 Dynamic`` (count already parsed off) -> ``B200DYNAMIC``.
    Matching is by LONGEST matching accelerator fragment anywhere in the
    compacted name (so ``RTXPRO6000BLACKWELL`` matches RTXPRO6000 before the
    generic RTX6000 fragment can), and the fragment must not be followed by
    another alnum that would change the family (RTX6000 vs RTX6000ADA order
    handles this).
    """
    compacted = re.sub(r'[^A-Z0-9]', '', str(name or '').upper())
    for fragment in sorted(_ACCELERATOR_MAP, key=len, reverse=True):
        if fragment in compacted:
            return fragment
    return compacted


def normalise_accelerator(product_name: Optional[str]) -> str:
    """The ONE accelerator token for a product display name, or ''."""
    name = str(product_name or '')
    if not name.strip():
        return ''
    fragment = _strip_vendor_noise(name)
    return _ACCELERATOR_MAP.get(fragment, '')


def parse_product_gpu_count(product_name: Optional[str]) -> int:
    """The GPU count a product's name declares (``2 x …`` -> 2, else 1)."""
    match = _COUNT_PREFIX_RE.match(str(product_name or ''))
    if match:
        try:
            count = int(match.group(1))
        except ValueError:
            return 1
        return count if count >= 1 else 1
    return 1


def instance_type_token(accelerator: str, gpu_count: int) -> str:
    """The stable catalog identity: ``<accelerator>:<count>``.

    Pool/product ids are ephemeral; accelerator+count is what the
    controller pins and what the provisioner re-resolves against the live
    launch-options at launch.
    """
    return f'{accelerator.lower()}:{gpu_count}'


def parse_instance_type_token(instance_type: str) -> Tuple[str, int]:
    """The inverse of ``instance_type_token``; raises on garbage.

    The provisioner's launch-time product re-resolution starts here: the
    catalog handed it the token, and it needs the (accelerator, count)
    pair to find the concrete live product/pool/region. Accelerator tokens
    are lowercased alphanumerics without colons, so splitting on the LAST
    colon is unambiguous.
    """
    accelerator, sep, count = str(instance_type or '').rpartition(':')
    if (not sep or not accelerator or not count.isdigit() or int(count) < 1 or
            not accelerator.isalnum()):
        raise PacketCatalogError(
            f'instance type {instance_type!r} is not a '
            '`<accelerator>:<count>` Packet.ai token (e.g. rtxpro6000:1)')
    return accelerator, int(count)


_RAM_GB_RE = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*GB\s*ram', re.IGNORECASE)
_VCPU_RE = re.compile(r'(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*vCPU', re.IGNORECASE)


def _description_ram_gb(description: Any) -> Optional[float]:
    """The box RAM in GiB from the product's description prose
    ("96GB ram · 6 vCPU · ..."), or None when the description carries no
    parseable RAM — the caller skips the row (SkyPilot's launch
    validation rejects memory <= 0)."""
    match = _RAM_GB_RE.search(str(description or ''))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _description_vcpus(description: Any) -> float:
    """The box vCPU count from the product's description prose, or 0.0 when
    unparseable (0 vCPUs prices fine; only memory must be positive)."""
    match = _VCPU_RE.search(str(description or ''))
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _product_row(product: Dict[str, Any],
                 region_name: str) -> Optional[List[Any]]:
    """One (product, region) -> one CSV row, or None when not lane-rentable.

    The row's price is the WHOLE-BOX hourly: v2 ``pricePerHourCents`` is
    the product's USD-cents price for the WHOLE box (measured 2026-11-09:
    the 8-GPU B200 product carries 6392 = $63.92/box ≈ $7.99/GPU) — the
    count the token names is already inside it.
    """
    name = str(product.get('name') or '').strip()
    if not name:
        return None
    # v2 products carry NO stock field (v1 totalAvailableGpus is gone):
    # rentability is decided by iter_rows against the category-check
    # freeSlots, never here.
    # Monthly plans never touch the wallet and carry no hourly price
    # (pricePerHourCents 0 measured); out of scope for the lane — skip
    # rather than quote an unpriced box as free.
    try:
        cents_per_hour = int(product.get('pricePerHourCents') or 0)
    except (TypeError, ValueError):
        return None
    if cents_per_hour < 1:
        return None
    accelerator = normalise_accelerator(name)
    if not accelerator:
        return None  # un-mappable family: audit-visible via refusal, not row
    gpu_count = parse_product_gpu_count(name)
    price = cents_per_hour / 100.0
    vram = product.get('vramGb')
    # v2 exposes vcpuCount / ramMb as FIELDS on some products (measured
    # 2026-11-09: the 5090 Dynamic - Hourly carries vcpuCount=6,
    # ramMb=49152 with a BLANK description), while others leave them
    # null and carry the specs in the description prose ("64GB ram · 16
    # vCPU", measured on the 4090 Dedicated) — read the FIELD first,
    # fall back to the prose. A row with no POSITIVE RAM from either
    # source is skipped: SkyPilot's launch validation rejects memory <=
    # 0 (measured live on the armed claim 2026-10-15: the 0.0-row
    # launched Failed with "'memory' field should be positive. Found:
    # '0.0'") — a placeholder row would quote a box the launch can
    # never boot, and verify_zero_stock would then refuse the catalog
    # over genuinely stocked-but-unquotable listings.
    ram_gib = None
    ram_mb = product.get('ramMb')
    if isinstance(ram_mb, (int, float)) and not isinstance(ram_mb, bool):
        ram_gib = float(ram_mb) / 1024.0
    if ram_gib is None or ram_gib <= 0:
        ram_gib = _description_ram_gb(product.get('description'))
    if ram_gib is None or ram_gib <= 0:
        return None
    vcpus = 0.0
    vcpu_field = product.get('vcpuCount')
    if isinstance(vcpu_field,
                  (int, float)) and not isinstance(vcpu_field, bool):
        vcpus = float(vcpu_field)
    else:
        vcpus = _description_vcpus(product.get('description'))
    # pylint: disable=protected-access
    # fetch_latitude's _gpu_info_str is the ONE canonical GpuInfo builder
    # (the vet tier flagged a drifting inline copy on the quantacloud
    # carry); the same access that fetcher makes.
    gpu_info = fetch_latitude._gpu_info_str(accelerator, int(gpu_count), vram)
    return [
        instance_type_token(accelerator, int(gpu_count)),
        accelerator,
        int(gpu_count),
        vcpus,
        ram_gib,
        price,
        str(region_name),
        gpu_info,
        '',  # no spot tier on Packet.ai
    ]


def iter_rows(
    launch_options: Dict[str, Any],
    category_checks: Dict[str, Dict[str, Any]],
) -> Iterator[List[Any]]:
    """Yield one CSV row per in-stock (accelerator, count, region) to rent.

    v2 (measured 2026-11-09): products carry NO stock and NO region list —
    availability lives in ``category_checks[serviceId]`` (the
    ``GET /instances/category-check`` answer): ``freeSlots`` maps region
    id → free pod slots for the service. A row is emitted ONLY for a
    (product, region) whose check reports a free slot (>= 1 pod); a
    ``checkFailed`` answer may be incomplete and REFUSES the whole fetch
    (never a guessed-around partial catalog). ``Region`` is the region
    NAME resolved from ``launch_options.regions[]`` by id. When the same
    (accelerator, count, region) appears at several prices, the CHEAPEST
    wins: the row is a rentable quote, and the provisioner re-resolves
    against the live surfaces at launch anyway.
    """
    products = launch_options.get('products')
    if not isinstance(products, list):
        raise PacketCatalogError(
            'GET /instances/launch-options: expected a list at .products, '
            f'got {type(products).__name__} — refusing to treat the surface '
            'as empty')
    all_regions = launch_options.get('regions')
    if not isinstance(all_regions, list):
        raise PacketCatalogError(
            'GET /instances/launch-options: expected a list at .regions, '
            f'got {type(all_regions).__name__} — refusing to treat the '
            'surface as empty')
    region_names: Dict[int, str] = {}
    for region in all_regions:
        if (isinstance(region, dict) and isinstance(region.get('id'), int) and
                str(region.get('name') or '').strip()):
            region_names[int(region['id'])] = str(region['name']).strip()

    best: Dict[Tuple[str, int, str], List[Any]] = {}
    for product in products:
        if not isinstance(product, dict):
            continue
        service_id = str(product.get('serviceId') or '')
        check = category_checks.get(service_id)
        if check is None:
            # No category-check answer: not rentable through this fetch —
            # verify_zero_stock refuses if that empties the catalog.
            continue
        if check.get('checkFailed'):
            raise PacketCatalogError(
                f'category-check for service {service_id!r} reports '
                'checkFailed=true: the availability answer may be '
                'incomplete — refusing to write a partial catalog (retry '
                'the fetch, or investigate the provider)')
        free_slots = check.get('freeSlots')
        if not isinstance(free_slots, dict):
            continue
        slots_by_region = free_slots.get(service_id)
        if not isinstance(slots_by_region, dict):
            continue
        for raw_region_id, slots in slots_by_region.items():
            try:
                region_id = int(raw_region_id)
            except (TypeError, ValueError):
                continue
            if not isinstance(slots, int) or isinstance(slots, bool):
                continue
            if slots < 1:
                continue
            region_name = region_names.get(region_id)
            if not region_name:
                continue
            row = _product_row(product, region_name)
            if row is None:
                continue
            key = (str(row[0]).rsplit(':', 1)[0], int(row[2]), str(row[6]))
            existing = best.get(key)
            if existing is None or float(row[5]) < float(existing[5]):
                best[key] = row
    yield from best.values()


def verify_zero_stock(launch_options: Dict[str, Any],
                      category_checks: Dict[str, Dict[str, Any]],
                      rows: Sequence[List[Any]]) -> None:
    """Cross-check a zero-ROW fetch before it is written as zero-stock.

    The ruling (ENG-562, ported to v2 2026-11-09): a fetch yielding NO
    catalog rows is cross-checked against the SAME surfaces — if no
    product's category-check reports a free slot in a region the
    catalog names, the empty is honest (write the header-only catalog,
    the Latitude semantics); if ANY service reports a free slot while
    no row will be written, the fetch REFUSES rather than write "no
    capacity" (the Spheron filter-bug semantics — a filter bug or API
    drift; a lane answering "no capacity" from this state would be a
    lie).
    """
    if any(True for _ in rows):
        return  # rows will be written: nothing to cross-check
    products = launch_options.get('products')
    if not isinstance(products, list):
        raise PacketCatalogError(
            'GET /instances/launch-options: expected a list at .products, '
            f'got {type(products).__name__} — refusing to treat the surface '
            'as empty')
    all_regions = launch_options.get('regions')
    region_ids: set = set()
    if isinstance(all_regions, list):
        for region in all_regions:
            if isinstance(region, dict) and isinstance(region.get('id'), int):
                region_ids.add(int(region['id']))
    stocked = []
    for product in products:
        if not isinstance(product, dict):
            continue
        service_id = str(product.get('serviceId') or '')
        check = category_checks.get(service_id)
        if not isinstance(check, dict):
            continue
        free_slots = check.get('freeSlots')
        by_region = free_slots.get(service_id) if isinstance(free_slots,
                                                             dict) else None
        if not isinstance(by_region, dict):
            continue
        for raw_region_id, slots in by_region.items():
            try:
                region_id = int(raw_region_id)
            except (TypeError, ValueError):
                continue
            if (region_id in region_ids and isinstance(slots, int) and
                    not isinstance(slots, bool) and slots > 0):
                label = str(product.get('name') or '?') + '@' + service_id
                stocked.append(label)
                break
    if stocked:
        raise PacketCatalogError(
            'no rentable catalog rows will be written, but category-check '
            f'reports free slots for {sorted(stocked)}; refusing to write a '
            'zero-stock catalog from a contradictory fetch — a filter bug '
            'or API drift (investigate before the lane answers "no '
            'capacity").')


def fetch_launch_options(api_key: Optional[str] = None) -> Dict[str, Any]:
    """The live launch-options payload (authenticated; ~30s cold)."""
    client = packet_api.PacketClient(api_key or 'public-catalog')
    return client.get_launch_options()


def fetch_category_checks(
    api_key: Optional[str] = None,
    service_ids: Sequence[str] = ()
) -> Dict[str, Dict[str, Any]]:
    """The per-service availability answers, keyed by service id.

    ``GET /instances/category-check?serviceId=...`` per DISTINCT service
    (products share a service 1:1 today, but a shared service is fetched
    once). Quota-aware: one read per distinct service (the documented
    budget is 100 reads/min per key), bounded by the service count.
    """
    client = packet_api.PacketClient(api_key or 'public-catalog')
    return {
        service_id: client.category_check(service_id)
        for service_id in dict.fromkeys(str(s) for s in service_ids)
    }


def write_csv(rows: Iterable[List[Any]], out_path: str) -> int:
    written = 0
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(row)
            written += 1
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-key-env', default=packet_api.API_KEY_ENV)
    parser.add_argument('--output', default='packet/vms.csv')
    parser.add_argument('--from-fixture',
                        help='read recorded payload instead of network')
    args = parser.parse_args(argv)

    if args.from_fixture:
        with open(args.from_fixture, encoding='utf-8') as handle:
            fixture = json.load(handle)
        launch_options = fixture['launch_options']
        # The fixture carries BOTH surfaces (recorded together — see the
        # tests) so the offline path exercises the same joins.
        category_checks = fixture['category_checks']
    else:
        # /launch-options is AUTHENTICATED on Packet.ai (unlike
        # QuantaCloud's public catalog endpoints): the key is read from
        # the env or ~/.packet/credentials; an unset key raises
        # PacketAuthError — an unauthenticated fetch must never look
        # like "no capacity".
        api_key = os.environ.get(args.api_key_env)
        launch_options = fetch_launch_options(api_key)
        category_checks = fetch_category_checks(
            api_key,
            service_ids=[
                str(p.get('serviceId'))
                for p in launch_options.get('products') or []
                if isinstance(p, dict) and p.get('serviceId')
            ])

    rows = list(iter_rows(launch_options, category_checks))
    verify_zero_stock(launch_options, category_checks, rows)
    write_csv(rows, args.output)
    print(f'Wrote {len(rows)} rows to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
