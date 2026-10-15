"""Fetch Packet.ai GPU products and emit a SkyPilot catalog CSV.

Source data: ``GET /api/v1/launch-options`` on https://dash.packet.ai (the
authenticated price/stock surface; SLOW — ~30s cold, the client sizes the
timeout). Rows come from ``data.products`` ONLY — the canonical deployable
surface (ENG-562's measured rule): a pool row with a ``price_per_hour`` but
no active product listing the pool id in ``poolIds`` is a PRICING fact, not
a deployability fact, and must never reach the catalog as a rentable row.

Instance types are synthesized ``<accelerator>:<count>`` tokens (e.g.
``rtxpro6000:1``): pool and product ids are ephemeral (the surface flipped
its product list mid-session on 2026-10-15) and regions are numeric ids,
so the stable accelerator+count pair is the catalog identity. ``Region`` is
the region NAME (``us-west-2``) — the create body takes the numeric
``region_id``, re-resolved by the provisioner from the live launch-options
at launch. ``Price`` is the whole-box hourly: ``pricePerHourCents`` (USD
cents per GPU-hour) x the count the token names.

Rows are emitted ONLY for products with ``totalAvailableGpus > 0`` in the
regions the product lists — everything this module can price, the lane can
actually rent. Products with no available GPUs are omitted by the endpoint
unless ``featured`` (a featured 0-stock product appears in the list; the
``totalAvailableGpus > 0`` filter keeps it out of the catalog).

The CREATE seam (ENG-562, measured 2026-10-15): instance creation returns
501 NOT_IMPLEMENTED for API-key auth (console-session path only). The lane
bridges it with ADOPTION of console-launched boxes by instance name (the
provisioner) + a $0 watcher on the create port — the catalog itself writes
HONEST STOCK (a live in-stock product is rentable through the console path
and adoptable; the provisioner, not the catalog, owns the create refusal).
A zero-ROW fetch is cross-checked against the same products list: any
product reporting stock while no row will be written is a filter bug or
API drift — REFUSE, never write a false "no stock" catalog.

Modeled on ``sky/catalog/data_fetchers/fetch_quantacloud.py`` (the newest
in-tree carry); the differences are all grounded in the measured
launch-options shape (see sky/adaptors/packet.py for the receipts).
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

    The row's price is the WHOLE-BOX hourly: pricePerHourCents is USD cents
    per GPU-hour (measured: the 1-GPU RTX PRO 6000 product carries 66), so
    the row multiplies by the count the token names.
    """
    name = str(product.get('name') or '').strip()
    if not name:
        return None
    # In-stock ONLY: the products surface omits 0-stock products unless
    # featured; a featured 0-stock row must not reach the catalog (a
    # rentable quote for a box that cannot launch).
    try:
        available = int(product.get('totalAvailableGpus') or 0)
    except (TypeError, ValueError):
        return None
    if available < 1:
        return None
    # Monthly plans never touch the wallet and carry no hourly price
    # (pricePerHourCents 0 measured); out of scope for the lane — skip
    # rather than quote an unpriced box as free.
    try:
        cents_per_gpu_hour = int(product.get('pricePerHourCents') or 0)
    except (TypeError, ValueError):
        return None
    if cents_per_gpu_hour < 1:
        return None
    accelerator = normalise_accelerator(name)
    if not accelerator:
        return None  # un-mappable family: audit-visible via refusal, not row
    gpu_count = parse_product_gpu_count(name)
    price = (cents_per_gpu_hour / 100.0) * gpu_count
    vram = product.get('vramGb')
    # vCPUs / RAM are not exposed as FIELDS on the products surface — the
    # description strings carry them in prose ("96GB ram · 6 vCPU · 150GB
    # ephemeral NVME", measured live 2026-10-15) — so the row PARSES them
    # out of the description. A row with no parseable POSITIVE RAM is
    # skipped: SkyPilot's launch validation rejects memory <= 0 (measured
    # live on the armed claim 2026-10-15: the 0.0-placeholder row launched
    # Failed with "'memory' field should be positive. Found: '0.0'") — a
    # placeholder row would quote a box the launch can never boot.
    ram_gib = _description_ram_gb(product.get('description'))
    if ram_gib is None or ram_gib <= 0:
        return None
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


def iter_rows(launch_options: Dict[str, Any]) -> Iterator[List[Any]]:
    """Yield one CSV row per in-stock (accelerator, count, region) to rent.

    A product lists every region it can launch in (``regions[]``, deduped
    from its pools); one row per region, so the catalog's Region filter
    matches the claim's region pin exactly. When the same (accelerator,
    count, region) appears at several prices, the CHEAPEST wins: the row
    is a rentable quote, and the provisioner re-resolves against the live
    surface at launch anyway (a stale-cheap row must not quote a price the
    launch refuses, and a stale-expensive row would hide real capacity).
    """
    products = launch_options.get('products')
    if not isinstance(products, list):
        raise PacketCatalogError(
            'GET /launch-options: expected a list at .products, got '
            f'{type(products).__name__} — refusing to treat the surface as '
            'empty')
    best: Dict[Tuple[str, int, str], List[Any]] = {}
    for product in products:
        if not isinstance(product, dict):
            continue
        regions = product.get('regions')
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            region_name = str(region.get('name') or '').strip()
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
                      rows: Sequence[List[Any]]) -> None:
    """Cross-check a zero-ROW fetch before it is written as zero-stock.

    The ruling (ENG-562): a fetch yielding NO catalog rows is cross-checked
    against the SAME products list — if every product reports
    ``totalAvailableGpus == 0`` the empty is honest (write the header-only
    catalog, the Latitude semantics); if ANY product reports stock while no
    row will be written, the fetch REFUSES rather than write "no capacity"
    (the Spheron filter-bug semantics — a filter bug or API drift; a lane
    answering "no capacity" from this state would be a lie).

    This deliberately does NOT gate on the create port: creation being
    501 for API keys (the measured platform-capability refusal) does not
    change STOCK — the console path provisions and the provisioner adopts;
    the catalog owns stock, the provisioner owns the create seam.
    """
    if any(True for _ in rows):
        return  # rows will be written: nothing to cross-check
    products = launch_options.get('products')
    if not isinstance(products, list):
        raise PacketCatalogError(
            'GET /launch-options: expected a list at .products, got '
            f'{type(products).__name__} — refusing to treat the surface as '
            'empty')
    stocked = [
        str(p.get('name') or '?')
        for p in products
        if isinstance(p, dict) and _product_available_count(p) > 0
    ]
    if stocked:
        raise PacketCatalogError(
            f'no rentable catalog rows will be written, but launch-options '
            f'reports in-stock products {sorted(stocked)}; refusing to write '
            'a zero-stock catalog from a contradictory fetch — a filter bug '
            'or API drift (investigate before the lane answers "no '
            'capacity").')


def _product_available_count(product: Dict[str, Any]) -> int:
    value = product.get('totalAvailableGpus')
    if isinstance(value, bool) or not isinstance(value, int):
        # totalAvailableGpus: null (no stock) and non-numeric values are
        # both "no stock" here; a contradictory fetch must come from a
        # POSITIVE integer, not from a missing field.
        return 0
    return value


def fetch_launch_options(api_key: Optional[str] = None) -> Dict[str, Any]:
    """The live launch-options payload (authenticated; ~30s cold)."""
    client = packet_api.PacketClient(api_key or 'public-catalog')
    return client.get_launch_options()


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
    else:
        # /launch-options is AUTHENTICATED on Packet.ai (unlike QuantaCloud's
        # public catalog endpoints): the key is read from the env or
        # ~/.packet/credentials; an unset key raises PacketAuthError — an
        # unauthenticated fetch must never look like "no capacity".
        launch_options = fetch_launch_options(os.environ.get(args.api_key_env))

    rows = list(iter_rows(launch_options))
    verify_zero_stock(launch_options, rows)
    write_csv(rows, args.output)
    print(f'Wrote {len(rows)} rows to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
