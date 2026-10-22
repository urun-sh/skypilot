"""Fetch MassedCompute GPU products and emit a SkyPilot catalog CSV.

One row per in-capacity (product, region) pair the lane can actually rent,
from the authenticated ``GET /gpu-inventory`` (Bearer token; verified live
2026-10-21):

* **InstanceType is the product name** (``gpu_1x_pro_6000_blackwell``).
  The name is the launch API's ``productName`` — stable, unlike
  QuantaCloud's ephemeral offer UUIDs — so the catalog token passes
  through to the launch verbatim and nothing is re-resolved.
* **AcceleratorName comes from ``_ACCELERATOR_MAP``**, keyed by the GPU
  family slug parsed out of the product name (the ``gpu_<N>x_<slug>``
  prefix carries the per-box GPU count; variant suffixes like
  ``_high_ram``/``_low_storage`` are stripped). Every family the live
  inventory carries on 2026-10-21 is mapped (PRO 6000 Blackwell, PRO 4500
  Blackwell, 6000 ADA, A6000, A5000, A100 incl. SXM4/DGX, A30, H100 incl.
  SXM5/NVL, H200 NVL, B200/B300 SXM6); an unmapped family passes through
  COMPACTED (underscores/hyphens removed, uppercased) so an unrecognized
  GPU stays visible in the catalog for audit instead of silently hiding
  the offer.
* **CPU-only products are excluded** (``cpu_small_amd_epyc`` and
  friends): the lane rents accelerator shapes; a CPU box has no
  accelerator token and would pollute the stock aggregation.
* **Price is ``price_cents_per_hour`` converted to USD** (÷100): 219
  cents/hr → $2.19/hr for the 1x RTX PRO 6000 Blackwell. A product without
  a usable price is skipped: an unpriceable row would quote as free (fail
  closed, not silent).
* **Region is the product's ``regions_with_capacity_available`` name
  token** (``us-east-1``, ``us-central-9``...), ONE ROW PER region with
  capacity: the controller's residency filter exact-matches ``Region``
  against budget ``allowedRegions`` tokens, and the provisioner pins the
  launch's ``regionName`` by the same token — the catalog must emit
  exactly the token the launch API accepts.
* **AvailableUnits is the product's ``capacity_available``** on every row
  of that product: the stock aggregation (sky/catalog/offerings.py,
  ``_STOCK_BY_AVAILABLE_UNITS`` — the Vast semantics) sums the units
  behind each row, so row multiplicity must never be read as stock.
* **A zero-row fetch is cross-checked before it is written**
  (``verify_zero_stock``): every GPU product reporting
  ``capacity_available > 0`` while no row will be written is a
  CONTRADICTION — a filter bug or API drift must never read as "no
  capacity" (the Spheron filter-bug semantics). A genuinely sold-out GPU
  market (every GPU product at zero, CPU capacity notwithstanding) is the
  honest header-only write.
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

from sky.adaptors import massed as massed_api
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
    # The units column the stock aggregation reads for this provider
    # (sky/catalog/offerings.py _STOCK_BY_AVAILABLE_UNITS — the Vast
    # semantics): capacity_available per product row.
    'AvailableUnits',
]

# GPU family slug (the product name minus the ``gpu_<N>x_`` prefix and any
# variant suffix) -> the ONE accelerator token the platform's identities
# use for that family (see fetch_latitude/fetch_quantacloud's maps and
# _PROVIDER_GPU_IDENTITIES in skypilot-controller). Keyed by FAMILY, not
# the full product name, so one entry covers every count and variant of a
# family (gpu_2x_pro_6000_blackwell is the same RTXPRO6000 as the 1x).
# Keys are lowercase; lookup lowercases the parsed slug.
_ACCELERATOR_MAP = {
    # Every whole-card RTX PRO 6000 Blackwell variant -> the family the
    # rtx6000 shape's identities carry (same token as the
    # Vast/Spheron/Latitude/QuantaCloud lanes).
    'pro_6000_blackwell': 'RTXPRO6000',
    'pro_4500_blackwell': 'RTXPRO4500',
    '6000_ada': 'RTX6000ADA',
    'a6000': 'A6000',
    'a5000': 'A5000',
    'a100': 'A100',
    'a100_sxm4': 'A100',
    'dgx_a100': 'A100',
    'a30': 'A30',
    'h100': 'H100',
    'h100_sxm5': 'H100',
    'h100_nvl': 'H100',
    'h200_nvl': 'H200',
    'b200_sxm6': 'B200',
    'b300_sxm6': 'B300',
}

# Product-name suffixes that mark a VARIANT of a family (same GPUs,
# different RAM/storage/billing): stripped before the family lookup so a
# variant maps to the family accelerator, never to its own token.
_VARIANT_SUFFIXES = (
    '_high_ram',
    '_low_ram',
    '_nvlink',
    '_spot',
    '_low_storage',
)

# ``gpu_<N>x_<slug>`` — the product-name grammar every GPU product in the
# live inventory follows (cpu_* products do not match and are skipped).
_PRODUCT_RE = re.compile(r'^gpu_(\d+)x_([A-Za-z0-9_-]+)$')

# ``1x RTX PRO 6000 Blackwell (96GB)`` — the per-GPU vram the GpuInfo
# column needs, parsed from the product description (the specs carry only
# system RAM, not GPU RAM).
_VRAM_RE = re.compile(r'\((\d+(?:\.\d+)?)\s*GB\)', re.IGNORECASE)


class MassedCatalogError(RuntimeError):
    """Raised when the fetcher cannot run or the payload is unusable."""


def _parse_product(product_name: str) -> Optional[Tuple[int, str]]:
    """``gpu_1x_pro_6000_blackwell`` -> ``(1, 'pro_6000_blackwell')``.

    Returns None for anything that is not a ``gpu_<N>x_<slug>`` product
    (the CPU-only ``cpu_*`` boxes): those have no accelerator token and
    are not lane-rentable shapes.
    """
    match = _PRODUCT_RE.match(str(product_name or '').strip())
    if match is None:
        return None
    count = int(match.group(1))
    if count < 1:
        return None
    family = match.group(2).lower()
    for suffix in sorted(_VARIANT_SUFFIXES, key=len, reverse=True):
        if family.endswith(suffix):
            family = family[:-len(suffix)]
            break
    if not family:
        return None
    return count, family


def normalise_accelerator(family_slug: Optional[str]) -> str:
    """The family slug -> the catalog AcceleratorName.

    Unmapped families pass through compacted (underscores/hyphens
    removed, uppercased): an unrecognized GPU is a real accelerator nobody
    requests through our identities, and passing it through keeps it
    visible in the catalog for audit instead of silently hiding the
    offer.
    """
    slug = str(family_slug or '').strip().lower()
    if not slug:
        return ''
    return _ACCELERATOR_MAP.get(slug,
                                slug.replace('_', '').replace('-', '').upper())


def _vram_gib_per_gpu(description: Any) -> Optional[float]:
    """The per-GPU vram from the product description, or None.

    ``1x RTX PRO 6000 Blackwell (96GB)`` -> 96.0. The GpuInfo column
    carries per-GPU memory (the DeviceMemoryGiB pass in
    sky/catalog/common.py reads it); None leaves the memory entry absent
    rather than invented.
    """
    match = _VRAM_RE.search(str(description or ''))
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _capacity(product: Dict[str, Any]) -> int:
    """The product's ``capacity_available``, 0 when unusable/missing."""
    value = product.get('capacity_available')
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _product_rows(product_name: str, product: Dict[str,
                                                   Any]) -> Iterator[List[Any]]:
    """One product -> one CSV row per region with capacity, or nothing.

    Skips (fail closed, never a garbage row): non-GPU products (the
    cpu_* boxes), zero-capacity products (honest sold-out), products with
    no region carrying capacity (nothing rentable anywhere), and products
    without a usable price (would quote as free).
    """
    parsed = _parse_product(product_name)
    if parsed is None:
        return
    gpu_count, family = parsed
    instance_type = (product.get('instance_type') or {})
    raw_cents = instance_type.get('price_cents_per_hour')
    if isinstance(
            raw_cents,
            bool) or not isinstance(raw_cents,
                                    (int, float)) or float(raw_cents) <= 0:
        return  # unpriceable: skip rather than quote as free
    # CENTS per hour -> USD per hour (the catalog's Price unit): 219 -> 2.19.
    price = float(raw_cents) / 100.0
    specs = instance_type.get('specs') or {}
    vcpus = float(specs.get('vcpu_count') or 0)
    mem_gib = float(specs.get('memory_gib') or 0)
    capacity = _capacity(product)
    if capacity < 1:
        return  # sold out: nothing rentable under this product
    regions = []
    for region in product.get('regions_with_capacity_available') or []:
        if not isinstance(region, dict):
            continue
        name = str(region.get('name') or '').strip()
        if name:
            regions.append(name)
    if not regions:
        return  # capacity count but no rentable region named
    accelerator = normalise_accelerator(family)
    if not accelerator:
        return
    vram = _vram_gib_per_gpu(instance_type.get('description'))
    # The GpuInfo column carries the legacy 'Gpus'-WRAPPED shape that
    # sky/catalog/common's DeviceMemoryGiB pass reads
    # (row['Gpus'][0]['MemoryInfo']['SizeInMiB']) — a WELL-FORMED but
    # unwrapped dict raises an UNCAUGHT KeyError there (the Prime lane's
    # live #39 defect; fetch_latitude's _gpu_info_str is the ONE canonical
    # builder so this lane cannot drift from common.py's contract).
    gpu_info = fetch_latitude._gpu_info_str(accelerator, gpu_count, vram)
    for region in regions:
        yield [
            product_name,
            accelerator,
            gpu_count,
            vcpus,
            mem_gib,
            price,
            region,
            gpu_info,
            '',  # no spot tier on MassedCompute
            capacity,
        ]


def iter_rows(inventory: Dict[str, Dict[str, Any]]) -> Iterator[List[Any]]:
    """Yield one CSV row per in-capacity (family, count, region) we rent.

    VARIANT DEDUPE (the exactly-one-SKU contract, live 2026-10-22): the
    vendor's inventory carries product VARIANTS — same card, different
    RAM/storage/billing (``gpu_1x_pro_6000_blackwell_low_storage`` is the
    same 96GB Blackwell with less disk) — and the variant-stripping above
    maps every variant to the SAME family accelerator. Emitting one row
    per product made (RTXPRO6000, 1) resolve to TWO instance types, and
    the runner's SKU pin refuses anything but exactly one ("a claim must
    launch against exactly one SKU at one price" — measured live, the
    first massed E2E claims died at the budget quote). One row per
    (accelerator, count, region): the BASE product (no variant suffix)
    wins; a group with no base falls to the cheapest variant.
    """
    rows: List[List[Any]] = []
    for product_name, product in (inventory or {}).items():
        if not isinstance(product, dict):
            continue
        rows.extend(_product_rows(str(product_name), product))
    groups: Dict[Tuple[Any, Any, str], List[List[Any]]] = {}
    for row in rows:
        groups.setdefault((row[1], row[2], row[6]), []).append(row)
    for group in groups.values():
        if len(group) == 1:
            yield group[0]
            continue
        base = next(
            (r for r in group
             if not any(str(r[0]).endswith(s) for s in _VARIANT_SUFFIXES)),
            None)
        yield base or min(group, key=lambda r: float(r[5]))


def verify_zero_stock(inventory: Dict[str, Dict[str, Any]],
                      rows: Sequence[List[Any]]) -> None:
    """Cross-check a zero-ROW fetch before it is written as zero-stock.

    The ruling (the QuantaCloud ENG-515 semantics): an inventory yielding
    NO catalog rows is contradicted by any GPU product reporting
    ``capacity_available > 0`` — the fetch REFUSES rather than write "no
    capacity" (a filter bug or API drift must never read as a sold-out
    market). CPU products never trigger the refusal: the lane
    deliberately excludes them (no accelerator token), so their capacity
    is real but not rentable here. A genuinely sold-out GPU market (every
    GPU product at zero) is the honest header-only write.
    """
    if any(True for _ in rows):
        return  # rows will be written: nothing to cross-check
    stocked = sorted(
        str(name)
        for name, product in (inventory or {}).items()
        if isinstance(product, dict) and
        _parse_product(str(name)) is not None and _capacity(product) > 0)
    if stocked:
        raise MassedCatalogError(
            f'no rentable catalog rows will be written from {len(inventory or {})} '
            f'inventory product(s), but the GPU products {stocked} report '
            'capacity_available > 0; refusing to write a zero-stock catalog '
            'from a contradictory fetch. Either a filter bug / API drift '
            '(investigate before the lane answers "no capacity") or the '
            'capacity is not renderable as a rentable (product, region) '
            'row — do not silence this by writing the header-only '
            'catalog.')


def fetch_inventory(api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The live GPU inventory (AUTHENTICATED endpoint; the key is
    required — verified live 2026-10-21). An unreachable API raises from
    the client (fail loud), it never yields an empty map."""
    key = api_key or os.environ.get(massed_api.API_KEY_ENV, '').strip()
    return massed_api.MassedClient(key).list_gpu_inventory()


def fetch_images(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """The live bootable images list (AUTHENTICATED endpoint)."""
    key = api_key or os.environ.get(massed_api.API_KEY_ENV, '').strip()
    return massed_api.MassedClient(key).list_images()


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
    parser.add_argument('--api-key-env', default=massed_api.API_KEY_ENV)
    parser.add_argument('--output', default='massed/vms.csv')
    parser.add_argument('--from-fixture',
                        help='read a recorded payload instead of the network')
    args = parser.parse_args(argv)

    if args.from_fixture:
        with open(args.from_fixture, encoding='utf-8') as handle:
            fixture = json.load(handle)
        inventory = fixture['gpu_inventory']
    else:
        # The inventory endpoint is AUTHENTICATED (unlike QuantaCloud's
        # public catalog): an unset key raises MassedAuthError from the
        # client (fail loud), it never yields an empty map that would read
        # as "no capacity".
        api_key = os.environ.get(args.api_key_env, '').strip() or None
        inventory = fetch_inventory(api_key)

    rows = list(iter_rows(inventory))
    verify_zero_stock(inventory, rows)
    count = write_csv(rows, args.output)
    print(f'wrote {count} rows to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
