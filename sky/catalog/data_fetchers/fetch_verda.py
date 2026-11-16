"""Fetch Verda instance types and emit a SkyPilot catalog CSV.

One row per IN-STOCK (instance_type, location) pair the lane can rent,
built from the join of three authenticated endpoints (all measured from
the OpenAPI spec at https://api.verda.com, reviewed 2026-11-16):

* ``GET /v1/instance-availability`` — the ONLY stock source: which
  instance-type codes are in stock per location code. The spec warns
  that a price does NOT mean availability, so a type's rows exist ONLY
  where its code appears in a location's ``availabilities``.
* ``GET /v1/instance-types`` — per-type specs and the AUTHORITATIVE
  ``location_prices`` (per-location ``price_per_hour`` and
  ``spot_price`` STRINGS). The deprecated top-level price fields are
  NEVER read.
* ``GET /v1/locations`` — location code -> NAME mapping.

Rules the join enforces:

* **Currency: USD rows only.** The runner's budget caps are USD, so a
  type whose ``currency`` is not ``usd`` is skipped ENTIRELY (all its
  locations) — non-usd rows would poison every budget comparison. The
  loud skip is recorded in ``skipped``; the live probe will confirm what
  our account returns once credentials exist.
* **Stock without a price is skipped, never fabricated.** A type in
  stock at a location whose ``location_prices`` has no entry for that
  code is skipped with a reason recorded in ``skipped``.
* **Region column is the location NAME** (``"Finland 1"``) — the lane's
  convention: the catalog's Region is what the user sees and the cloud
  module filters by; the provisioner maps the name back to the location
  code via ``GET /v1/locations`` at launch.
* **InstanceType is the Verda code** (``1H100.80S.22V``); AcceleratorName
  is derived from ``model`` (``"H100"`` -> ``H100``).
* **Spot is REAL on this API**: ``spot_price`` per location is emitted
  in SpotPrice (unlike vast, this provider genuinely has a spot tier).

The fetcher FAILS CLOSED on missing credentials: unlike quantacloud's
public catalog endpoints, every Verda endpoint needs the OAuth2 token,
so an unauthenticated fetch must raise, never emit an empty catalog that
would read as "no capacity".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from sky.adaptors import verda as verda_api
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

# The OS image the provisioner deploys (measured from GET /v1/images
# examples; ubuntu-22.04-cuda-12.4-docker-style slug). Only used by the
# provisioner, kept here so the lane's deploy knobs live in one module.
DEFAULT_IMAGE = 'ubuntu-24.04-cuda-12.8-open-docker'


class VerdaCatalogError(RuntimeError):
    """Raised when the fetcher cannot run or the payload is unusable."""


def normalise_accelerator(model: Optional[str]) -> str:
    """The accelerator token from the type's GPU ``model``.

    The provider's ``model`` ("H100") is already the platform-style
    token; unmapped/empty models pass through uppercased so an
    unrecognized GPU stays visible instead of silently vanishing.
    """
    token = str(model or '').strip().upper()
    return token


def location_price(type_row: Dict[str, Any],
                   location_code: str) -> Optional[Tuple[float, float]]:
    """(on-demand, spot) price at ``location_code`` from
    ``location_prices`` ONLY — the deprecated top-level price fields are
    never read (spec: "Deprecated: read location_prices")."""
    for price_row in type_row.get('location_prices') or []:
        if not isinstance(price_row, dict):
            continue
        if str(price_row.get('location_code') or '') != location_code:
            continue
        on_demand = _as_float(price_row.get('price_per_hour'))
        spot = _as_float(price_row.get('spot_price'))
        if on_demand is None:
            return None  # unpriceable row: never quote as free
        return on_demand, spot if spot is not None else on_demand
    return None


def _as_float(raw: Any) -> Optional[float]:
    """Parse the provider's STRING prices ("2.90"); None when unusable."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _type_row(type_row: Dict[str, Any], region_name: str, use_spot: bool,
              price: float, spot_price: float) -> List[Any]:
    gpu = type_row.get('gpu') or {}
    gpu_count = int(gpu.get('number_of_gpus') or 0)
    cpu = type_row.get('cpu') or {}
    memory = type_row.get('memory') or {}
    gpu_memory = type_row.get('gpu_memory') or {}
    accelerator = normalise_accelerator(type_row.get('model'))
    # GpuInfo must be the 'Gpus'-wrapped shape common.py's DeviceMemoryGiB
    # pass reads; fetch_latitude._gpu_info_str is the ONE canonical
    # builder (an unwrapped dict raises an UNCAUGHT KeyError there — the
    # Prime lane hit this live 2026-10-07, fork #39).
    gpu_info = fetch_latitude._gpu_info_str(accelerator, gpu_count,
                                            gpu_memory.get('size_in_gigabytes'))
    row_price = spot_price if use_spot else price
    return [
        str(type_row.get('instance_type') or ''),
        accelerator,
        gpu_count,
        float(cpu.get('number_of_cores') or 0),
        float(memory.get('size_in_gigabytes') or 0),
        row_price,
        region_name,
        gpu_info,
        spot_price,  # the location's spot price; the API HAS a spot tier
    ]


def rows_and_skips(
    instance_types: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    availability: List[Dict[str, Any]],
) -> Tuple[List[List[Any]], Dict[str, List[str]]]:
    """Materialize ``iter_rows`` into (rows, skip-reasons) for writing."""
    rows: List[List[Any]] = []
    skips: Dict[str, List[str]] = {}
    code_to_name = {
        str(loc.get('code')): str(loc.get('name') or loc.get('code'))
        for loc in locations
        if isinstance(loc, dict) and loc.get('code')
    }
    type_by_code = {
        str(t.get('instance_type')): t
        for t in instance_types
        if isinstance(t, dict) and t.get('instance_type')
    }
    # Currency gate first: non-usd types are skipped wholesale. Their
    # in-stock codes must not also be double-reported as "missing from
    # instance-types" below.
    currency_skipped: set = set()
    for code, type_row in list(type_by_code.items()):
        if str(type_row.get('currency') or 'usd') != 'usd':
            skips.setdefault(code, []).append(
                f'non-usd currency {type_row.get("currency")!r} — the '
                'runner budget caps are USD; row skipped (loud by design)')
            del type_by_code[code]
            currency_skipped.add(code)
    for avail_row in availability:
        location_code = str(avail_row.get('location_code') or '')
        if not location_code:
            continue
        region_name = code_to_name.get(location_code, location_code)
        for code in avail_row.get('availabilities') or []:
            code = str(code)
            type_row = type_by_code.get(code)
            if type_row is None:
                if code not in currency_skipped:
                    skips.setdefault(code, []).append(
                        f'{location_code}: in stock but missing from '
                        '/v1/instance-types')
                continue
            gpu = type_row.get('gpu') or {}
            if int(gpu.get('number_of_gpus') or 0) < 1:
                skips.setdefault(code,
                                 []).append(f'{location_code}: CPU-only shape')
                continue
            prices = location_price(type_row, location_code)
            if prices is None:
                # In stock but unpriceable here: skip the row, record why.
                skips.setdefault(code, []).append(
                    f'{location_code}: in stock but no location_prices '
                    'entry; refusing to fabricate a price')
                continue
            price, spot = prices
            rows.append(
                _type_row(type_row,
                          region_name,
                          use_spot=False,
                          price=price,
                          spot_price=spot))
    return rows, skips


def fetch_catalog(
    client: 'verda_api.VerdaClient'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    return (client.list_instance_types(), client.list_locations(),
            client.list_availability())


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
    parser.add_argument('--output', default='verda/vms.csv')
    parser.add_argument('--from-fixture',
                        help='read a recorded payload instead of the network')
    args = parser.parse_args(argv)

    if args.from_fixture:
        with open(args.from_fixture, encoding='utf-8') as handle:
            fixture = json.load(handle)
        instance_types, locations, availability = (fixture['instance_types'],
                                                   fixture['locations'],
                                                   fixture['availability'])
    else:
        # FAIL CLOSED: every catalog endpoint needs the OAuth2 token, so
        # missing credentials must raise here — never an empty catalog
        # that would read as "no capacity" (the quantacloud public-lane
        # difference).
        from sky.provision.verda import utils
        client = utils.client_from_env()
        instance_types, locations, availability = fetch_catalog(client)

    rows, skips = rows_and_skips(instance_types, locations, availability)
    count = write_csv(rows, args.output)
    for code, reasons in sorted(skips.items()):
        print(f'skipped {code}: {"; ".join(reasons)}', file=sys.stderr)
    print(f'wrote {count} rows to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
