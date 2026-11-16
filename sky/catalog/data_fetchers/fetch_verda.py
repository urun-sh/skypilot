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
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from sky.catalog.data_fetchers import fetch_latitude
from sky.provision.verda import utils

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


def location_price(
        type_row: Dict[str, Any],
        location_code: str) -> Optional[Tuple[float, Optional[float]]]:
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
        return on_demand, spot  # spot may be None: never fabricate it
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


def _type_row(type_row: Dict[str, Any], region_name: str,
              on_demand_price: Optional[Union[float, str]],
              spot_price: Optional[Union[float, str]]) -> List[Any]:
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
    # pylint: disable=protected-access
    # The same access the sibling fetchers make (see fetch_packet.py:
    # _gpu_info_str is the ONE canonical GpuInfo builder; the vet tier
    # flagged a drifting inline copy on the quantacloud carry).
    gpu_info = fetch_latitude._gpu_info_str(accelerator, gpu_count,
                                            gpu_memory.get('size_in_gigabytes'))
    # A tier's price is set ONLY where that tier reported stock; the
    # other tier's cell stays EMPTY (never the other tier's price).
    return [
        str(type_row.get('instance_type') or ''),
        accelerator,
        gpu_count,
        float(cpu.get('number_of_cores') or 0),
        float(memory.get('size_in_gigabytes') or 0),
        on_demand_price if on_demand_price is not None else '',
        region_name,
        gpu_info,
        spot_price if spot_price is not None else '',
    ]


def rows_and_skips(
    instance_types: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    availability: List[Dict[str, Any]],
    availability_spot: List[Dict[str, Any]],
) -> Tuple[List[List[Any]], Dict[str, List[str]]]:
    """Materialize the tier-aware join into (rows, skip-reasons).

    ``availability`` is the ON-DEMAND tier's stock (is_spot=false) and
    ``availability_spot`` the SPOT tier's (is_spot=true): the endpoint
    checks one tier per call, so a pair is priced per tier ONLY where
    that tier's list reports it in stock (CodeRabbit PR #103: the
    single default fetch established only on-demand stock while rows
    carried SpotPrice).
    """
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
    for code, row in list(type_by_code.items()):
        if str(row.get('currency') or 'usd') != 'usd':
            skips.setdefault(code, []).append(
                f'non-usd currency {row.get("currency")!r} — the '
                'runner budget caps are USD; row skipped (loud by design)')
            del type_by_code[code]
            currency_skipped.add(code)

    def _avail_pairs(availability_rows: List[Dict[str, Any]]) -> set:
        pairs = set()
        for avail_row in availability_rows:
            location_code = str(avail_row.get('location_code') or '')
            if not location_code:
                continue
            for code in avail_row.get('availabilities') or []:
                pairs.add((location_code, str(code)))
        return pairs

    on_demand_pairs = _avail_pairs(availability)
    spot_pairs = _avail_pairs(availability_spot)
    # Union of both tiers' in-stock pairs; region name comes from the
    # location code.
    pairs = sorted(on_demand_pairs | spot_pairs)
    for location_code, code in pairs:
        region_name = code_to_name.get(location_code, location_code)
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
        # Price a tier ONLY where that tier reports stock; a tier
        # without stock gets an EMPTY cell (never the other tier's
        # price: quoting spot stock as on-demand — or vice versa — is
        # exactly the bug class this join exists to prevent).
        on_demand_price = price if (location_code,
                                    code) in on_demand_pairs else ''
        spot_price = spot if (location_code, code) in spot_pairs else ''
        if on_demand_price == '' and spot_price == '':
            # Unreachable (every pair came from a tier list) but kept
            # fail-closed: never emit a row with NO price at all.
            skips.setdefault(
                code,
                []).append(f'{location_code}: no tier stock for either price')
            continue
        rows.append(
            _type_row(type_row,
                      region_name,
                      on_demand_price=on_demand_price,
                      spot_price=spot_price))
    return rows, skips


def fetch_catalog(
    client: 'utils.VerdaClient'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
           List[Dict[str, Any]]]:
    """(instance_types, locations, on-demand availability, spot
    availability): ONE tier per availability call (see
    ``VerdaClient.list_availability``)."""
    return (client.list_instance_types(), client.list_locations(),
            client.list_availability(), client.list_availability(is_spot=True))


def write_csv(rows: Iterable[List[Any]], out_path: str) -> int:
    """Write the catalog CSV ATOMICALLY (tempfile + os.replace): the
    catalog reader (catalog/common.py) takes NO lock against this
    writer, so a plain 'w' truncate would expose an empty/partial
    snapshot to a concurrent lookup — the fetch_ornn twin's pattern
    (CodeRabbit PR #103)."""
    written = 0
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(out_path) or '.',
                                    prefix='.verda-catalog-',
                                    suffix='.csv')
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for row in rows:
                writer.writerow(row)
                written += 1
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
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
        availability_spot = fixture.get('availability_spot', [])
    else:
        # FAIL CLOSED: every catalog endpoint needs the OAuth2 token, so
        # missing credentials must raise here — never an empty catalog
        # that would read as "no capacity" (the quantacloud public-lane
        # difference). The utils import is hoisted to module top like
        # the vast/mithril twins (fetch_vast.py).
        client = utils.client_from_env()
        instance_types, locations, availability, availability_spot = (
            fetch_catalog(client))

    rows, skips = rows_and_skips(instance_types, locations, availability,
                                 availability_spot)
    count = write_csv(rows, args.output)
    for code, reasons in sorted(skips.items()):
        print(f'skipped {code}: {"; ".join(reasons)}', file=sys.stderr)
    print(f'wrote {count} rows to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
