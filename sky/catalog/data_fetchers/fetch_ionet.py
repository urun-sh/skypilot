"""Fetch io.net (https://ai.io.net) hardware and emit a SkyPilot catalog CSV.

One row per IN-STOCK (config, location) pair the lane can actually rent,
from the authenticated ``GET /enterprise/v1/io-cloud/vmaas/hardware``
(x-api-key header; verified live 2026-11-03):

* **InstanceType is the config's ``deploy_id``** (``8H100.80S.176V``).
  The deploy_id is the deploy API's ``hardware_id`` — stable, and unique
  per hardware config (it repeats across locations: ``L4x2`` appears in
  both FR and PL) — so the catalog token passes through to the deploy
  verbatim and nothing is re-resolved.
* **Stock is the price oracle, not the hardware list.** io.net publishes
  NO unit counts: the hardware row's ``sold_out`` field returns null
  (ignored), and the price endpoint's ``available_replica_count`` is
  CAPPED AT 1 — it is a >=1 availability flag, never a count (treating it
  as units would be dishonest). This lane is therefore ``in_stock``
  tier (like lambda/shadeform/latitude): a row is emitted ONLY for
  configs whose price probe succeeds — "offered right now" — and NO
  units column is written (row multiplicity must never be read as
  stock).
* **Price is the per-VM ``price`` for the WHOLE box** (a 8-GPU config's
  hourly price is for the entire server, not per GPU). A config without
  a usable price is skipped: an unpriceable row would quote as free
  (fail closed, not silent).
* **Region is the lowercase location country code** (``us``, ``fi``,
  ``fr``, ``in``, ``pl``): the controller's residency filter
  exact-matches ``Region`` against budget ``allowedRegions`` tokens, so
  the catalog must emit the lowercase form of the uppercase code the
  deploy API takes back as ``location_ids`` (["US", ...]).
* **AcceleratorName comes from ``_ACCELERATOR_MAP``**, keyed by the
  compacted uppercase GPU ``name`` (every family the live inventory
  carries on 2026-11-03 is mapped: H100, A100 incl. DGX, H200, L4, L40,
  A30, RTX4090, RTXPRO6000 incl. PRO 6000 BLACKWELL, 6000 ADA /
  RTX6000Ada, V100, V100_32G, B200); an unmapped name passes through
  COMPACTED (spaces/underscores removed, uppercased, vendor prefixes
  stripped) so an unrecognized GPU stays visible in the catalog for
  audit instead of silently hiding the offer.
* **A zero-row fetch is REFUSED, never written** (the Ornn/Massed
  honesty ruling): an empty hardware list, or a non-empty list where
  EVERY probe fails, is either "not configured / API drift" or a total
  sold-out market — indistinguishable from a filter bug — so the fetch
  REFUSES rather than write a header-only CSV that would read as
  "no capacity" (a header-only write is honest ONLY for a genuinely
  sold-out market, and with ~50 configs a 100% probe failure has never
  been observed live: 46 of 47 were in stock on 2026-11-03).

Needs ``IO_NET_API_KEY``; without it the client raises from the adaptor
(fail loud) — the fetcher never emits an empty catalog that would read
as "no capacity".

GpuInfo is built with fetch_latitude's canonical builder (the legacy
'Gpus'-wrapped shape that sky/catalog/common's DeviceMemoryGiB pass
reads — row['Gpus'][0]['MemoryInfo']['SizeInMiB']; a well-formed but
unwrapped dict raises an uncaught KeyError there, the Prime lane's
live #39 defect).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from sky.adaptors import ionet as ionet_api
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
    # NO AvailableUnits column: io.net is in_stock tier — the price
    # oracle's available_replica_count is CAPPED AT 1 (an availability
    # flag, never a count), so no units are written and row multiplicity
    # must never be read as stock (see the module docstring).
]

# io.net hardware ``name`` (compacted: uppercased, vendor prefixes
# stripped, spaces/underscores collapsed) -> the ONE accelerator token
# the platform's identities use for that family (see fetch_latitude/
# fetch_massed's maps and _PROVIDER_GPU_IDENTITIES in
# skypilot-controller). Keys are the COMPACTED name; lookup lowercases
# nothing (the compact form is already uppercase).
_ACCELERATOR_MAP = {
    'H100': 'H100',
    'A100': 'A100',
    'DGXA100': 'A100',
    'H200': 'H200',
    'L4': 'L4',
    'A30': 'A30',
    'L40': 'L40',
    'RTX4090': 'RTX4090',
    'RTXPRO6000': 'RTXPRO6000',
    'PRO6000BLACKWELL': 'RTXPRO6000',
    '6000ADA': 'RTX6000ADA',
    'RTX6000ADA': 'RTX6000ADA',
    'V100': 'V100',
    'V10032G': 'V100-32GB',
    'B200': 'B200',
}

# Vendor prefixes stripped before the family lookup (the live names are
# bare like "H100", but the docs show "NVIDIA H100" style strings too).
_VENDOR_PREFIXES = ('NVIDIA ', 'GEFORCE ')

# The sleep between price-oracle probes in the live fetch path: the
# oracle is called once per (config, location) — ~50 calls for the live
# inventory on 2026-11-03 — and a modest pause keeps the fetch under
# the API's rate limits (mirrors the other lanes' pacing).
_PROBE_SLEEP_SECONDS = 0.15


class IonetCatalogError(RuntimeError):
    """Raised when the fetcher cannot run or the payload is unusable."""


def normalise_accelerator(name: Optional[str]) -> str:
    """The io.net GPU ``name`` -> the catalog AcceleratorName.

    Compacts the raw name — uppercase, strip 'NVIDIA '/'GeForce '
    prefixes, collapse spaces/underscores — then maps to the family
    token: ``'PRO 6000 BLACKWELL'`` -> ``RTXPRO6000``,
    ``'V100_32G'`` -> ``V100-32GB``, ``'DGX A100'`` -> ``A100``.
    Unmapped names pass through compacted: an unrecognized GPU is a
    real accelerator nobody requests through our identities, and
    passing it through keeps it visible in the catalog for audit
    instead of silently hiding the offer.
    """
    compacted = str(name or '').strip().upper()
    changed = True
    while changed:
        changed = False
        for prefix in _VENDOR_PREFIXES:
            if compacted.startswith(prefix):
                compacted = compacted[len(prefix):].strip()
                changed = True
    compacted = re.sub(r'[ _]+', '', compacted)
    if not compacted:
        return ''
    return _ACCELERATOR_MAP.get(compacted, compacted)


def _hardware_row(hardware: Dict[str, Any], in_stock: bool) -> Optional[List[Any]]:
    """One hardware config -> one CSV row when in stock, else None.

    Skips (fail closed, never a garbage row): configs with no usable
    price (would quote as free) and configs with no accelerator token
    after normalization.
    """
    if not in_stock:
        return None  # the price probe failed: sold out / not offered
    price_raw = hardware.get('price')
    try:
        price = float(price_raw) if price_raw is not None else 0.0
    except (TypeError, ValueError):
        return None  # unpriceable: skip rather than quote as free
    if price <= 0:
        return None
    gpu_count = hardware.get('num_cards')
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, (int, float)) \
            or int(gpu_count) < 1:
        return None
    gpu_count = int(gpu_count)
    accelerator = normalise_accelerator(hardware.get('name'))
    if not accelerator:
        return None
    deploy_id = str(hardware.get('deploy_id') or '').strip()
    location = str(hardware.get('location') or '').strip()
    if not deploy_id or not location:
        return None  # no stable identity / no rentable location token
    vcpus = float(hardware.get('vcpu') or 0)
    mem_gib = float(hardware.get('memory') or 0)
    vram_raw = hardware.get('vram_per_card')
    try:
        vram = float(vram_raw) if vram_raw is not None else None
    except (TypeError, ValueError):
        vram = None
    # The GpuInfo column carries the legacy 'Gpus'-WRAPPED shape that
    # sky/catalog/common's DeviceMemoryGiB pass reads
    # (row['Gpus'][0]['MemoryInfo']['SizeInMiB']) — fetch_latitude's
    # builder is the ONE canonical emitter so this lane cannot drift
    # from common.py's contract.
    gpu_info = fetch_latitude._gpu_info_str(accelerator, gpu_count, vram)
    return [
        deploy_id,
        accelerator,
        gpu_count,
        vcpus,
        mem_gib,
        price,
        # The deploy API takes uppercase location_ids (["US"]); the
        # catalog's Region token is the lowercase form (the residency
        # filter's token), so lowercase here and uppercase at launch.
        location.lower(),
        gpu_info,
        '',  # no spot tier on io.net
    ]


def iter_rows(
    hardware: List[Dict[str, Any]],
    stock_probe: Callable[[Dict[str, Any]], bool],
) -> Iterator[List[Any]]:
    """Yield one CSV row per IN-STOCK (config, location) we can rent.

    PURE core: the stock oracle is injected as ``stock_probe`` so tests
    can mock availability without the network. The probe receives the
    hardware row and returns True when the config is rentable in that
    row's location (the live path builds it on the adaptor's
    ``price_check`` — a None/absent quote means sold out / not offered,
    and such rows are EXCLUDED: a row the deploy must refuse would let
    the QUOTE admit candidates the LAUNCH cannot serve).
    """
    for row in hardware or []:
        if not isinstance(row, dict):
            continue
        try:
            in_stock = bool(stock_probe(row))
        except Exception:  # pylint: disable=broad-except
            # A probe that blows up is treated as NOT in stock: fail
            # closed, never emit a row on the strength of an error.
            in_stock = False
        emitted = _hardware_row(row, in_stock)
        if emitted is not None:
            yield emitted


def fetch_hardware(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """The live hardware list (AUTHENTICATED endpoint; the key is
    required — verified live 2026-11-03). An unreachable API raises from
    the client (fail loud), it never yields an empty list."""
    key = api_key or os.environ.get(ionet_api.API_KEY_ENV, '').strip()
    return ionet_api.IonetClient(key).list_hardware()


def make_stock_probe(
    client: 'ionet_api.IonetClient',
    pace: bool = True,
) -> Callable[[Dict[str, Any]], bool]:
    """The REAL stock oracle: a config is in stock when its price probe
    succeeds. ``price_check`` returns None for
    "No available instances found for deployment" (HTTP 200, their
    sold-out shape) — None is NOT stock (the in_stock-tier ruling: a row
    exists only when offered right now).

    ``pace`` inserts the modest sleep BETWEEN probe calls (~50 oracle
    calls for the live inventory: keeps the fetch under the API's rate
    limits); the sleep lives in the probe, not in iter_rows, so the
    PURE core stays sleep-free and test-mockable.
    """
    calls = 0

    def probe(hardware: Dict[str, Any]) -> bool:
        nonlocal calls
        calls += 1
        if pace and calls > 1:
            time.sleep(_PROBE_SLEEP_SECONDS)
        return client.price_check(
            hardware_id=str(hardware.get('deploy_id') or ''),
            gpus_per_vm=int(hardware.get('num_cards') or 0),
            location=str(hardware.get('location') or '').upper(),
        ) is not None

    return probe


def verify_nonempty(hardware: List[Dict[str, Any]],
                    rows: List[List[Any]]) -> None:
    """Cross-check the fetch before a zero-row catalog is written.

    The ruling (the Ornn/Massed honesty semantics, adapted): io.net
    publishes no unit counts, so a header-only CSV can ONLY be the
    honest "no capacity" state when every probe against a NON-EMPTY
    hardware list failed. An EMPTY hardware list is never honest
    stock-out (47 configs were live on 2026-11-03) — it is an
    unconfigured key or API drift and must read as "not configured".
    """
    if any(True for _ in rows):
        return  # rows will be written: nothing to cross-check
    count = len(hardware or [])
    if count == 0:
        raise IonetCatalogError(
            'io.net hardware list is empty; refusing to write a zero-row '
            'catalog that would read as "no capacity". An empty list is '
            'not an honest stock-out (47 configs were live on '
            '2026-11-03) — it means the API is unreachable, the key is '
            'unconfigured, or the payload shape drifted. Investigate '
            'before the lane answers "no capacity".')
    raise IonetCatalogError(
        f'no rentable catalog rows will be written from {count} io.net '
        'hardware config(s): EVERY price probe failed. A 100% probe '
        'failure is either API drift / a filter bug or a total sold-out '
        'market — indistinguishable here — so the fetch refuses to write '
        'a header-only catalog. Investigate before the lane answers '
        '"no capacity"; do not silence this by writing the header-only '
        'catalog.')


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
    parser.add_argument('--api-key-env', default=ionet_api.API_KEY_ENV)
    parser.add_argument('--output', default='ionet/vms.csv')
    parser.add_argument('--from-fixture',
                        help='read a recorded payload instead of the network')
    args = parser.parse_args(argv)

    if args.from_fixture:
        with open(args.from_fixture, encoding='utf-8') as handle:
            fixture = json.load(handle)
        hardware = (fixture.get('data') or {}).get('hardware') \
            or fixture.get('hardware') or []
    else:
        # The hardware endpoint is AUTHENTICATED: an unset key raises
        # IonetAuthError from the client (fail loud), it never yields
        # an empty list that would read as "no capacity".
        api_key = os.environ.get(args.api_key_env, '').strip() or None
        hardware = fetch_hardware(api_key)

    # The live probe is BUILT on the adaptor's price_check (None quote
    # = sold out) and paced between calls; iter_rows stays the PURE
    # row-emission core (the same function the tests exercise with a
    # mocked probe).
    # An empty key constructs the client but fails loud at the first
    # authenticated call (the "not configured" layer, not a silent None).
    client = ionet_api.IonetClient(
        os.environ.get(args.api_key_env, '').strip())
    probe = make_stock_probe(client)
    rows = list(iter_rows(hardware, probe))
    verify_nonempty(hardware, rows)
    count = write_csv(rows, args.output)
    print(f'wrote {count} rows to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
