"""Provider GPU offerings from the local service catalog CSVs.

Aggregates what each GPU provider is currently offering (GPU types,
counts, cheapest prices, regions) straight from the catalog CSVs the
skypilot-controller runner refreshes continuously under the versioned
catalog dir (``~/.sky/catalogs/v8/<cloud>/vms.csv``). Purely local: no
provider APIs or the hosted catalog mirror are contacted, and a missing
CSV is reported as absent rather than fetched — fetching on miss is what
makes ``/list_accelerators`` hang and 504 at the gateway (the GitHub
fetch has no timeout).

This module is imported by sky/catalog/__init__.py at module level, which
in turn is imported by sky/clouds/cloud.py while it is still
initializing; everything under ``sky.*`` is therefore imported lazily
inside functions.
"""
import importlib.util
import os
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

if typing.TYPE_CHECKING:
    import pandas as pd

# Clouds reported by default. Canonical names matching the
# sky/catalog/<name>_catalog.py modules; each has a <cloud>/vms.csv
# following the common catalog schema.
DEFAULT_OFFERINGS_CLOUDS = [
    'aws',
    'gcp',
    'shadeform',
    'spheron',
    'lambda',
    'vast',
    'latitude',
    'quantacloud',
    'primeintellect',
    'ornn',
    'packet',
    'massed',
    'ionet',
]

_CLOUDS_CSV_FILENAME = '{cloud}/vms.csv'

# ---------------------------------------------------------------------------
# Live-stock semantics (an explicit allowlist — NEVER inferred from row
# multiplicity: for most clouds extra rows are pricing/option variants,
# not inventory).
# ---------------------------------------------------------------------------

# These providers' catalog fetchers emit ONE ROW PER LIVE OFFER, so row
# multiplicity within a (AcceleratorName, Region) bucket IS stock:
# units = sum(AcceleratorCount), listings = row count.
_STOCK_BY_ROW_MULTIPLICITY = frozenset({
    'spheron',
    'primeintellect',
    'quantacloud',
    'ornn',
})

# Vast (and now MassedCompute) dedupe sibling offers per
# (InstanceType, Region, HostingType) stub at fetch time (fetch_vast.py;
# fetch_massed.py emits one row per product-region with the product's
# capacity_available), so their rows carry an AvailableUnits column with
# the number of machines collapsed into each row:
# units = sum(AvailableUnits x AcceleratorCount), listings =
# sum(AvailableUnits).
_STOCK_BY_AVAILABLE_UNITS = frozenset({'vast', 'massed'})

# stock_mode published per provider so the dashboard can label the stock
# honestly (legend/tooltip wording differs per tier):
#   'units'    — catalog rows carry live inventory counts; the
#                region_details carry live units/listings.
#   'in_stock' — catalog rows are ALREADY live-filtered at fetch time
#                (only in-stock regions/offers are written); no counts
#                are published, but region PRESENCE means in-stock at
#                last refresh.
#   'catalog'  — static offering catalog, no stock notion (units/listings
#                stay null; region presence = offering coverage only).
# Anything not listed defaults to 'catalog'.
STOCK_MODE_UNITS = 'units'
STOCK_MODE_IN_STOCK = 'in_stock'
STOCK_MODE_CATALOG = 'catalog'
_STOCK_MODES: Dict[str, str] = {
    'vast': STOCK_MODE_UNITS,
    'spheron': STOCK_MODE_UNITS,
    'primeintellect': STOCK_MODE_UNITS,
    'quantacloud': STOCK_MODE_UNITS,
    'ornn': STOCK_MODE_UNITS,
    'massed': STOCK_MODE_UNITS,
    'latitude': STOCK_MODE_IN_STOCK,
    'lambda': STOCK_MODE_IN_STOCK,
    'shadeform': STOCK_MODE_IN_STOCK,
    'ionet': STOCK_MODE_IN_STOCK,
}


def _pd() -> 'pd':
    """The pandas LazyImport proxy (deferred; see module docstring)."""
    # pylint: disable=import-outside-toplevel
    from sky.adaptors import common as adaptors_common
    return adaptors_common.LazyImport('pandas')


def _absent(error: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'present': False,
        'updated_at': None,
        'gpus': None,
    }
    if error is not None:
        result['error'] = error
    return result


def _to_count(value: Any) -> Union[int, float]:
    count = float(value)
    if count.is_integer():
        return int(count)
    return count


def _cheapest(series: Optional['pd.Series']) -> Optional[float]:
    """Cheapest non-null price in the series, None if none available."""
    if series is None:
        return None
    prices = _pd().to_numeric(series, errors='coerce').dropna()
    if prices.empty:
        return None
    return float(prices.min())


def _stock_counts(bucket: 'pd.DataFrame',
                  cloud: str) -> Tuple[Optional[int], Optional[int]]:
    """(units, listings) live stock for one (GPU, Region) bucket.

    units counts live GPU chips available in the bucket, listings the
    distinct rentable offers/machines behind it; both None when the
    cloud has no live-stock notion (catalog coverage only). Stock
    semantics are an explicit allowlist, never inferred from row
    multiplicity.
    """
    pd = _pd()
    name = cloud.lower()
    if name in _STOCK_BY_ROW_MULTIPLICITY:
        # One row per live offer: row multiplicity IS stock.
        listings = int(len(bucket))
        units: Optional[int] = None
        if 'AcceleratorCount' in bucket.columns:
            counts = pd.to_numeric(bucket['AcceleratorCount'],
                                   errors='coerce').dropna()
            if not counts.empty:
                units = int(counts.sum())
        return units, listings
    if name in _STOCK_BY_AVAILABLE_UNITS and 'AvailableUnits' in bucket.columns:
        # Rows carry AvailableUnits (machines collapsed into each row by
        # the fetcher's stub dedup); row multiplicity would undercount.
        machines = pd.to_numeric(bucket['AvailableUnits'],
                                 errors='coerce').dropna()
        if machines.empty:
            return None, None
        listings = int(machines.sum())
        units = None
        if 'AcceleratorCount' in bucket.columns:
            chips = pd.to_numeric(bucket['AcceleratorCount'],
                                  errors='coerce').fillna(0)
            units = int((machines * chips.loc[machines.index]).sum())
        return units, listings
    # Catalog-only (aws, gcp, ...) or already live-filtered providers
    # (latitude, lambda, shadeform): no counts published.
    return None, None


def _region_details(rows: 'pd.DataFrame', cloud: str) -> List[Dict[str, Any]]:
    """Per-region buckets for one accelerator, sorted by region name.

    Each entry carries the region's live stock (units/listings, None
    when the cloud publishes no counts) and the cheapest on-demand /
    spot price within the region (None when the region has no priced
    rows).
    """
    if 'Region' not in rows.columns:
        return []
    rows = rows[rows['Region'].notna()]
    details: List[Dict[str, Any]] = []
    # groupby sorts by region name.
    for region, bucket in rows.groupby(rows['Region'].astype(str)):
        units, listings = _stock_counts(bucket, cloud)
        details.append({
            'region': str(region),
            'units': units,
            'listings': listings,
            'price': _cheapest(bucket.get('Price')),
            'spot_price': _cheapest(bucket.get('SpotPrice')),
        })
    return details


# GCP G4 shapes that carry a FRACTIONAL vGPU slice (1/8, 1/4 and 1/2 of a
# card) but are priced as count-1 catalog rows; only g4-standard-48 is a
# whole card. The catalog deliberately prices every G4 shape ("the
# CATALOG's pricing view, not a launchability claim", test_gcp.py) and
# the RUNNER excludes the slices via _CATALOG_UNTRUSTED_SKUS; the
# offerings aggregation must apply the same rule or the dashboard quotes
# a SLICE as the GPU's price (2026-11-09: the "cheapest RTX PRO 6000"
# showed the 1/8 slice's $0.24 spot while the whole card is $4.50
# on-demand / $1.68 spot — gcp is the only offerings cloud whose
# catalog carries fractional-GPU rows).
_GCP_FRACTIONAL_GPU_INSTANCE_TYPES = frozenset({
    'g4-standard-6',
    'g4-standard-12',
    'g4-standard-24',
})

# The GCP host↔accelerator JOIN (#91's deferred follow-up, ENG-581):
# GCP prices a GPU machine as HOST VM + per-GPU SKUs, and the catalog
# carries the two in SEPARATE rows — GPU rows with an empty
# InstanceType (bare SKU price) and host rows with an empty
# AcceleratorName (host cost). Quoting the bare SKU as the card's
# price underquotes by the whole host: measured live 2026-11-09, the
# us-central1 RTXPRO6000 SKU $1.09565 + the g4-standard-48 host
# $3.40428 = $4.49993/GPU-hr — the runner's pinned truth ($4.4999)
# exactly, while the SKU-only row read $1.10 on-demand / $0.45 spot
# (host spot $1.27068 joins to $1.71778).
#
# The join: a bare-SKU GPU row of accelerator A at count N in a
# (region, zone) takes the same-place host row for A's family shape at
# count N and adds its per-GPU slice (host price / N). No host row →
# the row keeps its SKU-only price (never a fabricated host). Families
# WITHOUT a fixed count↔shape convention stay out entirely:
#   * A4/B200: get_vm_price's special case prices the host row 0 — the
#     B200 SKU IS the whole-box price, nothing to join.
#   * N1 legacy attachable GPUs (V100/P100/T4/…): any n1-standard-*
#     size can host any count — no convention; those rows stay
#     SKU-only (documented here rather than guessed).
_GCP_GPU_HOST_SHAPES = {
    'RTXPRO6000': lambda count: f'g4-standard-{48 * count}',
    'L4': lambda count: f'g2-standard-{12 * count}',
    'A100': lambda count: f'a2-highgpu-{count}g',
    'H100': lambda count: f'a3-highgpu-{count}g',
}


def _join_gcp_host_prices(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Whole-card prices for GCP bare-SKU GPU rows: add the host slice."""
    if df.empty or 'InstanceType' not in df.columns:
        return df
    host_rows = df[df['AcceleratorName'].isna() & df['InstanceType'].notna()]
    if host_rows.empty:
        return df
    host_price = {
        # AvailabilityZone is optional (not every offerings CSV carries
        # it); the key degrades to region-only matching.
        (row.InstanceType, row.Region, getattr(row, 'AvailabilityZone', '')):
            (row.Price, row.SpotPrice) for row in host_rows.itertuples()
    }
    gpu_mask = (df['AcceleratorName'].notna() & df['InstanceType'].isna() &
                df['AcceleratorName'].isin(_GCP_GPU_HOST_SHAPES))
    if not gpu_mask.any():
        return df
    df = df.copy()
    unjoinable = []
    for idx, row in df[gpu_mask].iterrows():
        count = int(row['AcceleratorCount'])
        if count < 1:
            continue
        shape = _GCP_GPU_HOST_SHAPES[row['AcceleratorName']](count)
        host = host_price.get(
            (shape, row['Region'], row.get('AvailabilityZone', '')))
        if host is None:
            # No same-place host row: a bare GPU SKU is not a
            # rentable card (the GPU cannot be bought without its
            # host), so the row is DROPPED, not quoted SKU-only —
            # one unjoined row would win every _cheapest() and
            # re-create the underquote this join exists to fix.
            unjoinable.append(idx)
            continue
        if host[0] == host[0]:  # not NaN
            df.loc[idx, 'Price'] = row['Price'] + host[0] / count
        if host[1] == host[1]:  # not NaN
            df.loc[idx, 'SpotPrice'] = row['SpotPrice'] + host[1] / count
    if unjoinable:
        df = df.drop(index=unjoinable)
    return df


def _gpu_offerings(df: 'pd.DataFrame', cloud: str) -> List[Dict[str, Any]]:
    """Per-accelerator aggregation, gpus_only convention of
    common.list_accelerators_impl.
    """
    if (cloud == 'gcp' and 'InstanceType' in df.columns and not df.empty):
        # The host↔accelerator join (#91's follow-up) runs FIRST: the
        # GpuInfo filter below drops the bare host rows (empty GpuInfo),
        # and the join reads them — bare-SKU GPU rows gain their host
        # slice so the dashboard quotes the WHOLE card (measured:
        # SKU $1.09565 + g4-standard-48 host $3.40428 = $4.49993 = the
        # runner's pinned truth).
        df = _join_gcp_host_prices(df)
    if 'GpuInfo' in df.columns:
        df = df[~df['GpuInfo'].isna()]
    if (cloud == 'gcp' and 'InstanceType' in df.columns and not df.empty):
        # Never quote a fractional G4 slice as a GPU price (see
        # _GCP_FRACTIONAL_GPU_INSTANCE_TYPES); the whole-card rows
        # price the GPU the dashboard actually sells.
        df = df[~df['InstanceType'].isin(_GCP_FRACTIONAL_GPU_INSTANCE_TYPES)]
    if 'AcceleratorName' not in df.columns:
        return []
    df = df[df['AcceleratorName'].notna()]
    gpus: List[Dict[str, Any]] = []
    # groupby sorts by accelerator name.
    for name, rows in df.groupby('AcceleratorName'):
        gpu: Dict[str, Any] = {
            'name': name,
            'counts': [],
            'price': None,
            'spot_price': None,
            'regions': [],
            'region_details': _region_details(rows, cloud),
        }
        if 'AcceleratorCount' in rows.columns:
            gpu['counts'] = sorted({
                _to_count(count) for count in rows['AcceleratorCount'].dropna()
            })
        gpu['price'] = _cheapest(rows.get('Price'))
        gpu['spot_price'] = _cheapest(rows.get('SpotPrice'))
        if 'Region' in rows.columns:
            gpu['regions'] = sorted(set(rows['Region'].dropna().astype(str)))
        gpus.append(gpu)
    return gpus


def _cloud_offerings(cloud: str) -> Dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    from sky.catalog import common

    name = cloud.lower()
    # The stock tier is a property of the PROVIDER, not of the data on
    # disk: the dashboard must be able to label absent/empty catalogs
    # too, so stock_mode is published on every outcome below.
    stock_mode = _STOCK_MODES.get(name, STOCK_MODE_CATALOG)
    try:
        if importlib.util.find_spec(f'sky.catalog.{name}_catalog') is None:
            return {**_absent(error='unknown cloud'), 'stock_mode': stock_mode}
        catalog_path = common.get_catalog_path(
            _CLOUDS_CSV_FILENAME.format(cloud=name))
        # The exists() check MUST gate the read: common.read_catalog on a
        # missing file triggers _update_catalog, which fetches from
        # GitHub with no timeout. Here we read the CSV directly so no
        # fetch path exists at all; a missing CSV is simply reported
        # as absent.
        if not os.path.exists(catalog_path):
            return {**_absent(), 'stock_mode': stock_mode}
        # 'NA' is a live georegion token in vast catalogs (North
        # America); pandas' default NA-string list would silently
        # turn every NA row's Region into NaN and drop it from the
        # per-region stock. Only the EMPTY string is missing data here
        # (numeric columns still coerce through to_numeric).
        df = _pd().read_csv(catalog_path, keep_default_na=False, na_values=[''])
        return {
            'present': True,
            'updated_at': float(os.path.getmtime(catalog_path)),
            'stock_mode': stock_mode,
            'gpus': _gpu_offerings(df, name),
        }
    except Exception as e:  # pylint: disable=broad-except
        return {**_absent(error=str(e)), 'stock_mode': stock_mode}


def get_provider_gpu_offerings(
        clouds: Optional[List[str]] = None) -> Dict[str, Any]:
    """Aggregates current GPU offerings from local provider catalogs.

    Reads the <cloud>/vms.csv catalog of each requested cloud (default:
    DEFAULT_OFFERINGS_CLOUDS) and returns, per cloud, its mtime, its
    stock_mode (units/in_stock/catalog — see _STOCK_MODES) and the
    per-GPU offering rows (distinct accelerator counts, cheapest
    on-demand/spot price, distinct regions, and per-region
    region_details with live units/listings where the provider
    publishes them). Clouds whose catalog CSV is not on disk are
    reported as ``present: false`` — never fetched.

    Returns:
        A dict with a single 'providers' key mapping each requested
        cloud to {'present', 'updated_at', 'stock_mode', 'gpus'} (+
        'error' on failure). Each GPU entry carries 'region_details':
        one row per (GPU, Region) bucket, sorted by region name, with
        'units'/'listings' (live stock; null when the provider has no
        live-stock notion) and the region's cheapest 'price'/
        'spot_price'. See /provider_gpu_offerings in sky/server/server.py.
    """
    if not clouds:
        clouds = DEFAULT_OFFERINGS_CLOUDS
    return {'providers': {cloud: _cloud_offerings(cloud) for cloud in clouds}}
