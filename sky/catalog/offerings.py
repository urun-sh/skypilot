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

# Vast dedupes sibling offers per (InstanceType, Region, HostingType)
# stub at fetch time (fetch_vast.py), so its rows carry an AvailableUnits
# column with the number of machines collapsed into each row:
# units = sum(AvailableUnits x AcceleratorCount), listings =
# sum(AvailableUnits).
_STOCK_BY_AVAILABLE_UNITS = frozenset({'vast'})

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
    'latitude': STOCK_MODE_IN_STOCK,
    'lambda': STOCK_MODE_IN_STOCK,
    'shadeform': STOCK_MODE_IN_STOCK,
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


def _gpu_offerings(df: 'pd.DataFrame', cloud: str) -> List[Dict[str, Any]]:
    """Per-accelerator aggregation, gpus_only convention of
    common.list_accelerators_impl.
    """
    if 'GpuInfo' in df.columns:
        df = df[~df['GpuInfo'].isna()]
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
