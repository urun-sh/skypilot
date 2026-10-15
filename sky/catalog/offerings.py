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
from typing import Any, Dict, List, Optional, Union

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


def _gpu_offerings(df: 'pd.DataFrame') -> List[Dict[str, Any]]:
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
    try:
        if importlib.util.find_spec(f'sky.catalog.{name}_catalog') is None:
            return _absent(error='unknown cloud')
        catalog_path = common.get_catalog_path(
            _CLOUDS_CSV_FILENAME.format(cloud=name))
        # The exists() check MUST gate the read: common.read_catalog on a
        # missing file triggers _update_catalog, which fetches from
        # GitHub with no timeout. Here we read the CSV directly so no
        # fetch path exists at all; a missing CSV is simply reported
        # as absent.
        if not os.path.exists(catalog_path):
            return _absent()
        df = _pd().read_csv(catalog_path)
        return {
            'present': True,
            'updated_at': float(os.path.getmtime(catalog_path)),
            'gpus': _gpu_offerings(df),
        }
    except Exception as e:  # pylint: disable=broad-except
        return _absent(error=str(e))


def get_provider_gpu_offerings(
        clouds: Optional[List[str]] = None) -> Dict[str, Any]:
    """Aggregates current GPU offerings from local provider catalogs.

    Reads the <cloud>/vms.csv catalog of each requested cloud (default:
    DEFAULT_OFFERINGS_CLOUDS) and returns, per cloud, its mtime and the
    per-GPU offering rows (distinct accelerator counts, cheapest
    on-demand/spot price, distinct regions). Clouds whose catalog CSV is
    not on disk are reported as ``present: false`` — never fetched.

    Returns:
        A dict with a single 'providers' key mapping each requested
        cloud to {'present', 'updated_at', 'gpus'} (+ 'error' on
        failure). See /provider_gpu_offerings in sky/server/server.py.
    """
    if not clouds:
        clouds = DEFAULT_OFFERINGS_CLOUDS
    return {'providers': {cloud: _cloud_offerings(cloud) for cloud in clouds}}
