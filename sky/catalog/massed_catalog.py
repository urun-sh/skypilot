"""MassedCompute | Catalog.

Loads instance/pricing data for the MassedCompute provider from the
catalog CSV emitted by ``sky.catalog.data_fetchers.fetch_massed``.

Instance types are the provider's STABLE product names
(``gpu_1x_pro_6000_blackwell`` — the launch API's ``productName``), so
unlike QuantaCloud there is no ephemeral offer UUID to re-resolve: the
catalog token passes through to the launch verbatim. ``Region`` is the
provider's own region token (``us-east-1``, ``us-central-9``, ...) and
prices are the product's ``price_cents_per_hour`` converted to USD. The
fetcher emits rows ONLY for products with live capacity in a live
capacity region, so everything this module can price, the lane can
actually rent.

Modeled on ``sky/catalog/quantacloud_catalog.py``; no deliberate
differences in behavior — a MassedCompute catalog row is as fixed as a
QuantaCloud slug:count token.
"""

import typing
from typing import Dict, List, Optional, Tuple, Union

from sky.adaptors import common as adaptors_common
from sky.catalog import common

if typing.TYPE_CHECKING:
    import pandas as pd

    from sky.clouds import cloud
else:
    pd = adaptors_common.LazyImport('pandas')

_CATALOG_PATH = 'massed/vms.csv'

# Mirror of the CSV columns fetch_massed writes; order matters to
# DictWriter/DictReader round-trips.
_CSV_COLUMNS = [
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'Price',
    'Region',
    'GpuInfo',
    'SpotPrice',
    'AvailableUnits',
]

# (stat stamp of the catalog CSV, materialized frame), published in ONE
# assignment; None until the first load.
_cache = None


def _load_df() -> 'pd.DataFrame':
    """Read and clean the catalog CSV. Caller applies the stat stamp."""
    try:
        df = common.read_catalog(_CATALOG_PATH)
    except FileNotFoundError as exc:
        # FAIL LOUD. An empty frame here is indistinguishable from
        # "MassedCompute has no capacity right now", so an unconfigured
        # controller would silently look like a stocked-out provider and
        # every massed claim would be refused for the wrong reason.
        # Greptile caught this class of bug on the Spheron carry (#338).
        raise RuntimeError(
            f'MassedCompute catalog {_CATALOG_PATH!r} is missing. It is '
            'written by sky.catalog.data_fetchers.fetch_massed '
            '(the authenticated inventory endpoint; the controller primes '
            'it at startup). Refusing to report an empty catalog, which '
            "would read as 'no capacity' rather than 'not configured'."
        ) from exc
    df = df[df['InstanceType'].notna()]
    if 'AcceleratorName' in df.columns:
        df = df[df['AcceleratorName'].notna()]
        df = df.assign(
            AcceleratorName=df['AcceleratorName'].astype(str).str.strip())
    return df.reset_index(drop=True)


def _get_df() -> 'pd.DataFrame':
    """Get the dataframe, re-reading when the CSV on disk changes.

    The catalog CSV is REWRITTEN under this process by
    massed_catalog_refresh (skypilot-controller). Caching the materialized
    frame forever silently froze the catalog for long-lived interpreters
    (the shadeform twin of this file rejected a region the vendor had
    added with "Invalid region", prod-usw2 2026-10-14). The stat
    comparison makes candidate enumeration and launch validation agree.
    """
    global _cache
    stamp = common.catalog_stat(_CATALOG_PATH)
    cache = _cache
    if cache is None or cache[0] != stamp:
        # ONE assignment: a concurrent same-interpreter caller must never
        # see the new stamp beside the old frame (or the reverse) -- a
        # mixed pair would freeze the old frame for exactly as long as the
        # stamp stays current (CodeRabbit, urun-sh/skypilot#48).
        _cache = (stamp, _load_df())
        cache = _cache
    return cache[1]


def _is_not_found_error(err: ValueError) -> bool:
    msg = str(err).lower()
    return 'not found' in msg or 'not supported' in msg


def _call_or_default(func, default):
    try:
        return func()
    except ValueError as err:
        if _is_not_found_error(err):
            return default
        raise


def instance_type_exists(instance_type: str) -> bool:
    return common.instance_type_exists_impl(_get_df(), instance_type)


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if zone is not None:
        raise ValueError(
            f'MassedCompute does not support zones, but got zone {zone!r}.')
    return common.validate_region_zone_impl('massed', _get_df(), region,
                                            zone)


def get_hourly_cost(
    instance_type: str,
    use_spot: bool = False,
    region: Optional[str] = None,
    zone: Optional[str] = None,
) -> float:
    return common.get_hourly_cost_impl(_get_df(), instance_type, use_spot,
                                       region, zone)


def get_vcpus_mem_from_instance_type(
    instance_type: str,) -> Tuple[Optional[float], Optional[float]]:
    return _call_or_default(
        lambda: common.get_vcpus_mem_from_instance_type_impl(
            _get_df(), instance_type),
        (None, None),
    )


def get_default_instance_type(
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    disk_tier: Optional[str] = None,
    local_disk: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    use_spot: bool = False,
    max_hourly_cost: Optional[float] = None,
) -> Optional[str]:
    # MassedCompute products ship fixed included storage; disk tier is not
    # selectable.
    del disk_tier, local_disk
    return _call_or_default(
        lambda: common.get_instance_type_for_cpus_mem_impl(
            _get_df(), cpus, memory, region, zone, use_spot, max_hourly_cost),
        None,
    )


def get_accelerators_from_instance_type(
    instance_type: str,) -> Optional[Dict[str, Union[int, float]]]:
    return _call_or_default(
        lambda: common.get_accelerators_from_instance_type_impl(
            _get_df(), instance_type),
        None,
    )


def get_instance_type_for_accelerator(
    acc_name: str,
    acc_count: int,
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    use_spot: bool = False,
    local_disk: Optional[str] = None,
    region: Optional[str] = None,
    zone: Optional[str] = None,
    max_hourly_cost: Optional[float] = None,
) -> Tuple[Optional[List[str]], List[str]]:
    del local_disk  # unused
    return _call_or_default(
        lambda: common.get_instance_type_for_accelerator_impl(
            df=_get_df(),
            acc_name=acc_name,
            acc_count=acc_count,
            cpus=cpus,
            memory=memory,
            use_spot=use_spot,
            region=region,
            zone=zone,
            max_hourly_cost=max_hourly_cost,
        ),
        (None, []),
    )


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    df = _get_df()
    df_filtered = df[df['InstanceType'] == instance_type]
    return _call_or_default(
        lambda: common.get_region_zones(df_filtered, use_spot), [])


def list_accelerators(
    gpus_only: bool,
    name_filter: Optional[str],
    region_filter: Optional[str],
    quantity_filter: Optional[int],
    case_sensitive: bool = True,
    all_regions: bool = False,
    require_price: bool = True,
) -> Dict[str, List[common.InstanceTypeInfo]]:
    del require_price  # Unused: a catalog row exists only when priced.
    return common.list_accelerators_impl(
        'Massed',
        _get_df(),
        gpus_only,
        name_filter,
        region_filter,
        quantity_filter,
        case_sensitive,
        all_regions,
    )