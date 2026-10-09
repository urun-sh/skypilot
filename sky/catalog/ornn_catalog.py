"""Ornn catalog store (reads the CSV fetch_ornn writes).

Loads instance/pricing data for the Ornn provider from the catalog CSV
emitted by ``sky.catalog.data_fetchers.fetch_ornn``.

Instance types are stable tokens ``ornn:<gpu-slug>:<gpus>`` (order and
reservation ids are ephemeral); ``Region`` is the single constant ``ornn``
(a namespace column — the bid API takes no region) and prices are the
whole-box hourly (ask/GPU-hour x count) of in-stock books. The fetcher
emits rows ONLY for books with a live ask, so everything this module can
price, the lane can actually rent.

Modeled on ``sky/catalog/latitude_catalog.py`` (itself modeled on
``spheron_catalog.py``); the deliberate differences are the catalog path
and the token parse helpers the clouds module threads through
``make_deploy_resources_variables`` (the provisioner re-resolves the live
book from the token at launch).
"""

import typing
from typing import Dict, List, Optional, Tuple, Union

from sky.adaptors import common as adaptors_common
from sky.catalog import common
from sky.catalog.data_fetchers import fetch_ornn

if typing.TYPE_CHECKING:
    import pandas as pd

    from sky.clouds import cloud
else:
    pd = adaptors_common.LazyImport("pandas")

_CATALOG_PATH = "ornn/vms.csv"

# Mirror of the CSV columns fetch_ornn writes; order matters to
# DictWriter/DictReader round-trips.
_CSV_COLUMNS = [
    "InstanceType",
    "AcceleratorName",
    "AcceleratorCount",
    "vCPUs",
    "MemoryGiB",
    "Price",
    "Region",
    "GpuInfo",
    "SpotPrice",
]

# The single region token the fetcher writes (a namespace column: the bid
# API takes no region).
REGION = fetch_ornn.REGION

_df = None


def _get_df() -> "pd.DataFrame":
    global _df
    if _df is None:
        try:
            df = common.read_catalog(_CATALOG_PATH)
        except FileNotFoundError as exc:
            # FAIL LOUD. An empty frame here is indistinguishable from
            # "Ornn has no capacity right now", so an unconfigured
            # controller would silently look like a stocked-out provider
            # and every Ornn claim would be refused for the wrong reason.
            # (The Greptile-#338 class; the refresh module in
            # skypilot-controller exists to keep this from ever mattering.)
            raise RuntimeError(
                f"Ornn catalog {_CATALOG_PATH!r} is missing. It is written "
                "by sky.catalog.data_fetchers.fetch_ornn (needs "
                "ORNN_MCP_CREDENTIALS). Refusing to report an empty "
                "catalog, which would read as 'no capacity' rather than "
                "'not configured'.") from exc
        else:
            df = df[df["InstanceType"].notna()]
            if "AcceleratorName" in df.columns:
                df = df[df["AcceleratorName"].notna()]
                df = df.assign(AcceleratorName=df["AcceleratorName"].astype(
                    str).str.strip())
            _df = df.reset_index(drop=True)
    return _df


def _is_not_found_error(err: ValueError) -> bool:
    msg = str(err).lower()
    return "not found" in msg or "not supported" in msg


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
        raise ValueError(f"Ornn does not support zones, got zone {zone!r}")
    return common.validate_region_zone_impl("ornn", _get_df(), region, zone)


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
    # Spot VMs ship the node's fixed storage; disk tier is not selectable.
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
                                       use_spot: bool) -> List["cloud.Region"]:
    df = _get_df()
    df_filtered = df[df["InstanceType"] == instance_type]
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
        "Ornn",
        _get_df(),
        gpus_only,
        name_filter,
        region_filter,
        quantity_filter,
        case_sensitive,
        all_regions,
    )


# -- token helpers ---------------------------------------------------------


def gpu_slug_from_instance_type(instance_type: str) -> str:
    """The bid's ``gpuSlug`` parsed from the ``ornn:<slug>:<gpus>`` token."""
    return fetch_ornn.gpu_slug_from_instance_type(instance_type)


def gpu_count_from_instance_type(instance_type: str) -> int:
    """The bid's GPU count parsed from the ``ornn:<slug>:<gpus>`` token."""
    return fetch_ornn.gpu_count_from_instance_type(instance_type)
