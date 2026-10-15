import os
import tempfile
import time
from unittest import mock

import numpy as np
import orjson
import pandas as pd
import pytest

from sky import catalog
from sky.catalog import common as catalog_common
from sky.catalog import offerings as catalog_offerings
from sky.utils import annotations


def test_rtxpro6000_in_common_gpus():
    # RTXPRO6000 (NVIDIA RTX PRO 6000 Blackwell) must appear in the common GPU
    # list so that `sky show-gpus` surfaces it across clouds. Naming matches
    # the AWS and RunPod catalogs (no hyphens), not GCP's `nvidia-rtx-pro-6000`.
    assert 'RTXPRO6000' in catalog.get_common_gpus()


@mock.patch('sky.catalog.common.requests.get')
def test_read_catalog_triggers_update_on_stale_file(mock_get):
    """Test that read_catalog (and the LazyDataFrame it returns)
    does an update when the catalog file is stale, and that
    it's cached for the duration of the request."""
    DUMMY_CSV = 'col1,col2\n1,2\n3,4\n'
    NEW_DUMMY_CSV = 'col1,col2\n5,6\n7,8\n'

    class DummyResponse:

        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code != 200:
                raise Exception('HTTP error')

    mock_get.return_value = DummyResponse(DUMMY_CSV)

    # Create a random file name.
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        filename = os.path.join(f'gcp/{os.path.basename(tmp.name)}.csv')

    try:
        df = catalog_common.read_catalog(filename, pull_frequency_hours=1)
        # Force the CSV to be written to disk.
        df.head()

        # The file on disk and the DataFrame should match DUMMY_CSV.
        abs_catalog_path = catalog_common.get_catalog_path(filename)
        assert os.path.exists(abs_catalog_path)
        with open(abs_catalog_path) as f:
            content_on_disk = f.read()
        assert content_on_disk == DUMMY_CSV
        pd.testing.assert_frame_equal(df._df, pd.read_csv(abs_catalog_path))

        # Modify the file's mtime to be 2 hours ago.
        new_time = time.time() - 60 * 60 * 2
        os.utime(abs_catalog_path, (new_time, new_time))

        # Patch requests.get again to return NEW_DUMMY_CSV.
        mock_get.return_value = DummyResponse(NEW_DUMMY_CSV)

        # We haven't cleared annotations.FUNCTIONS_NEED_RELOAD_CACHE,
        # so _load_df should still be cached and update_if_stale_func
        # should not be called, i.e. the file on disk and
        # DataFrame should not be updated.
        df.head()
        pd.testing.assert_frame_equal(df._df, pd.read_csv(abs_catalog_path))

        # Clear the cache.
        annotations.clear_request_level_cache()

        # Now update_if_stale_func should be called and should trigger a new fetch.
        df.head()
        # The file and DataFrame should match NEW_DUMMY_CSV.
        with open(abs_catalog_path) as f:
            content_on_disk = f.read()
        assert content_on_disk == NEW_DUMMY_CSV
        pd.testing.assert_frame_equal(df._df, pd.read_csv(abs_catalog_path))
    finally:
        if os.path.exists(abs_catalog_path):
            os.remove(abs_catalog_path)
        meta_path = os.path.join(catalog_common._ABSOLUTE_VERSIONED_CATALOG_DIR,
                                 '.meta', filename + '.md5')
        if os.path.exists(meta_path):
            os.remove(meta_path)
        lock_path = os.path.join(catalog_common._ABSOLUTE_VERSIONED_CATALOG_DIR,
                                 '.meta', filename + '.lock')
        if os.path.exists(lock_path):
            os.remove(lock_path)


@pytest.mark.parametrize(
    "cpus, memory, region, zone, expected",
    [
        ('4', '16', None, None, 'a'),  # Exact match
        ('4+', '16+', None, None, 'a'),  # At least match, cheapest
        ('16', '128', None, None, None),  # No match
        ('1+', None, 'asia-southeast1', None, 'd'),  # Region filtering
        ('1+', None, 'us-west1', 'us-west1-b', 'b'),  # Zone filtering
        ('1+', None, 'us-west1', 'us-west1-c',
         None),  # Zone filtering, no match
        # Regression test for https://github.com/skypilot-org/skypilot/pull/6293:
        # b is cheaper but only available in us-west1-b; so c is chosen.
        ('8', '32', 'us-west1', 'us-west1-a', 'c'),
    ])
def test_get_instance_type_for_cpus_mem_impl_with_az(cpus, memory, region, zone,
                                                     expected):
    """Test get_instance_type_for_cpus_mem_impl with a DataFrame that includes AvailabilityZone."""
    df = pd.DataFrame([
        {
            'InstanceType': 'a',
            'vCPUs': 4,
            'MemoryGiB': 16,
            'Price': 1.0,
            'Region': 'us-west1',
            'AvailabilityZone': 'us-west1-a'
        },
        {
            'InstanceType': 'b',
            'vCPUs': 8,
            'MemoryGiB': 32,
            'Price': 2.0,
            'Region': 'us-west1',
            'AvailabilityZone': 'us-west1-b'
        },
        {
            'InstanceType': 'c',
            'vCPUs': 8,
            'MemoryGiB': 32,
            'Price': 5.0,
            'Region': 'us-west1',
            'AvailabilityZone': 'us-west1-a'
        },
        {
            'InstanceType': 'd',
            'vCPUs': 8,
            'MemoryGiB': 32,
            'Price': 3.0,
            'Region': 'asia-southeast1',
            'AvailabilityZone': 'asia-southeast1-a'
        },
    ])
    result = catalog_common.get_instance_type_for_cpus_mem_impl(
        df, cpus=cpus, memory_gb_or_ratio=memory, region=region, zone=zone)
    assert result == expected


@pytest.mark.parametrize(
    "cpus, memory, region, expected",
    [
        ('4', '16', None, 'a'),  # Exact match
        ('1+', None, None, 'a'),  # At least match, cheapest
        ('8+', '32+', None, 'c'),  # At least match, higher req
        ('16', '128', None, None),  # No match
        ('1+', None, 'asia-southeast1', 'b'),  # Region filtering, cheapest
        ('1+', None, 'europe-west1', None),  # Region filtering, no match
    ])
def test_get_instance_type_for_cpus_mem_impl_no_az(cpus, memory, region,
                                                   expected):
    """Test get_instance_type_for_cpus_mem_impl with a DataFrame that does not include AvailabilityZone."""
    df = pd.DataFrame([
        {
            'InstanceType': 'a',
            'vCPUs': 4,
            'MemoryGiB': 16,
            'Price': 1.0,
            'Region': 'us-east1'
        },
        {
            'InstanceType': 'b',
            'vCPUs': 4,
            'MemoryGiB': 16,
            'Price': 3.0,
            'Region': 'asia-southeast1'
        },
        {
            'InstanceType': 'c',
            'vCPUs': 8,
            'MemoryGiB': 32,
            'Price': 5.0,
            'Region': 'asia-southeast1'
        },
    ])
    result = catalog_common.get_instance_type_for_cpus_mem_impl(
        df, cpus=cpus, memory_gb_or_ratio=memory, region=region)
    assert result == expected


def test_get_hourly_cost_returns_python_float():
    """Test that get_hourly_cost_impl returns Python float, not numpy.float64.

    This is a regression test for GitHub issue #7969 where numpy.float64
    values couldn't be serialized by orjson in the API server.
    """
    df = pd.DataFrame([
        {
            'InstanceType': 'test-instance',
            'Price': np.float64(1.5),
            'SpotPrice': np.float64(0.5),
            'Region': 'us-west1',
            'AvailabilityZone': 'us-west1-a'
        },
    ])

    # Test on-demand pricing
    cost = catalog_common.get_hourly_cost_impl(df,
                                               'test-instance',
                                               use_spot=False,
                                               region=None,
                                               zone=None)
    assert isinstance(cost, float)
    assert not isinstance(cost, np.floating)
    assert type(cost) == float

    # Test spot pricing
    spot_cost = catalog_common.get_hourly_cost_impl(df,
                                                    'test-instance',
                                                    use_spot=True,
                                                    region=None,
                                                    zone=None)
    assert isinstance(spot_cost, float)
    assert not isinstance(spot_cost, np.floating)
    assert type(spot_cost) == float


def test_catalog_prices_are_json_serializable():
    """Test that catalog prices can be serialized with orjson.

    This is a regression test for GitHub issue #7969 where the API server
    failed to serialize cost_report responses containing numpy.float64 values.
    """
    df = pd.DataFrame([
        {
            'InstanceType': 'test-instance',
            'Price': np.float64(2.5),
            'SpotPrice': np.float64(1.0),
            'Region': 'us-west1',
        },
    ])

    cost = catalog_common.get_hourly_cost_impl(df,
                                               'test-instance',
                                               use_spot=False,
                                               region=None,
                                               zone=None)

    # Should serialize without TypeError
    serialized = orjson.dumps(cost)
    assert serialized == b'2.5'

    # Should also work in a dict (simulating API response)
    response = {'total_cost': cost}
    serialized_dict = orjson.dumps(response)
    assert orjson.loads(serialized_dict) == {'total_cost': 2.5}


# Synthetic catalog DataFrame for local disk tests.
# Mirrors real AWS patterns:
#   m5.large    - no local disk         ($0.10)
#   d2.large    - SSD, no NVMe, 500 GB  ($0.14)
#   i3.large    - SSD + NVMe, 475 GB    ($0.16)
#   i3.2xlarge  - SSD + NVMe, 1900 GB   ($0.62)
_LOCAL_DISK_DF = pd.DataFrame([
    {
        'InstanceType': 'm5.large',
        'vCPUs': 2,
        'MemoryGiB': 8,
        'Price': 0.10,
        'Region': 'us-east-1',
        'AvailabilityZone': 'us-east-1a',
        'LocalDiskType': float('nan'),
        'LocalDiskSize': float('nan'),
        'LocalDiskCount': float('nan'),
        'NVMeSupported': False,
    },
    {
        'InstanceType': 'i3.large',
        'vCPUs': 2,
        'MemoryGiB': 15.25,
        'Price': 0.16,
        'Region': 'us-east-1',
        'AvailabilityZone': 'us-east-1a',
        'LocalDiskType': 'ssd',
        'LocalDiskSize': 475.0,
        'LocalDiskCount': 1,
        'NVMeSupported': True,
    },
    {
        'InstanceType': 'i3.2xlarge',
        'vCPUs': 8,
        'MemoryGiB': 61,
        'Price': 0.62,
        'Region': 'us-east-1',
        'AvailabilityZone': 'us-east-1a',
        'LocalDiskType': 'ssd',
        'LocalDiskSize': 950.0,
        'LocalDiskCount': 2,
        'NVMeSupported': True,
    },
    {
        'InstanceType': 'd2.large',
        'vCPUs': 2,
        'MemoryGiB': 15.25,
        'Price': 0.14,
        'Region': 'us-east-1',
        'AvailabilityZone': 'us-east-1a',
        'LocalDiskType': 'ssd',
        'LocalDiskSize': 500.0,
        'LocalDiskCount': 1,
        'NVMeSupported': False,
    },
])


@pytest.mark.parametrize(
    'local_disk, expected',
    [
        # NVMe at-least: cheapest NVMe with total >= 500 is i3.2xlarge
        # (i3.large only has 475)
        ('nvme:500+', 'i3.2xlarge'),
        # NVMe at-least: only i3.2xlarge has total >= 1500
        ('nvme:1500+', 'i3.2xlarge'),
        # NVMe at-least: i3.large (475 >= 100) is cheapest NVMe
        ('nvme:100+', 'i3.large'),
        # SSD at-least: NVMe instances also qualify; d2.large ($0.14,
        # 500 GB) is cheapest SSD with >= 400
        ('ssd:400+', 'd2.large'),
        # NVMe exact: only i3.large has total ~475
        ('nvme:475', 'i3.large'),
        # NVMe exact: no instance has exactly 300 GB NVMe
        ('nvme:300', None),
        # SSD exact: d2.large has 500 GB total (non-NVMe SSD)
        ('ssd:500', 'd2.large'),
        # No local disk: cheapest instance overall (m5.large)
        (None, 'm5.large'),
        # Size too large: nothing has >= 5000
        ('nvme:5000+', None),
    ],
)
def test_filter_with_local_disk(local_disk, expected):
    """Test that filter_with_local_disk + instance selection picks the
    correct (cheapest) instance satisfying local disk requirements."""
    filtered = catalog_common.filter_with_local_disk(_LOCAL_DISK_DF.copy(),
                                                     local_disk)
    result = catalog_common.get_instance_type_for_cpus_mem_impl(
        filtered, cpus='1+', memory_gb_or_ratio=None, region=None)
    assert result == expected


@pytest.mark.parametrize(
    'local_disk, expected_instances',
    [
        # NVMe filters out non-NVMe and non-SSD instances
        ('nvme:100+', ['i3.2xlarge', 'i3.large']),
        # SSD includes NVMe instances too (NVMe is a superset of SSD)
        ('ssd:100+', ['d2.large', 'i3.2xlarge', 'i3.large']),
        # None returns all instances (no filtering)
        (None, ['d2.large', 'i3.2xlarge', 'i3.large', 'm5.large']),
    ],
)
def test_filter_with_local_disk_instance_sets(local_disk, expected_instances):
    """Test that filter_with_local_disk returns the correct candidate set."""
    filtered = catalog_common.filter_with_local_disk(_LOCAL_DISK_DF.copy(),
                                                     local_disk)
    assert sorted(
        filtered['InstanceType'].tolist()) == sorted(expected_instances)


# ---------------------------------------------------------------------------
# Config-based pricing tests (Kubernetes / Slurm)
# ---------------------------------------------------------------------------

# -- get_hourly_cost_from_pricing (common function, dict-based) ------------


def test_cpu_only_instance_uses_cpu_memory_pricing():
    """CPU-only: 4CPU--16GB with rates 0.05/0.01 -> $0.36/hr."""
    pricing = {'cpu': 0.05, 'memory': 0.01}
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=4,
                                                       memory=16,
                                                       accelerator_name=None,
                                                       accelerator_count=None)
    assert cost == pytest.approx(4 * 0.05 + 16 * 0.01)


def test_gpu_instance_uses_accelerator_pricing_only():
    """GPU instance uses ONLY accelerator rate; cpu/memory ignored."""
    pricing = {
        'cpu': 0.05,
        'memory': 0.01,
        'accelerators': {
            'A100': 3.50,
        },
    }
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=8,
                                                       memory=64,
                                                       accelerator_name='A100',
                                                       accelerator_count=2)
    # All-in accelerator pricing: 2 * 3.50 = 7.00 (NOT 8*0.05+64*0.01+7.00)
    assert cost == pytest.approx(2 * 3.50)


def test_accelerator_name_case_insensitive():
    """Accelerator lookup should be case-insensitive."""
    pricing = {'accelerators': {'A100': 3.50}}
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=0,
                                                       memory=0,
                                                       accelerator_name='a100',
                                                       accelerator_count=1)
    assert cost == pytest.approx(3.50)


def test_unknown_accelerator_returns_zero():
    """GPU instance with unconfigured accelerator returns $0.00."""
    pricing = {'cpu': 0.05, 'memory': 0.01, 'accelerators': {'A100': 3.50}}
    cost = catalog_common.get_hourly_cost_from_pricing(
        pricing,
        cpus=4,
        memory=16,
        accelerator_name='TPUv5e',
        accelerator_count=4)
    assert cost == 0.0


def test_empty_pricing_returns_zero():
    """Empty pricing dict returns 0.0."""
    cost = catalog_common.get_hourly_cost_from_pricing({},
                                                       cpus=4,
                                                       memory=16,
                                                       accelerator_name='A100',
                                                       accelerator_count=4)
    assert cost == 0.0


# -- merge_pricing_dicts ---------------------------------------------------


def test_merge_pricing_dicts_partial_override():
    """Override only cpu; memory and accelerators inherited."""
    base = {'cpu': 0.04, 'memory': 0.01, 'accelerators': {'V100': 2.50}}
    override = {'cpu': 0.06}
    merged = catalog_common.merge_pricing_dicts(base, override)
    assert merged == {
        'cpu': 0.06,
        'memory': 0.01,
        'accelerators': {
            'V100': 2.50
        },
    }


def test_merge_pricing_dicts_accelerator_addition():
    """Override adds A100; V100 inherited from base."""
    base = {'cpu': 0.04, 'accelerators': {'V100': 2.50}}
    override = {'accelerators': {'A100': 4.00}}
    merged = catalog_common.merge_pricing_dicts(base, override)
    assert merged['accelerators'] == {'V100': 2.50, 'A100': 4.00}
    assert merged['cpu'] == 0.04


def test_merge_pricing_dicts_does_not_mutate_base():
    """Merge must not modify the original dicts."""
    base = {'cpu': 0.04, 'accelerators': {'V100': 2.50}}
    override = {'cpu': 0.06, 'accelerators': {'A100': 4.00}}
    catalog_common.merge_pricing_dicts(base, override)
    assert base == {'cpu': 0.04, 'accelerators': {'V100': 2.50}}


# -- K8s config resolution (kubernetes_catalog._get_pricing) ---------------

# Kubernetes pricing config: cloud-level default with A100 and H100.
_K8S_CONFIG = {
    'kubernetes': {
        'pricing': {
            'cpu': 0.05,
            'memory': 0.01,
            'accelerators': {
                'A100': 3.50,
                'H100': 5.00,
            },
        },
        'context_configs': {
            'expensive-ctx': {
                'pricing': {
                    'cpu': 0.08,
                    'accelerators': {
                        'A100': 4.00,
                    },
                },
            },
        },
    },
}


@mock.patch('sky.skypilot_config.get_nested')
def test_k8s_context_override_inherits_cloud_defaults(mock_nested):
    """Context override inherits missing keys from cloud-level via merge."""
    mock_nested.side_effect = _mock_get_nested(_K8S_CONFIG)
    from sky.catalog import kubernetes_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = kubernetes_catalog._get_pricing('expensive-ctx')
    # cpu overridden to 0.08, memory inherited 0.01
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=4,
                                                       memory=16,
                                                       accelerator_name=None,
                                                       accelerator_count=None)
    assert cost == pytest.approx(4 * 0.08 + 16 * 0.01)


@mock.patch('sky.skypilot_config.get_nested')
def test_k8s_context_override_accelerator(mock_nested):
    """Context override A100 at $4.00/GPU, H100 inherited at $5.00/GPU."""
    mock_nested.side_effect = _mock_get_nested(_K8S_CONFIG)
    from sky.catalog import kubernetes_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = kubernetes_catalog._get_pricing('expensive-ctx')
    cost_a100 = catalog_common.get_hourly_cost_from_pricing(
        pricing, cpus=0, memory=0, accelerator_name='A100', accelerator_count=1)
    assert cost_a100 == pytest.approx(4.00)
    cost_h100 = catalog_common.get_hourly_cost_from_pricing(
        pricing, cpus=0, memory=0, accelerator_name='H100', accelerator_count=2)
    assert cost_h100 == pytest.approx(2 * 5.00)


@mock.patch('sky.skypilot_config.get_nested')
def test_k8s_no_pricing_config_returns_zero(mock_nested):
    """No pricing config at all returns 0.0."""
    mock_nested.side_effect = _mock_get_nested({})
    from sky.catalog import kubernetes_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = kubernetes_catalog._get_pricing('some-ctx')
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=4,
                                                       memory=16,
                                                       accelerator_name='A100',
                                                       accelerator_count=4)
    assert cost == 0.0


# -- Slurm config resolution (slurm_catalog._get_pricing) -----------------

# Slurm config with partial overrides at each level to test merging:
# - Cloud-level: cpu=0.04, memory=0.01, V100=2.50
# - my-slurm cluster: overrides only cpu (0.06) and A100 (4.00);
#     memory and V100 should be inherited from cloud-level.
# - high-pri partition: overrides only A100 (5.00);
#     cpu, memory, V100 should be inherited from merged cluster level.
_SLURM_CONFIG = {
    'slurm': {
        'pricing': {
            'cpu': 0.04,
            'memory': 0.01,
            'accelerators': {
                'V100': 2.50,
            },
        },
        'cluster_configs': {
            'my-slurm': {
                'pricing': {
                    'cpu': 0.06,
                    'accelerators': {
                        'A100': 4.00,
                    },
                },
                'partition_configs': {
                    'high-pri': {
                        'pricing': {
                            'accelerators': {
                                'A100': 5.00,
                            },
                        },
                    },
                },
            },
        },
    },
}


def _mock_get_nested(config):
    """Create a mock for skypilot_config.get_nested."""

    def _get(keys, default_value, override_configs=None):
        del override_configs
        obj = config
        for key in keys:
            if not isinstance(obj, dict) or key not in obj:
                return default_value
            obj = obj[key]
        return obj

    return _get


@mock.patch('sky.skypilot_config.get_nested')
def test_slurm_cluster_inherits_cloud_defaults(mock_nested):
    """Cluster overrides cpu and adds A100; memory and V100 inherited."""
    mock_nested.side_effect = _mock_get_nested(_SLURM_CONFIG)
    from sky.catalog import slurm_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = slurm_catalog._get_pricing(region='my-slurm')
    # cpu overridden to 0.06, memory inherited 0.01
    assert pricing['cpu'] == 0.06
    assert pricing['memory'] == 0.01
    # V100 inherited from cloud, A100 added by cluster
    assert pricing['accelerators']['V100'] == 2.50
    assert pricing['accelerators']['A100'] == 4.00


@mock.patch('sky.skypilot_config.get_nested')
def test_slurm_partition_inherits_cluster_and_cloud(mock_nested):
    """Partition overrides only A100; cpu, memory, V100 inherited."""
    mock_nested.side_effect = _mock_get_nested(_SLURM_CONFIG)
    from sky.catalog import slurm_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = slurm_catalog._get_pricing(region='my-slurm', zone='high-pri')
    # cpu from cluster (0.06), memory from cloud (0.01)
    assert pricing['cpu'] == 0.06
    assert pricing['memory'] == 0.01
    # A100 overridden by partition to 5.00, V100 inherited from cloud
    assert pricing['accelerators']['A100'] == 5.00
    assert pricing['accelerators']['V100'] == 2.50


@mock.patch('sky.skypilot_config.get_nested')
def test_slurm_fallback_to_cluster_when_partition_unmatched(mock_nested):
    """Unknown partition falls back to merged cluster pricing."""
    mock_nested.side_effect = _mock_get_nested(_SLURM_CONFIG)
    from sky.catalog import slurm_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = slurm_catalog._get_pricing(region='my-slurm', zone='unknown-part')
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=4,
                                                       memory=16,
                                                       accelerator_name=None,
                                                       accelerator_count=None)
    assert cost == pytest.approx(4 * 0.06 + 16 * 0.01)


@mock.patch('sky.skypilot_config.get_nested')
def test_slurm_fallback_to_cloud_level(mock_nested):
    """Unknown cluster falls back to cloud-level default."""
    mock_nested.side_effect = _mock_get_nested(_SLURM_CONFIG)
    from sky.catalog import slurm_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    pricing = slurm_catalog._get_pricing(region='unknown-cluster')
    cost = catalog_common.get_hourly_cost_from_pricing(pricing,
                                                       cpus=4,
                                                       memory=16,
                                                       accelerator_name=None,
                                                       accelerator_count=None)
    assert cost == pytest.approx(4 * 0.04 + 16 * 0.01)


# -- End-to-end get_hourly_cost tests (instance type string -> cost) ---------
# These exercise the full pipeline: instance type parsing + config resolution.
# One per cloud, with accelerators, to verify the parser ↔ pricing glue.


@mock.patch('sky.skypilot_config.get_nested')
def test_k8s_get_hourly_cost_end_to_end(mock_nested):
    """K8s get_hourly_cost: instance type string → parsed resources → cost."""
    mock_nested.side_effect = _mock_get_nested(_K8S_CONFIG)
    from sky.catalog import kubernetes_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    cost = kubernetes_catalog.get_hourly_cost('8CPU--64GB--A100:2',
                                              use_spot=False,
                                              region=None)
    # Accelerator-only pricing: cpu/memory ignored for GPU instances
    expected = 2 * 3.50
    assert cost == pytest.approx(expected)


@mock.patch('sky.skypilot_config.get_nested')
def test_slurm_get_hourly_cost_end_to_end(mock_nested):
    """Slurm get_hourly_cost: full 3-level merge via the public API."""
    mock_nested.side_effect = _mock_get_nested(_SLURM_CONFIG)
    from sky.catalog import slurm_catalog  # isort: skip  # pylint: disable=import-outside-toplevel
    cost = slurm_catalog.get_hourly_cost('8CPU--64GB--A100:4',
                                         use_spot=False,
                                         region='my-slurm',
                                         zone='high-pri')
    # Accelerator-only pricing: partition A100=5.00, cpu/memory ignored
    expected = 4 * 5.00
    assert cost == pytest.approx(expected)


def _efa_catalog_df():
    """Tiny AWS-catalog-like frame exercising the EFA sizing helper.

    Same-accelerator variants differ in EFA-per-accelerator, which is the crux
    of the sizing rule: H100 (p5.48xlarge 4/GPU vs p5.4xlarge 1/GPU), H200
    (p5e 4/GPU vs p5en 2/GPU). A100 is present but not EFA-capable (NaN).
    """
    return pd.DataFrame([
        {
            'InstanceType': 'p5.4xlarge',
            'AcceleratorName': 'H100',
            'AcceleratorCount': 1,
            'MaximumEfaInterfaces': 1.0
        },
        {
            'InstanceType': 'p5.48xlarge',
            'AcceleratorName': 'H100',
            'AcceleratorCount': 8,
            'MaximumEfaInterfaces': 32.0
        },
        {
            'InstanceType': 'p5e.48xlarge',
            'AcceleratorName': 'H200',
            'AcceleratorCount': 8,
            'MaximumEfaInterfaces': 32.0
        },
        {
            'InstanceType': 'p5en.48xlarge',
            'AcceleratorName': 'H200',
            'AcceleratorCount': 8,
            'MaximumEfaInterfaces': 16.0
        },
        {
            'InstanceType': 'p4d.24xlarge',
            'AcceleratorName': 'A100',
            'AcceleratorCount': 8,
            'MaximumEfaInterfaces': np.nan
        },
    ])


def test_efa_count_full_node_uses_hosting_variant_ratio():
    df = _efa_catalog_df()
    # H100:8 only fits the 8-GPU p5.48xlarge (4 EFA/GPU) -> the full 32.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H100',
                                                             8) == 32


def test_efa_count_partial_request_uses_leanest_variant():
    df = _efa_catalog_df()
    # H100:1 fits p5.4xlarge (1 EFA/GPU) and p5.48xlarge (4) -> min ratio 1 -> 1.
    # Sizing from the max (4) would be unschedulable on p5.4xlarge (1 EFA).
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H100', 1) == 1


def test_efa_count_avoids_stranding_on_leaner_variant():
    df = _efa_catalog_df()
    # H200:1 -> min(p5e 4, p5en 2) = 2, not 4. At 4, p5en (16 EFA / 8 GPU) fits
    # only 4 pods and strands 4 of its 8 GPUs.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H200', 1) == 2


def test_efa_count_partial_multi_gpu():
    df = _efa_catalog_df()
    # H100:4 only fits the 8-GPU node -> floor(4 * 4 EFA/GPU) = 16.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H100',
                                                             4) == 16


def test_efa_count_none_when_request_exceeds_all_variants():
    df = _efa_catalog_df()
    # No H100 variant has >= 16 accelerators to host the request -> None.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H100',
                                                             16) is None


def test_efa_count_none_cases():
    df = _efa_catalog_df()
    # Accelerator present but only NaN EFA -> None.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'A100',
                                                             8) is None
    # Accelerator absent -> None.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'V100',
                                                             1) is None
    # Missing column (older/hosted catalog) -> None, not a KeyError.
    df2 = df.drop(columns=['MaximumEfaInterfaces'])
    assert catalog_common.get_efa_count_for_accelerator_impl(df2, 'H100',
                                                             1) is None


def test_efa_count_case_insensitive_fullmatch():
    df = _efa_catalog_df()
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'h100',
                                                             8) == 32
    # Full match, not a prefix/substring.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H10',
                                                             1) is None


def test_efa_count_at_least_one():
    # A sub-1 EFA-per-accelerator ratio still requests >= 1 EFA.
    df = pd.DataFrame([{
        'InstanceType': 'trn1.32xlarge',
        'AcceleratorName': 'Trainium',
        'AcceleratorCount': 16,
        'MaximumEfaInterfaces': 8.0
    }])
    # 8 / 16 = 0.5 EFA/acc -> floor(1 * 0.5) = 0 -> clamped to 1.
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'Trainium',
                                                             1) == 1


def test_efa_count_duplicate_index_safe():
    # Catalog merges can leave duplicate index labels; the min-ratio selection
    # must not mis-select or raise on them.
    df = _efa_catalog_df()
    df.index = [0] * len(df)
    assert catalog_common.get_efa_count_for_accelerator_impl(df, 'H100',
                                                             8) == 32


# ---------------------------------------------------------------------------
# Provider GPU offerings (sky/catalog/offerings.py)
# ---------------------------------------------------------------------------

_OFFERINGS_CSV_HEADER = (
    'InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,'
    'GpuInfo,Price,SpotPrice,Region\n')

# Header of a vast catalog written by a fetcher that carries live
# stock: AvailableUnits (machines collapsed into the stub row).
_OFFERINGS_CSV_HEADER_STOCK = (
    'InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,'
    'GpuInfo,Price,SpotPrice,Region,AvailableUnits\n')


def _write_offerings_csv(catalog_dir,
                         cloud,
                         rows,
                         header=_OFFERINGS_CSV_HEADER):
    cloud_dir = os.path.join(catalog_dir, cloud)
    os.makedirs(cloud_dir, exist_ok=True)
    csv_path = os.path.join(cloud_dir, 'vms.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write('\n'.join(rows) + '\n')
    return csv_path


# The three stock tiers across the default clouds (see
# catalog_offerings._STOCK_MODES): 'units' = live counts, 'in_stock' =
# rows live-filtered at fetch, 'catalog' = offering coverage only.
_EXPECTED_STOCK_MODES = {
    'aws': 'catalog',
    'gcp': 'catalog',
    'shadeform': 'in_stock',
    'spheron': 'units',
    'lambda': 'in_stock',
    'vast': 'units',
    'latitude': 'in_stock',
    'quantacloud': 'units',
    'primeintellect': 'units',
    'ornn': 'units',
}


def _patch_catalog_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_common, '_ABSOLUTE_VERSIONED_CATALOG_DIR',
                        str(tmp_path))
    monkeypatch.chdir(tmp_path)


def test_provider_gpu_offerings_aggregates_catalog(tmp_path, monkeypatch):
    _patch_catalog_dir(monkeypatch, tmp_path)
    csv_path = _write_offerings_csv(
        tmp_path,
        'vast',
        [
            # Sorted by name; counts distinct+sorted; prices are minima.
            'vast-a100-x1,A100,1,16,96,"{}",3.50,2.20,us-east',
            'vast-a100-x1-cheap,A100,1,16,96,"{}",3.00,2.10,us-west',
            'vast-rtx6000ada-x1,RTX6000ADA,1,8,32,"{}",0.50,0.30,us-east',
            'vast-rtx6000ada-x2,RTX6000ADA,2,16,64,"{}",0.90,0.55,us-west',
            'vast-rtx6000ada-x4,RTX6000ADA,4,32,128,"{}",1.80,1.10,us-east',
            # CPU-only row: filtered out by the gpus_only convention.
            'vast-cpu-x8,,8,32,64,,2.10,1.20,us-east',
        ])
    result = catalog.get_provider_gpu_offerings(clouds=['vast'])
    # This fixture's vast CSV predates the AvailableUnits column, so
    # the region buckets carry prices only (units/listings None) —
    # the honest label for a catalog that cannot carry counts.
    assert result == {
        'providers': {
            'vast': {
                'present': True,
                'updated_at': os.path.getmtime(csv_path),
                'stock_mode': 'units',
                'gpus': [
                    {
                        'name': 'A100',
                        'counts': [1],
                        'price': 3.0,
                        'spot_price': 2.1,
                        'regions': ['us-east', 'us-west'],
                        'region_details': [
                            {
                                'region': 'us-east',
                                'units': None,
                                'listings': None,
                                'price': 3.5,
                                'spot_price': 2.2,
                            },
                            {
                                'region': 'us-west',
                                'units': None,
                                'listings': None,
                                'price': 3.0,
                                'spot_price': 2.1,
                            },
                        ],
                    },
                    {
                        'name': 'RTX6000ADA',
                        'counts': [1, 2, 4],
                        'price': 0.5,
                        'spot_price': 0.3,
                        'regions': ['us-east', 'us-west'],
                        'region_details': [
                            {
                                'region': 'us-east',
                                'units': None,
                                'listings': None,
                                'price': 0.5,
                                'spot_price': 0.3,
                            },
                            {
                                'region': 'us-west',
                                'units': None,
                                'listings': None,
                                'price': 0.9,
                                'spot_price': 0.55,
                            },
                        ],
                    },
                ],
            },
        },
    }


def test_provider_gpu_offerings_mtime_passthrough(tmp_path, monkeypatch):
    _patch_catalog_dir(monkeypatch, tmp_path)
    csv_path = _write_offerings_csv(
        tmp_path, 'shadeform', ['shade-x1,H100,1,16,96,"{}",2.00,1.00,us-east'])
    os.utime(csv_path, (1000000.0, 1000000.0))
    result = catalog.get_provider_gpu_offerings(clouds=['shadeform'])
    assert result['providers']['shadeform']['updated_at'] == 1000000.0


def test_provider_gpu_offerings_missing_catalog_never_fetches(
        tmp_path, monkeypatch):
    _patch_catalog_dir(monkeypatch, tmp_path)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError(
            'A missing catalog must be reported absent, never fetched '
            '(the fetch has no timeout and hangs the request).')

    monkeypatch.setattr(catalog_common.requests, 'get', _fail_fetch)
    # Also guard the hosted-catalog fetch path itself.
    monkeypatch.setattr(
        catalog_common, 'read_catalog', lambda *a, **k:
        (_ for _ in
         ()).throw(AssertionError('read_catalog must not be '
                                  'called')))
    result = catalog.get_provider_gpu_offerings(clouds=['ornn'])
    assert result == {
        'providers': {
            'ornn': {
                'present': False,
                'updated_at': None,
                'stock_mode': 'units',
                'gpus': None,
            },
        },
    }


def test_provider_gpu_offerings_default_clouds_no_network(
        tmp_path, monkeypatch):
    _patch_catalog_dir(monkeypatch, tmp_path)

    def _fail_fetch(*args, **kwargs):
        raise AssertionError('Offerings must never touch the network.')

    monkeypatch.setattr(catalog_common.requests, 'get', _fail_fetch)
    result = catalog.get_provider_gpu_offerings()
    assert sorted(result['providers'].keys()) == sorted(
        catalog_offerings.DEFAULT_OFFERINGS_CLOUDS)
    for cloud in catalog_offerings.DEFAULT_OFFERINGS_CLOUDS:
        # stock_mode is published even for absent catalogs (the tier is
        # a property of the provider, not of the data on disk).
        assert result['providers'][cloud] == {
            'present': False,
            'updated_at': None,
            'stock_mode': _EXPECTED_STOCK_MODES[cloud],
            'gpus': None,
        }


def test_provider_gpu_offerings_missing_prices(tmp_path, monkeypatch):
    _patch_catalog_dir(monkeypatch, tmp_path)
    _write_offerings_csv(
        tmp_path,
        'lambda',
        [
            # GPU with no price rows at all -> price/spot_price None.
            'lambda-h100-x1,H100,1,16,96,"{}",,,us-east',
            # GPU with some priced rows -> minima over the priced rows only.
            'lambda-a100-x1,A100,1,16,96,"{}",,1.10,us-east',
            'lambda-a100-x2,A100,2,32,192,"{}",2.50,1.80,us-west',
        ])
    result = catalog.get_provider_gpu_offerings(clouds=['lambda'])
    gpus = result['providers']['lambda']['gpus']
    # lambda rows are already live-filtered (in_stock tier): no counts
    # published, but each region bucket still carries its minima.
    assert gpus == [
        {
            'name': 'A100',
            'counts': [1, 2],
            'price': 2.5,
            'spot_price': 1.1,
            'regions': ['us-east', 'us-west'],
            'region_details': [
                {
                    'region': 'us-east',
                    'units': None,
                    'listings': None,
                    # us-east row has an EMPTY on-demand price: the
                    # region bucket has no price, only a spot price.
                    'price': None,
                    'spot_price': 1.1,
                },
                {
                    'region': 'us-west',
                    'units': None,
                    'listings': None,
                    'price': 2.5,
                    'spot_price': 1.8,
                },
            ],
        },
        {
            'name': 'H100',
            'counts': [1],
            'price': None,
            'spot_price': None,
            'regions': ['us-east'],
            'region_details': [{
                'region': 'us-east',
                'units': None,
                'listings': None,
                'price': None,
                'spot_price': None,
            },],
        },
    ]


def test_provider_gpu_offerings_unknown_cloud():
    result = catalog.get_provider_gpu_offerings(clouds=['notacloud'])
    assert result == {
        'providers': {
            'notacloud': {
                'present': False,
                'updated_at': None,
                'stock_mode': 'catalog',
                'gpus': None,
                'error': 'unknown cloud',
            },
        },
    }


def test_provider_gpu_offerings_region_details_row_multiplicity(
        tmp_path, monkeypatch):
    """Row-multiplicity stock providers (spheron/primeintellect/
    quantacloud/ornn): one catalog row per LIVE offer, so within a
    (GPU, Region) bucket units sums AcceleratorCount and listings
    counts rows. Prices are per-region minima.
    """
    _patch_catalog_dir(monkeypatch, tmp_path)
    _write_offerings_csv(
        tmp_path,
        'spheron',
        [
            # EU: three live offers (1 + 1 + 2 GPUs = 4 chips).
            'sph-rtx-eu1,RTX4090,1,16,32,"{}",1.50,0.70,EU',
            'sph-rtx-eu2,RTX4090,1,16,32,"{}",1.20,0.60,EU',
            'sph-rtx-eu4,RTX4090,2,32,64,"{}",2.40,1.10,EU',
            # NA: one live offer.
            'sph-rtx-na,RTX4090,1,16,32,"{}",1.90,0.90,NA',
            # A second GPU type: single offer, no spot price.
            'sph-h100,H100,8,96,640,"{}",9.90,,NA',
        ])
    result = catalog.get_provider_gpu_offerings(clouds=['spheron'])
    provider = result['providers']['spheron']
    assert provider['present'] is True
    assert provider['stock_mode'] == 'units'
    gpus = provider['gpus']
    rtx = next(gpu for gpu in gpus if gpu['name'] == 'RTX4090')
    # Sorted by region name; prices are the per-region minima.
    assert rtx['region_details'] == [
        {
            'region': 'EU',
            'units': 4,
            'listings': 3,
            'price': 1.2,
            'spot_price': 0.6,
        },
        {
            'region': 'NA',
            'units': 1,
            'listings': 1,
            'price': 1.9,
            'spot_price': 0.9,
        },
    ]
    h100 = next(gpu for gpu in gpus if gpu['name'] == 'H100')
    assert h100['region_details'] == [
        {
            'region': 'NA',
            'units': 8,
            'listings': 1,
            'price': 9.9,
            'spot_price': None,
        },
    ]


def test_provider_gpu_offerings_vast_available_units(tmp_path, monkeypatch):
    """Vast dedupes sibling offers per (InstanceType, Region,
    HostingType) stub at fetch time, so its rows carry AvailableUnits
    (machines collapsed into each row): units = sum(AvailableUnits x
    AcceleratorCount), listings = sum(AvailableUnits) per (GPU,
    Region) bucket — NOT row counts.
    """
    _patch_catalog_dir(monkeypatch, tmp_path)
    _write_offerings_csv(
        tmp_path,
        'vast',
        [
            # Stub row standing for 3 single-GPU machines, plus its
            # spot-price marker row (AvailableUnits 0: no extra stock).
            'vast-rtx-x1,RTX4090,1,16,32,"{}",1.20,0.60,EU,3',
            'vast-rtx-x1-print,RTX4090,1,16,32,"{}",1.20,0.90,EU,0',
            # A 2-GPU stub row standing for 2 machines.
            'vast-rtx-x2,RTX4090,2,32,64,"{}",2.40,1.10,NA,2',
        ],
        header=_OFFERINGS_CSV_HEADER_STOCK)
    result = catalog.get_provider_gpu_offerings(clouds=['vast'])
    provider = result['providers']['vast']
    assert provider['stock_mode'] == 'units'
    gpu = provider['gpus'][0]
    assert gpu['region_details'] == [
        {
            'region': 'EU',
            'units': 3,  # 3 machines x 1 GPU
            'listings': 3,
            'price': 1.2,
            'spot_price': 0.6,
        },
        {
            'region': 'NA',
            'units': 4,  # 2 machines x 2 GPUs
            'listings': 2,
            'price': 2.4,
            'spot_price': 1.1,
        },
    ]


def test_provider_gpu_offerings_catalog_only_no_stock(tmp_path, monkeypatch):
    """Catalog-only clouds (aws, gcp, ...): no live-stock notion, so
    units/listings are null; each region bucket still carries its
    cheapest prices (offering coverage)."""
    _patch_catalog_dir(monkeypatch, tmp_path)
    _write_offerings_csv(tmp_path, 'aws', [
        'p3.2xlarge,V100,1,8,61,"{}",3.06,1.20,us-east-1',
        'p3.8xlarge,V100,4,32,244,"{}",12.24,4.80,us-east-1',
        'p3.2xlarge-eu,V100,1,8,61,"{}",3.60,1.40,eu-west-1',
    ])
    result = catalog.get_provider_gpu_offerings(clouds=['aws'])
    provider = result['providers']['aws']
    assert provider['stock_mode'] == 'catalog'
    gpu = provider['gpus'][0]
    assert gpu['region_details'] == [
        {
            'region': 'eu-west-1',
            'units': None,
            'listings': None,
            'price': 3.6,
            'spot_price': 1.4,
        },
        {
            'region': 'us-east-1',
            'units': None,
            'listings': None,
            'price': 3.06,
            'spot_price': 1.2,
        },
    ]


def test_provider_gpu_offerings_stock_mode_tiers(tmp_path, monkeypatch):
    """The three stock tiers are published per provider so the
    dashboard can label stock honestly: 'units' (live counts),
    'in_stock' (rows live-filtered at fetch; region presence means
    in-stock at last refresh), 'catalog' (offering coverage only —
    also the default for unknown clouds)."""
    _patch_catalog_dir(monkeypatch, tmp_path)
    _write_offerings_csv(tmp_path, 'ornn', [
        'ornn-a100,A100,1,16,96,"{}",2.00,1.00,EU',
    ])
    _write_offerings_csv(tmp_path, 'shadeform', [
        'shade-x1,H100,1,16,96,"{}",2.00,1.00,us-east',
    ])
    _write_offerings_csv(tmp_path, 'aws', [
        'p3.2xlarge,V100,1,8,61,"{}",3.06,1.20,us-east-1',
    ])
    result = catalog.get_provider_gpu_offerings(
        clouds=['ornn', 'shadeform', 'aws', 'notacloud'])
    providers = result['providers']
    assert providers['ornn']['stock_mode'] == 'units'
    assert providers['shadeform']['stock_mode'] == 'in_stock'
    assert providers['aws']['stock_mode'] == 'catalog'
    # Unknown clouds default to the catalog tier.
    assert providers['notacloud']['stock_mode'] == 'catalog'


def test_provider_gpu_offerings_backward_compat_fields(tmp_path, monkeypatch):
    """region_details/stock_mode are ADDITIVE: every pre-existing field
    keeps its shape and values (provider: present/updated_at/gpus;
    GPU: name/counts/price/spot_price/regions)."""
    _patch_catalog_dir(monkeypatch, tmp_path)
    csv_path = _write_offerings_csv(
        tmp_path, 'vast', ['vast-a100-x1,A100,1,16,96,"{}",3.50,2.20,us-east'])
    result = catalog.get_provider_gpu_offerings(clouds=['vast'])
    provider = result['providers']['vast']
    assert set(provider) == {'present', 'updated_at', 'stock_mode', 'gpus'}
    assert provider['present'] is True
    assert provider['updated_at'] == os.path.getmtime(csv_path)
    gpu = provider['gpus'][0]
    assert set(gpu) == {
        'name',
        'counts',
        'price',
        'spot_price',
        'regions',
        'region_details',
    }
    assert gpu['name'] == 'A100'
    assert gpu['counts'] == [1]
    assert gpu['price'] == 3.5
    assert gpu['spot_price'] == 2.2
    assert gpu['regions'] == ['us-east']
    assert gpu['region_details'] == [{
        'region': 'us-east',
        # Pre-AvailableUnits vast CSV: no counts can be derived.
        'units': None,
        'listings': None,
        'price': 3.5,
        'spot_price': 2.2,
    }]
