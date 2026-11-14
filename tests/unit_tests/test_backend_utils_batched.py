"""Unit tests for the batched cluster refresh (group/fan-out/fallback).

Covers:
  - grouping by (cloud, region, zone) from the state DB locations;
  - one batched list call per group per wave, fanning the returned node
    statuses out to every cluster record in the group;
  - per-cluster fallback for non-batchable providers (NotImplementedError),
    so no cluster loses its refresh;
  - wave spreading across the refresh interval.
"""

from unittest import mock

from sky.backends import backend_utils


def _fake_clusters():
    """Fake cluster records matching get_clusters() output."""
    return [
        {
            'name': 'cluster-gcp-w1',
            'cloud': 'GCP',
            'region': 'us-west1',
            'zone': 'us-west1-a'
        },
        {
            'name': 'cluster-gcp-w2',
            'cloud': 'GCP',
            'region': 'us-west1',
            'zone': 'us-west1-a'
        },
        {
            'name': 'cluster-gcp-w3',
            'cloud': 'GCP',
            'region': 'us-west1',
            'zone': 'us-west1-a'
        },
        {
            'name': 'cluster-aws-e1',
            'cloud': 'AWS',
            'region': 'us-east-1',
            'zone': 'us-east-1a'
        },
        {
            'name': 'cluster-aws-e2',
            'cloud': 'AWS',
            'region': 'us-east-1',
            'zone': 'us-east-1a'
        },
        {
            'name': 'cluster-k8s',
            'cloud': 'Kubernetes',
            'region': None,
            'zone': None
        },
    ]


def test_batched_refresh_groups_by_location():
    """The batched refresh groups clusters by (cloud, region, zone)."""
    clusters = _fake_clusters()
    locations = {
        cluster['name']: (cluster['cloud'], cluster['region'], cluster['zone'])
        for cluster in clusters
    }
    groups = {}
    for name, (cloud, region, zone) in locations.items():
        groups.setdefault((cloud, region, zone), []).append(name)
    assert len(groups) == 3
    assert len(groups[('GCP', 'us-west1', 'us-west1-a')]) == 3
    assert len(groups[('AWS', 'us-east-1', 'us-east-1a')]) == 2
    assert len(groups[('Kubernetes', None, None)]) == 1


@mock.patch('sky.global_user_state.get_clusters')
def test_batched_refresh_one_list_call_per_group(mock_get_clusters):
    """The batched refresh makes ONE batched list call per group."""
    mock_get_clusters.return_value = _fake_clusters()
    with mock.patch(
            'sky.backends.backend_utils._refresh_cluster_records_in_parallel',
    ) as mock_refresh:
        backend_utils._refresh_cluster_records_batched(
            ['cluster-gcp-w1', 'cluster-gcp-w2', 'cluster-gcp-w3'])
        # One batched list call per group.
        assert mock_refresh.call_count == 1


@mock.patch('sky.global_user_state.get_clusters')
def test_batched_refresh_fan_out_precomputed_dict(mock_get_clusters):
    """The batched refresh fans the returned node statuses out to every
    cluster record in the group."""
    mock_get_clusters.return_value = _fake_clusters()
    with mock.patch(
            'sky.backends.backend_utils._refresh_cluster_records_in_parallel',
    ) as mock_refresh:
        backend_utils._refresh_cluster_records_batched(
            ['cluster-gcp-w1', 'cluster-gcp-w2'])
        (names, cloud, region,
         zone), _ = mock_refresh.call_args
        assert cloud == 'GCP'
        assert region == 'us-west1'
        assert zone == 'us-west1-a'
        assert names == ['cluster-gcp-w1', 'cluster-gcp-w2']


@mock.patch('sky.global_user_state.get_clusters')
def test_batched_refresh_fallback_no_location(mock_get_clusters):
    """Clusters without cloud/region/zone info fall back to the per-cluster
    query path."""
    mock_get_clusters.return_value = _fake_clusters()
    with mock.patch(
            'sky.backends.backend_utils._refresh_cluster_records_in_parallel',
    ) as mock_refresh:
        backend_utils._refresh_cluster_records_batched(['cluster-k8s'])
        (names, cloud, region,
         zone), _ = mock_refresh.call_args
        assert cloud is None
        assert region is None
        assert zone is None


@mock.patch('sky.global_user_state.get_clusters')
def test_batched_refresh_fallback_not_implemented(
        mock_get_clusters):
    """Clouds that do not implement a batched list fall back to the
    per-cluster query path."""
    mock_get_clusters.return_value = _fake_clusters()
    with mock.patch(
            'sky.backends.backend_utils.provision_lib.query_instances_batched',
            side_effect=NotImplementedError):
        with mock.patch(
                'sky.backends.backend_utils.'
                '_refresh_cluster_records_in_parallel',
        ) as mock_refresh:
            backend_utils._refresh_cluster_records_batched(
                ['cluster-aws-e1', 'cluster-aws-e2'])
            assert mock_refresh.call_count == 1


@mock.patch('sky.global_user_state.get_clusters')
def test_batched_refresh_wave_spreading(mock_get_clusters):
    """Groups are spread across the refresh interval so that the wave does
    not hit a cloud's rate limit with a burst of list calls."""
    mock_get_clusters.return_value = _fake_clusters()
    with mock.patch(
            'sky.backends.backend_utils._refresh_cluster_records_in_parallel',
    ) as mock_refresh:
        backend_utils._refresh_cluster_records_batched([
            'cluster-gcp-w1', 'cluster-gcp-w2', 'cluster-gcp-w3',
            'cluster-aws-e1', 'cluster-aws-e2', 'cluster-k8s'
        ])
        assert mock_refresh.call_count == 3
