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
        (names, cloud, region, zone), _ = mock_refresh.call_args
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
        (names, cloud, region, zone), _ = mock_refresh.call_args
        assert cloud is None
        assert region is None
        assert zone is None


@mock.patch('sky.global_user_state.get_clusters')
def test_batched_refresh_fallback_not_implemented(mock_get_clusters):
    """Clouds that do not implement a batched list fall back to the
    per-cluster query path."""
    mock_get_clusters.return_value = _fake_clusters()
    with mock.patch(
            'sky.backends.backend_utils.provision_lib.query_instances_batched',
            side_effect=NotImplementedError):
        with mock.patch(
                'sky.backends.backend_utils.'
                '_refresh_cluster_records_in_parallel',) as mock_refresh:
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
    ) as mock_refresh, mock.patch(
            'sky.backends.backend_utils.time.sleep') as mock_sleep:
        backend_utils._refresh_cluster_records_batched([
            'cluster-gcp-w1', 'cluster-gcp-w2', 'cluster-gcp-w3',
            'cluster-aws-e1', 'cluster-aws-e2', 'cluster-k8s'
        ])
        assert mock_refresh.call_count == 3
        # 3 groups over the 60s daemon interval: one group every 20s, the
        # first one immediately.
        assert [c.args[0] for c in mock_sleep.call_args_list] == [20.0, 20.0]


def test_batched_list_is_routable_for_gcp_and_aws():
    """provision_lib routes through the provider package; without the
    re-export every group silently falls back to per-cluster queries."""
    from sky.provision import aws as aws_provision
    from sky.provision import gcp as gcp_provision
    assert callable(gcp_provision.query_instances_batched)
    assert callable(aws_provision.query_instances_batched)


def _handle(name_on_cloud):
    handle = mock.MagicMock()
    handle.cluster_name_on_cloud = name_on_cloud
    handle.cluster_yaml = f'{name_on_cloud}.yml'
    return handle


def test_in_scope_absent_cluster_gets_authoritative_empty_status():
    """A cluster in the batch's (project, zone, region, handler) scope that the
    listing did not return has no live instances: it must get {} (what its
    per-cluster query returns) instead of paying its own list call. A cluster
    in another project stays absent and falls back to its own query."""
    cfg = {'project_id': 'p', 'availability_zone': 'z-a', 'region': 'r'}
    other = dict(cfg, project_id='other')
    handles = {
        'up': _handle('up-oc'),
        'gone': _handle('gone-oc'),
        'elsewhere': _handle('elsewhere-oc')
    }
    cfgs = {'up-oc.yml': cfg, 'gone-oc.yml': cfg, 'elsewhere-oc.yml': other}
    seen = {}

    def fake_refresh(name, **kwargs):
        seen[name] = kwargs['precomputed_node_status_dict']

    with mock.patch('sky.global_user_state.get_cluster_from_name',
                    side_effect=lambda n: {'handle': handles[n]}), \
         mock.patch('sky.global_user_state.get_cluster_yaml_dict',
                    side_effect=lambda y: {'provider': cfgs[y]}), \
         mock.patch('sky.backends.backend_utils.provision_lib.'
                    'query_instances_batched',
                    return_value={'up-oc': {'i-1': ('UP', None)}}) as listed, \
         mock.patch('sky.backends.backend_utils._refresh_cluster',
                    side_effect=fake_refresh):
        backend_utils._refresh_cluster_records_in_parallel(
            ['up', 'gone', 'elsewhere'], 'GCP', 'r', 'z-a')
    assert listed.call_count == 1
    precomputed = seen['up']
    assert precomputed['up-oc'] == {'i-1': ('UP', None)}
    assert precomputed['gone-oc'] == {}
    assert 'elsewhere-oc' not in precomputed


def test_gcp_compute_filter_follows_next_page_token():
    """A zone-wide batched listing can exceed one 500-instance page; every
    page must be read or clusters on later pages would look terminated."""
    from sky.provision.gcp import instance_utils
    pages = [
        {
            'items': [{
                'name': 'a'
            }, {
                'name': 'b'
            }],
            'nextPageToken': 't1'
        },
        {
            'items': [{
                'name': 'c'
            }]
        },
    ]
    list_call = mock.MagicMock()
    list_call.return_value.execute.side_effect = pages
    resource = mock.MagicMock()
    resource.instances.return_value.list = list_call
    with mock.patch.object(instance_utils.GCPComputeInstance,
                           'load_resource',
                           return_value=resource):
        got = instance_utils.GCPComputeInstance.filter(
            'p', 'z', {'ray-cluster-name': '*'}, status_filters=None)
    assert sorted(got) == ['a', 'b', 'c']
    assert [c.kwargs['pageToken'] for c in list_call.call_args_list
           ] == [None, 't1']
    assert list_call.call_args_list[0].kwargs['filter'] == (
        '((labels.ray-cluster-name:*))')
