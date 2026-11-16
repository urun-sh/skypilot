"""custom_user_data: the per-launch startup-script injection point (boot-cut).

A launch that declares `gcp: custom_user_data: <script>` in its cluster
config overrides appends the script to the VM's startup metadata — it runs
at BOOT, before SSH, before file_mounts, before setup_commands. This is the
provider-agnostic injection point for content the runtime env needs BEFORE
SkyPilot's runtime prep (e.g. mounting the lazily-streamed SkyPilot runtime
env via nydusd); the key is plain cluster config, so every cloud template
that renders user-data can adopt it without a fork change per provider.
"""

from unittest import mock

import pytest

import sky
from sky.clouds import gcp as gcp_lib
from sky.utils import resources_utils


def _make_vars(resources):
    cloud = gcp_lib.GCP()
    region = sky.clouds.cloud.Region(name='us-west1')
    zone = sky.clouds.cloud.Zone(name='us-west1-a')
    zones = [region.set_zones([zone])]
    cluster_name = resources_utils.ClusterName(
        'bootfast-custom-user-data-test', 'bootfast-custom-user-data-test')
    with mock.patch.object(gcp_lib.GCP, 'get_project_id', return_value='p'):
        return cloud.make_deploy_resources_variables(
            resources, cluster_name, region, zones, num_nodes=1)


@pytest.mark.parametrize(
    'override',
    [
        {'gcp': {
            'custom_user_data': 'echo bootfast-hello'
        }},
    ])
def test_custom_user_data_appended_to_startup_metadata(override):
    resources = sky.Resources(cloud=gcp_lib.GCP(),
                              instance_type='n2-standard-8',
                              region='us-west1')
    resources._cluster_config_overrides = override
    vars_out = _make_vars(resources)
    user_data = vars_out.get('user_data')
    assert user_data is not None, (
        'custom_user_data override must render a startup script, got None')
    assert 'echo bootfast-hello' in user_data


def test_no_custom_user_data_keeps_startup_metadata_none():
    resources = sky.Resources(cloud=gcp_lib.GCP(),
                              instance_type='n2-standard-8',
                              region='us-west1')
    vars_out = _make_vars(resources)
    assert vars_out.get('user_data') is None


def test_custom_user_data_is_a_task_overrideable_key():
    """The key must ride task -> server (Resources.copy filters incoming
    overrides to OVERRIDEABLE_CONFIG_KEYS_IN_TASK); a key missing from the
    list is silently dropped before the server can act on it."""
    from sky.skylet import constants as skylet_constants

    assert ('gcp', 'custom_user_data') in skylet_constants.OVERRIDEABLE_CONFIG_KEYS_IN_TASK


def test_resources_copy_preserves_custom_user_data():
    """The full task path: resources.copy(_cluster_config_overrides=...) must
    SURVIVE the task-override filter and still land in the deploy variables."""
    from sky.skylet import constants as skylet_constants

    resources = sky.Resources(cloud=gcp_lib.GCP(),
                              instance_type='n2-standard-8',
                              region='us-west1')
    copied = resources.copy(
        _cluster_config_overrides={
            'gcp': {
                'custom_user_data': 'echo bootfast-hello'
            }
        })
    assert copied.cluster_config_overrides.get('gcp', {}).get(
        'custom_user_data') == 'echo bootfast-hello'
    del skylet_constants
    vars_out = _make_vars(copied)
    assert 'echo bootfast-hello' in (vars_out.get('user_data') or '')


def test_runtime_env_mount_is_provider_independent():
    """The mount invocation must live in the SHARED setup composition, not a
    per-cloud hook: every cloud template embeds uv_installation_commands, so
    the prepended mount runs before the guards on EVERY provider sky
    launches. gcp.custom_user_data stays as the GCP accelerator (mount at
    BOOT overlaps VM boot); this line is the provider-independent guarantee."""
    from sky.skylet import constants as skylet_constants

    assert "/usr/local/sbin/urun-sky-runtime-env-mount" in (
        skylet_constants.UV_INSTALLATION_COMMANDS)
    # bash-invoked (file_mounts mode preservation is not guaranteed across
    # clouds; a missing script is a no-op, never a setup failure).
    assert "bash /usr/local/sbin/urun-sky-runtime-env-mount" in (
        skylet_constants.UV_INSTALLATION_COMMANDS)
