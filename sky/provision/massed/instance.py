"""MassedCompute instance provisioning.

The lifecycle, and the MassedCompute specifics that shape it (each
verified against https://vm-docs.massedcompute.com/api/v1 and the live API
on 2026-10-21):

* **Instances carry names.** The launch body takes ``instanceName`` and
  ``GET /instance`` lists every running instance with its name, so the
  cluster identity (``cluster_name_on_cloud``) is passed as the launch's
  ``instanceName`` and every identity-bearing operation (query, cluster
  info, terminate) resolves through it SERVER-SIDE. There is no local
  cluster-identity store — the QuantaCloud wrinkle (deployments carry no
  name) does not exist on this API.
* **Products are stable.** The catalog's InstanceType is the product name
  (``gpu_1x_pro_6000_blackwell``), which is exactly the launch API's
  ``productName`` — no ephemeral offer UUID to re-resolve at launch (the
  QuantaCloud second wrinkle does not exist here either). The numeric
  ``imageId`` IS volatile, so it is re-resolved live by image NAME on
  every launch.
* **SSH keys are named at launch.** ``sshKeys`` takes key NAMES
  registered on the account, so the provisioner ensures the public key is
  registered (``POST /ssh-keys``, matched by MATERIAL) and passes its
  name in the launch body. The instance also reports
  ``username``/``password`` (password SSH); the lane uses the key, the
  same access path every sibling lane takes.
* **Readiness is a flag.** ``os_booted == 1`` plus a non-empty ``ip``;
  a just-launched instance can 404 briefly on its first status read
  (the read-after-write window), so ``query_instances`` re-polls a
  bounded window when the caller asks (``retry_if_missing``, the
  QuantaCloud ENG-575 lesson) and the boot poll tolerates 404 until its
  deadline.
* **Terminate is the only teardown.** There is a restart but no
  stop-and-keep, so STOP is declared unsupported at the cloud layer; if
  anything ever routes here it is refused loudly rather than silently
  destroying the box (and its disk) as a "stop".
"""

from __future__ import annotations

import time
import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.massed import utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    pass

logger = sky_logging.init_logger(__name__)

PROVIDER_NAME = 'massed'

# The documented SSH user. The live instance object carries its own
# ``username`` (the docs' example shows ``"Ubuntu"``); that live value is
# the source of truth and this constant is only the bootstrap/fallback
# (the Latitude lesson: documented root, real ubuntu — measure on first
# funded deploy).
SSH_USER = 'ubuntu'

# The literal the ray template carries until `configure_ssh_info`
# substitutes the real key into node_config['PublicKey'] (see the check in
# run_instances for why the literal itself must be refused).
_SSH_PUBLIC_KEY_PLACEHOLDER = 'skypilot:ssh_public_key_content'

#: The bounded read-after-write window for `query_instances`'s
#: `retry_if_missing` re-poll: a just-launched instance can be absent
#: from the account list on the first status poll until the launch lands
#: server-side. 60s at 5s polls is well under every launch-decision
#: timeout, and only ever taken when the caller explicitly asks for the
#: retry (launch paths); steady-state status polls never enter it.
_QUERY_RETRY_WINDOW_S = 60.0
_QUERY_RETRY_INTERVAL_S = 5.0


def _client() -> 'utils.MassedClient':
    return utils.client_from_env()


def _sky_status(instance: Dict[str, Any]):
    """Map one instance's readiness, refusing to guess at an unknown state.

    The steady-state signal is the ``os_booted`` flag plus the ip (the
    provider's status strings are not documented as a closed set);
    anything not booted-with-ip maps to None = transitional (SkyPilot
    keeps waiting). A terminated instance is ABSENT from the account
    list (termination removes it), so absence — not a status string — is
    the terminal signal here.
    """
    if utils.MassedClient.instance_booted(instance):
        return status_lib.ClusterStatus.UP
    return None


def _find_by_name(client: 'utils.MassedClient',
                  cluster_name_on_cloud: str) -> Optional[Dict[str, Any]]:
    """The account's instance named ``cluster_name_on_cloud``, or None.

    Names are the server-side identity (the launch body took the name),
    so an exact string match is the whole resolution — never a prefix or
    case-insensitive guess that could adopt a foreign box.
    """
    name = str(cluster_name_on_cloud or '').strip()
    if not name:
        return None
    for instance in client.list_instances():
        if client.instance_name(instance) == name:
            return instance
    return None


# -- provisioning ----------------------------------------------------------


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Create (or adopt by name) the cluster's single VM and wait for boot."""
    del cluster_name  # cluster_name_on_cloud is the identity we use

    client = _client()

    # InstanceType is the STABLE product name (the launch API's
    # productName, e.g. gpu_1x_pro_6000_blackwell) — the catalog token is
    # passed through verbatim; there is no ephemeral id inside it.
    product_name = str(
        (config.node_config or {}).get('InstanceType') or '').strip()
    if not product_name:
        raise utils.MassedError(
            'node_config is missing InstanceType (the MassedCompute '
            'product name, e.g. gpu_1x_pro_6000_blackwell)')

    # node_config['PublicKey'] is the ONE canonical source: backend_utils
    # puts Massed on the generic `auth.configure_ssh_info` path, which
    # substitutes the real key into the template's
    # `skypilot:ssh_public_key_content` placeholder. The placeholder
    # check is the point: if substitution did NOT run, the literal is
    # still TRUTHY, a bare `if not public_key` waves garbage into
    # ensure_ssh_key, and the launch registers a garbage key the box can
    # never answer (the same trap the Spheron carry documented).
    public_key = (config.node_config or {}).get('PublicKey')
    if isinstance(public_key, str):
        public_key = public_key.strip()
    if not public_key or public_key == _SSH_PUBLIC_KEY_PLACEHOLDER:
        raise utils.MassedError(
            'node_config is missing its PublicKey, or the placeholder '
            f'was never substituted (got {public_key!r}); '
            'auth.configure_ssh_info must run before provisioning — '
            'a MassedCompute VM without a real registered key is '
            'unreachable')

    created: List[str] = []
    existing = _find_by_name(client, cluster_name_on_cloud)
    if existing is None:
        # The SSH key is registered by NAME at launch: ensure the
        # MATERIAL is on the account (matching, not minting duplicates),
        # then pass the name it is registered under.
        ssh_key_name = client.ensure_ssh_key(utils.SSH_KEY_NAME, public_key)
        # The numeric imageId is volatile (a live list); the NAME is the
        # stable key, re-resolved on every launch.
        image_id = client.default_image_id()
        # regionName 'any' lets the provider place the box wherever the
        # product has capacity; a pinned catalog Region token pins it.
        region_name = str(region or '').strip() or utils.MassedClient.REGION_ANY
        instance_uuid = client.launch_instance(
            image_id=image_id,
            product_name=product_name,
            region_name=region_name,
            instance_name=cluster_name_on_cloud,
            ssh_key_names=[ssh_key_name],
        )
        created = [instance_uuid]
        logger.info(
            'launched MassedCompute instance %s (product %s, region %s, '
            'image %s)', instance_uuid, product_name, region_name, image_id)
    else:
        instance_uuid = utils.MassedClient.instance_uuid(existing)
        logger.info(
            'adopting existing MassedCompute instance %s for cluster %r',
            instance_uuid, cluster_name_on_cloud)

    # Readiness is os_booted == 1 plus an ip; the poll fails closed on
    # the deadline (a stuck boot must become a failed runner, not an
    # endless wait) and tolerates the just-launched 404 window.
    instance = client.wait_until_booted(instance_uuid)
    ip = utils.MassedClient.instance_ip(instance)
    logger.info('MassedCompute instance %s booted at %s', instance_uuid, ip)

    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=instance_uuid,
        resumed_instance_ids=[],
        created_instance_ids=created,
    )


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """No-op: run_instances already blocks until the box is booted."""
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """STOP is refused, not silently destroy-terminated.

    The dispatch contract (sky/provision/__init__.py stop_instances
    wrapper after `provider_name` is stripped) is this signature — a
    `(region, cluster_name, ...)` shape would TypeError at dispatch.
    Massed declares STOP unsupported in Massed._CLOUD_UNSUPPORTED_FEATURES
    (there is a restart but no stop-and-keep; terminate destroys the VM
    AND its disk), so SkyPilot never routes here; if anything ever does,
    refuse loudly rather than destroying the box as a "stop" while the
    caller believed it was preserved.
    """
    del cluster_name_on_cloud, provider_config, worker_only
    raise NotImplementedError(
        'stop is unsupported on MassedCompute: there is no stop-and-keep '
        'state — terminate destroys the VM and its disk. Use '
        'terminate_instances for teardown.')


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Terminate the cluster's VMs (see module docstring)."""
    del provider_config, worker_only  # single-node: head is the cluster
    if not str(cluster_name_on_cloud or '').strip():
        # An empty identity maps to NOTHING here — but the refusal stays
        # loud so a miswired caller can never destroy by guess (the same
        # rule the QuantaCloud/Latitude terminate paths carry).
        raise utils.MassedError(
            'refusing to terminate with an empty cluster_name_on_cloud')
    client = _client()
    name = str(cluster_name_on_cloud).strip()
    uuids = [
        utils.MassedClient.instance_uuid(instance)
        for instance in client.list_instances()
        if client.instance_name(instance) == name and
        utils.MassedClient.instance_uuid(instance)
    ]
    if not uuids:
        # Nothing named us anywhere: already torn down (termination
        # REMOVES the instance, so absence is the clean state).
        return
    client.terminate_instances(uuids)
    logger.info('terminated MassedCompute instance(s) %s', uuids)


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region, provider_config
    client = _client()
    instance = _find_by_name(client, cluster_name_on_cloud)
    if instance is None:
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)
    instance_uuid = utils.MassedClient.instance_uuid(instance)
    ip = utils.MassedClient.instance_ip(instance)
    # The provider reports the SSH user per instance (its ``username``
    # field); the documented constant is only the fallback (the Latitude
    # lesson: docs can say one thing, the box answers another).
    ssh_user = (utils.MassedClient.instance_ssh_user(instance) or SSH_USER)
    return common.ClusterInfo(
        instances={
            instance_uuid: [
                common.InstanceInfo(
                    instance_id=instance_uuid,
                    # A MassedCompute VM exposes one public address; there
                    # is no separate internal address to report.
                    internal_ip=ip,
                    external_ip=ip,
                    ssh_port=22,
                    tags={},
                    node_name=cluster_name_on_cloud,
                )
            ]
        },
        head_instance_id=instance_uuid,
        provider_name=PROVIDER_NAME,
        ssh_user=ssh_user,
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    del cluster_name, provider_config, non_terminated_only
    client = _client()
    # The read-after-write window (the QuantaCloud ENG-575 lesson): a
    # JUST-LAUNCHED instance can be absent from the account list on the
    # provider's first status poll. Without the retry, a launch-path
    # status read would treat the just-launched box as deleted and strand
    # the launch; with it, re-poll a bounded window before concluding
    # absence. Steady-state polls (no retry requested) never enter it,
    # and a box that is REALLY gone is unchanged: absence, reported as
    # absence.
    deadline = time.monotonic(
    ) + _QUERY_RETRY_WINDOW_S if retry_if_missing else 0.0
    while True:
        instance = _find_by_name(client, cluster_name_on_cloud)
        if instance is not None:
            break
        if not retry_if_missing or time.monotonic() >= deadline:
            return {}
        time.sleep(_QUERY_RETRY_INTERVAL_S)
    instance_uuid = utils.MassedClient.instance_uuid(instance)
    if not instance_uuid:
        # A row without a uuid cannot be addressed (terminate/status);
        # report absence rather than a keyless mapping.
        return {}
    return {instance_uuid: (_sky_status(instance), None)}


def open_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op: MassedCompute VMs have no port firewall in the launch path
    (direct SSH on port 22)."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op counterpart to open_ports."""
    del cluster_name_on_cloud, ports, provider_config