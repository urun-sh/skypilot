"""io.net instance provisioning.

The lifecycle, and the io.net specifics that shape it (each verified
against https://api.io.solutions/enterprise/v1/io-cloud/vmaas live on
2026-11-03):

* **Deployments carry names.** The deploy body takes
  ``resource_private_name`` and ``GET /deploy`` lists every deployment
  with its ``name``, so the cluster identity
  (``cluster_name_on_cloud``) is passed as the deploy's
  ``resource_private_name`` and every identity-bearing operation
  (query, cluster info, terminate) resolves through it SERVER-SIDE.
  There is no local cluster-identity store.
* ``deploy_id`` is the stable InstanceType.`` The catalog's
  InstanceType is the hardware config's ``deploy_id``
  (``8H100.80S.176V``), which is exactly the deploy API's
  ``hardware_id`` — no ephemeral offer UUID to re-resolve at launch.
  The catalog Region is the lowercase country code (``us``); the deploy
  body's ``location_ids`` wants the UPPERCASE one (``US``).
* **SSH keys ride inline.** ``ssh_keys`` is a name → MATERIAL map in
  the deploy body itself — io.net has no key registry to ensure against
  (the MassedCompute name-registry wrinkle does not exist here).
* **Readiness is a status string.** ``GET /deployment/{id}`` reports
  ``status``; 'running' is the steady state. A JUST-CREATED deployment
  can 404 on its first detail read (the read-after-write window), so
  the boot poll tolerates 404 until its deadline (the QuantaCloud
  ENG-575 lesson).
* **SSH arrives as a command string.** ``GET
  /deployment/{id}/vms`` workers carry ``ssh_access`` — the full
  ``ssh -p <port> <user>@<host>`` command the provider's own UI shows —
  parsed for host/user/port, with ``network_services``
  ``public_ip``/``public_port`` as the fallback address source.
* **Destroy is the only teardown.** ``DELETE /deployment/{id}``
  destroys the VMs and their data (first hour non-refundable); there is
  no stop-and-keep, so STOP is declared unsupported at the cloud
  layer — if anything ever routes here it is refused loudly rather than
  silently destroying the box while the caller believed it preserved.
"""

from __future__ import annotations

import time
import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.ionet import utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    pass

logger = sky_logging.init_logger(__name__)

PROVIDER_NAME = 'ionet'

# The documented SSH user for the 'general' image (Ubuntu 24.04 + CUDA).
# The live worker's ssh_access command carries its own user; that live
# value is the source of truth and this constant is only the fallback
# (the Latitude lesson: documented root, real ubuntu — measure on first
# funded deploy).
SSH_USER = 'root'

# The literal the ray template carries until `configure_ssh_info`
# substitutes the real key into node_config['PublicKey'] (see the check
# in run_instances for why the literal itself must be refused).
_SSH_PUBLIC_KEY_PLACEHOLDER = 'skypilot:ssh_public_key_content'

# The deploy body's duration for a fresh launch: 1 hour, the minimum the
# provider bills (non-refundable); extend_deployment adds time later if
# anything ever needs it.
_DEPLOY_DURATION_HOURS = 1

#: The bounded read-after-write window for `query_instances`'
#: `retry_if_missing` re-poll: a just-created deployment can be absent
#: from the account list on the first status poll until the deploy lands
#: server-side. 60s at 5s polls is well under every launch-decision
#: timeout, and only ever taken when the caller explicitly asks for the
#: retry (launch paths); steady-state status polls never enter it.
_QUERY_RETRY_WINDOW_S = 60.0
_QUERY_RETRY_INTERVAL_S = 5.0


def _client() -> 'utils.IonetClient':
    return utils.client_from_env()


def _sky_status(deployment: Dict[str, Any]):
    """Map one deployment's readiness, refusing to guess at an unknown
    state.

    The steady-state signal is the provider's ``status`` string
    ('running' while the boxes are up); the set is NOT documented as
    closed, so anything not 'running' maps to INIT = transitional — a
    PRESENT, provisioning deployment must not read as None, which the
    provision contract treats as terminated/terminating (the refresh
    path would drop the cluster record while the box is still booting).
    A destroyed deployment is ABSENT from the account list (destroy
    removes it), so absence — not a status string — is the terminal
    signal here.
    """
    if utils.IonetClient.deployment_status(deployment) == utils.STATUS_RUNNING:
        return status_lib.ClusterStatus.UP
    return status_lib.ClusterStatus.INIT


def _find_by_name(client: 'utils.IonetClient',
                  cluster_name_on_cloud: str) -> Optional[Dict[str, Any]]:
    """The account's deployment named ``cluster_name_on_cloud``, or None.

    Names are the server-side identity (the deploy body took the name),
    so an exact string match is the whole resolution — never a prefix or
    case-insensitive guess that could adopt a foreign deployment.
    """
    name = str(cluster_name_on_cloud or '').strip()
    if not name:
        return None
    for deployment in client.list_deployments():
        if client.deployment_name(deployment) == name:
            return deployment
    return None


def _worker_address(worker: Dict[str, Any],
                    client: 'utils.IonetClient') -> Tuple[str, int, str]:
    """The worker's (host, port, user) via the client's resolver.

    ``ssh_access`` (the full command string) is the primary source;
    ``network_services``' public_ip/public_port is the fallback. Neither
    yielding a host is a loud failure — never a guessed address for a
    box that is still billing.
    """
    address = client.worker_address(worker)
    if address is None:
        vm_id = str(worker.get('vm_id') or worker.get('device_id') or '?')
        raise utils.IonetError(
            f'worker {vm_id} carries neither a parseable ssh_access nor a '
            'network_services public_ip; refusing to guess an address')
    user, host, port = address
    return host, int(port), (user or SSH_USER)


def _worker_instance_id(worker: Dict[str, Any]) -> str:
    """The worker's stable instance id (vm_id, falling back to device_id)."""
    return str(worker.get('vm_id') or worker.get('device_id') or '').strip()


# -- provisioning ----------------------------------------------------------


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Create (or adopt by name) the cluster's deployment and wait for
    running."""
    del cluster_name  # cluster_name_on_cloud is the identity we use

    client = _client()

    # InstanceType is the STABLE deploy_id (the launch API's
    # hardware_id, e.g. 8H100.80S.176V) — the catalog token is passed
    # through verbatim; there is no ephemeral id inside it.
    hardware_id = str((config.node_config or {}).get('InstanceType') or
                      '').strip()
    if not hardware_id:
        raise utils.IonetError(
            'node_config is missing InstanceType (the io.net deploy_id, '
            'e.g. 8H100.80S.176V)')

    # gpus_per_vm is the hardware config's own card count — the deploy
    # body must describe the same box the catalog row priced (the whole
    # box, not a slice; the provider prices per-VM).
    gpus_per_vm = (config.node_config or {}).get('GpusPerVm')
    if not isinstance(gpus_per_vm, int) or isinstance(gpus_per_vm, bool) \
            or gpus_per_vm < 1:
        raise utils.IonetError(
            f'node_config is missing GpusPerVm (the config\'s card count), '
            f'got {gpus_per_vm!r}')

    # node_config['PublicKey'] is the ONE canonical source: backend_utils
    # puts IoNet on the generic `auth.configure_ssh_info` path, which
    # substitutes the real key into the template's
    # `skypilot:ssh_public_key_content` placeholder. The placeholder
    # check is the point: if substitution did NOT run, the literal is
    # still TRUTHY, a bare `if not public_key` waves garbage into the
    # deploy body, and the box boots unreachable (the same trap the
    # Spheron/Massed carry documented).
    public_key = (config.node_config or {}).get('PublicKey')
    if isinstance(public_key, str):
        public_key = public_key.strip()
    if not public_key or public_key == _SSH_PUBLIC_KEY_PLACEHOLDER:
        raise utils.IonetError(
            'node_config is missing its PublicKey, or the placeholder '
            f'was never substituted (got {public_key!r}); '
            'auth.configure_ssh_info must run before provisioning — '
            'an io.net VM without a real inline key is unreachable '
            'and still bills')

    # The catalog Region is the lowercase country code (us, fi, ...);
    # the deploy body's location_ids wants the UPPERCASE code (US, FI)
    # — the provider's documented shape.
    location = str(region or '').strip().upper()
    if not location:
        raise utils.IonetError(
            'refusing to deploy without a location (the catalog Region '
            'country code, e.g. us)')

    created: List[str] = []
    existing = _find_by_name(client, cluster_name_on_cloud)
    if existing is None:
        # ssh_keys is a name → MATERIAL map riding inline with the
        # deploy (no key registry on io.net); the name is only a label.
        deployment_id = client.deploy(
            resource_private_name=cluster_name_on_cloud,
            duration_hours=_DEPLOY_DURATION_HOURS,
            hardware_id=hardware_id,
            gpus_per_vm=gpus_per_vm,
            location_ids=[location],
            ssh_keys={utils.SSH_KEY_NAME: public_key},
            vm_image_type=utils.DEFAULT_IMAGE_TYPE,
        )
        created = [deployment_id]
        logger.info(
            'launched io.net deployment %s (hardware %s, location %s, '
            'image %s)', deployment_id, hardware_id, location,
            utils.DEFAULT_IMAGE_TYPE)
    else:
        deployment_id = utils.IonetClient.deployment_id(existing)
        logger.info(
            'adopting existing io.net deployment %s for cluster %r',
            deployment_id, cluster_name_on_cloud)

    # Readiness is status 'running'; the poll fails closed on the
    # deadline (a stuck deploy must become a failed runner, not an
    # endless wait — the box bills from the first hour) and tolerates
    # the just-created 404 window.
    client.wait_until_running(deployment_id)
    logger.info('io.net deployment %s is running', deployment_id)

    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=deployment_id,
        resumed_instance_ids=[],
        created_instance_ids=created,
    )


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """No-op: run_instances already blocks until the deployment runs."""
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """STOP is refused, not silently destroy-DONE.

    The dispatch contract (sky/provision/__init__.py stop_instances
    wrapper after `provider_name` is stripped) is this signature — a
    ``(region, cluster_name, ...)`` shape would TypeError at dispatch.
    IoNet declares STOP unsupported in its cloud class (there is no
    stop-and-keep state — destroy terminates the VMs AND their data,
    first hour non-refundable), so SkyPilot never routes here; if
    anything ever does, refuse loudly rather than destroying the box as
    a "stop" while the caller believed it was preserved.
    """
    del cluster_name_on_cloud, provider_config, worker_only
    raise NotImplementedError(
        'stop is unsupported on io.net: there is no stop-and-keep '
        'state — destroy terminates the deployment and its data. Use '
        'terminate_instances for teardown.')


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Terminate the cluster's deployments (see module docstring)."""
    del provider_config, worker_only  # single-node: head is the cluster
    if not str(cluster_name_on_cloud or '').strip():
        # An empty identity maps to NOTHING here — but the refusal stays
        # loud so a miswired caller can never destroy by guess (the same
        # rule the QuantaCloud/Latitude/Massed terminate paths carry).
        raise utils.IonetError(
            'refusing to terminate with an empty cluster_name_on_cloud')
    client = _client()
    name = str(cluster_name_on_cloud).strip()
    deployment_ids = [
        utils.IonetClient.deployment_id(deployment)
        for deployment in client.list_deployments()
        if client.deployment_name(deployment) == name and
        utils.IonetClient.deployment_id(deployment)
    ]
    if not deployment_ids:
        # Nothing named us anywhere: already torn down (destroy REMOVES
        # the deployment, so absence is the clean state).
        return
    for deployment_id in deployment_ids:
        # 404-tolerant per-deployment: one already-gone box must not
        # block the rest of the teardown.
        client.destroy_deployment(deployment_id)
    logger.info('terminated io.net deployment(s) %s', deployment_ids)


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region, provider_config
    client = _client()
    deployment = _find_by_name(client, cluster_name_on_cloud)
    deployment_id = utils.IonetClient.deployment_id(deployment or {})
    if deployment_id and client.deployment_status(
            deployment or {}) == utils.STATUS_RUNNING:
        # The SSH source is the worker row: ssh_access (the full command
        # string) with the network_services public_ip/public_port
        # fallback; the user rides in the command string itself.
        instances: Dict[str, List[common.InstanceInfo]] = {}
        ssh_user = SSH_USER
        for worker in client.get_deployment_vms(deployment_id):
            instance_id = _worker_instance_id(worker)
            if not instance_id:
                continue
            host, ssh_port, user = _worker_address(worker, client)
            ssh_user = user
            # The general image exposes one public address; there is no
            # separate internal address to report.
            instances[instance_id] = [
                common.InstanceInfo(
                    instance_id=instance_id,
                    internal_ip=host,
                    external_ip=host,
                    ssh_port=ssh_port,
                    tags={},
                    node_name=cluster_name_on_cloud,
                )
            ]
        return common.ClusterInfo(instances=instances,
                                  head_instance_id=(
                                      deployment_id if instances else None),
                                  provider_name=PROVIDER_NAME,
                                  ssh_user=ssh_user)
    return common.ClusterInfo(instances={},
                              head_instance_id=None,
                              provider_name=PROVIDER_NAME)


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
    # JUST-CREATED deployment can be absent from the account list on the
    # provider's first status poll. Without the retry, a launch-path
    # status read would treat the just-launched box as deleted and
    # strand the launch; with it, re-poll a bounded window before
    # concluding absence. Steady-state polls (no retry requested) never
    # enter it, and a box that is REALLY gone is unchanged: absence,
    # reported as absence.
    deadline = time.monotonic(
    ) + _QUERY_RETRY_WINDOW_S if retry_if_missing else 0.0
    while True:
        deployment = _find_by_name(client, cluster_name_on_cloud)
        if deployment is not None:
            break
        if not retry_if_missing or time.monotonic() >= deadline:
            return {}
        time.sleep(_QUERY_RETRY_INTERVAL_S)
    deployment_id = utils.IonetClient.deployment_id(deployment)
    if not deployment_id:
        # A row without an id cannot be addressed (terminate/status);
        # report absence rather than a keyless mapping.
        return {}
    return {deployment_id: (_sky_status(deployment), None)}


def open_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op: io.net opens ports through the deploy body's
    ``network_services`` map (passed in run_instances' provider config);
    there is no post-deploy firewall API in the launch path (SSH rides
    port 22)."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op counterpart to open_ports (ports die with the deployment)."""
    del cluster_name_on_cloud, ports, provider_config