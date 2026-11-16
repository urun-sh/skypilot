"""Verda instance provisioning.

The lifecycle, and the Verda-specific wrinkles that shape it (all
measured from the OpenAPI spec reviewed 2026-11-16):

* **Instances carry no cluster-identifying name field we can rely on**
  (``hostname`` is deploy-set, not query-set), so this module keeps a
  LOCAL mapping ``cluster_name_on_cloud -> instance_id``
  (``~/.sky/verda-clusters.json``, written atomically at the single
  create point). Every identity-bearing operation (query, cluster info,
  terminate) resolves through it and REFUSES to guess when missing: an
  unmapped live instance is someone's box and the operator decides —
  never a silent teardown.
* **Region is the location NAME** from the catalog ("Finland 1"); the
  deploy API wants the location CODE ("FIN-01"), resolved at launch via
  ``GET /v1/locations`` (never guessed from the name).
* **Teardown goes through the deprecated action endpoint** — the spec
  has no ``DELETE /v1/instances/{id}``; ``POST /v1/instances/action``
  with ``{"action": "delete", "id": ...}`` (404 is success).

Teardown is idempotent; stop is refused at the cloud layer (see
sky/clouds/verda.py).

The SSH user is the image's default; the stock Ubuntu/CUDA images run
as ``root`` (DataCrunch convention carried through the rebrand) —
UNMEASURED until the first funded deploy (the Latitude lesson:
documented and real users diverge).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
import typing
from typing import Any, Dict, Optional, Tuple

from filelock import FileLock

from sky import sky_logging
from sky.provision import common
from sky.utils import status_lib
from sky.provision.verda import utils

if typing.TYPE_CHECKING:
    pass

logger = sky_logging.init_logger(__name__)

PROVIDER_NAME = 'verda'

# The image's default SSH user. UNMEASURED until the first funded deploy
# (docs say root); the live instance object is the source of truth if it
# ever exposes a user field, and this constant is only the fallback.
SSH_USER = 'root'

# The literal the ray template carries until `configure_ssh_info`
# substitutes the real key into node_config['PublicKey'].
_SSH_PUBLIC_KEY_PLACEHOLDER = 'skypilot:ssh_public_key_content'

# Verda instance status -> SkyPilot cluster status (statuses measured
# from the GetInstanceResponsePublicApiDto enum, spec reviewed
# 2026-11-16). Terminal states surface as STOPPED (the ClusterStatus
# enum has no TERMINATED) so reconcile reaps them.
_STATUS_MAP: Dict[str, Optional[status_lib.ClusterStatus]] = {
    'running': status_lib.ClusterStatus.UP,
    # provisioning/ordered/new/validating are transitional -> None
    'provisioning': None,
    'ordered': None,
    'new': None,
    'validating': None,
    # offline/discontinued/deleting/notfound: the box is gone or going.
    'offline': status_lib.ClusterStatus.STOPPED,
    'discontinued': status_lib.ClusterStatus.STOPPED,
    'deleting': status_lib.ClusterStatus.STOPPED,
    'notfound': status_lib.ClusterStatus.STOPPED,
    'unknown': None,
    # error/installation_failed/no_capacity: terminal failures -> STOPPED
    # so reconcile reaps them, never silently retried.
    'error': status_lib.ClusterStatus.STOPPED,
    'installation_failed': status_lib.ClusterStatus.STOPPED,
    'no_capacity': status_lib.ClusterStatus.STOPPED,
}

# The local cluster-identity store (see module docstring).
_STATE_PATH = os.path.expanduser('~/.sky/verda-clusters.json')


def _client() -> 'utils.VerdaClient':
    return utils.client_from_env()


def _sky_status(instance: Dict[str, Any]):
    """Map one instance's status, refusing to guess at an unknown state."""
    raw = utils.instance_status(instance)
    if raw in _STATUS_MAP:
        return _STATUS_MAP[raw]
    # Open-set tolerance: an unknown slug is logged verbatim and treated
    # as in-flight (a new slug must not crash the reconcile loop).
    logger.info(
        'verda instance %s in unmapped status %r (treated as '
        'in-flight)', instance.get('id'), raw)
    return None


def _is_live(instance: Dict[str, Any]) -> bool:
    """An instance that still exists on the project in a non-final state."""
    return utils.instance_status(instance) not in (utils.STATUS_OFFLINE,
                                                   utils.STATUS_DISCONTINUED,
                                                   utils.STATUS_DELETING,
                                                   utils.STATUS_NOTFOUND)


# -- local cluster-identity store -----------------------------------------
#
# Fail-loud rules: a corrupt file raises (never silently forgets a
# billing box); a missing file is an empty map.


def _load_state() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_STATE_PATH, encoding='utf-8') as handle:
            state = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise utils.VerdaError(
            f'cluster-identity store {_STATE_PATH} is corrupt ({exc}); '
            'refusing to forget live instances — fix or remove the file '
            'manually after checking the account instances') from exc
    if not isinstance(state, dict):
        raise utils.VerdaError(
            f'cluster-identity store {_STATE_PATH} is not a JSON object; '
            'refusing to forget live instances')
    return state


def _save_state(state: Dict[str, Dict[str, Any]]) -> None:
    # Atomic replace: a torn state file must never lose a billing box's
    # id.
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(_STATE_PATH),
                                    prefix='.clusters-',
                                    suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, _STATE_PATH)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@contextlib.contextmanager
def _state_lock():
    """Serialize every read-check-mutate-save transaction on the state
    file (the concurrent-writer class the quantacloud lane documented)."""
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    lock = FileLock(_STATE_PATH + '.lock', timeout=120)
    with lock:
        yield


def _forget(cluster_name_on_cloud: str,
            instance_id: Optional[str] = None) -> None:
    """Remove the mapping under the state lock, re-reading the CURRENT
    state first: the forget transaction is load -> check -> delete ->
    save, and with ``instance_id`` given it removes OUR OWN mapping only
    when it still names THAT instance (a concurrent writer's replacement
    is never clobbered)."""
    with _state_lock():
        state = _load_state()
        current_id = _mapped_id(state, cluster_name_on_cloud)
        if current_id is None:
            return
        if instance_id is not None and current_id != instance_id:
            return
        del state[cluster_name_on_cloud]
        _save_state(state)


def _mapped_id(state: Dict[str, Dict[str, Any]],
               cluster_name_on_cloud: str) -> Optional[str]:
    entry = state.get(cluster_name_on_cloud)
    if not isinstance(entry, dict):
        return None
    instance_id = entry.get('instance_id')
    return str(instance_id) if instance_id else None


def _location_code_for_region(client: 'utils.VerdaClient',
                              region_name: str) -> str:
    """Map the catalog Region NAME back to the location CODE.

    Refuses to guess: a name with no matching code is a mapping drift
    between the catalog and the live locations list, and a guessed code
    would deploy into the wrong country.
    """
    for location in client.list_locations():
        if str(location.get('name') or '') == region_name:
            code = location.get('code')
            if code:
                return str(code)
    raise utils.VerdaError(
        f'region {region_name!r} has no matching location code in '
        'GET /v1/locations; refusing to guess the deploy location')


# -- provisioning ----------------------------------------------------------


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Create (or adopt) the cluster's single instance and wait for
    running."""
    del cluster_name  # cluster_name_on_cloud is the identity we use
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)

    if instance_id is not None:
        # Adopt the mapped instance: terminal failures refuse, anything
        # else waits for `running` below. A mapped instance that turned
        # terminal-offline is forgotten and falls through to the create
        # path (the foreign-box check excludes terminal boxes via
        # _is_live).
        existing = client.get_instance(instance_id)
        if not _is_live(existing):
            _forget(cluster_name_on_cloud, instance_id)
            instance_id = None
        elif utils.instance_status(existing) in ('error',
                                                 'installation_failed'):
            raise utils.VerdaError(
                f'instance {instance_id} for cluster '
                f'{cluster_name_on_cloud!r} is in '
                f'{utils.instance_status(existing)}; terminate it before '
                'relaunching (not auto-retried — report and decide)')
    created = []
    if instance_id is None:
        # No mapping: is the project holding someone else's box? A live
        # instance we cannot attribute is a guess — refuse it.
        live_others = [i for i in client.list_instances() if _is_live(i)]
        if live_others:
            ids = [str(i.get('id')) for i in live_others]
            raise utils.VerdaError(
                f'cluster {cluster_name_on_cloud!r} has no recorded '
                f'instance, but {len(live_others)} live instance(s) exist '
                f'on the project ({ids}); refusing to guess which is '
                'ours — terminate the foreign box from the console or '
                'restore the state file, then relaunch')

        node_config = config.node_config
        instance_type = node_config.get('InstanceType')
        if not instance_type:
            raise utils.VerdaError(
                'node_config is missing InstanceType (the Verda '
                'instance-type code, e.g. 1H100.80S.22V)')
        # node_config['PublicKey'] is the ONE canonical source; the
        # placeholder check is the point (see the quantacloud twin).
        public_key = (node_config or {}).get('PublicKey')
        if isinstance(public_key, str):
            public_key = public_key.strip()
        if not public_key or public_key == _SSH_PUBLIC_KEY_PLACEHOLDER:
            raise utils.VerdaError(
                'node_config is missing its PublicKey, or the placeholder '
                f'was never substituted (got {public_key!r}); '
                'auth.configure_ssh_info must run before provisioning — '
                'a Verda instance without a real public key is '
                'unreachable')

        use_spot = bool(
            node_config.get('use_spot') or
            (config.provider_config or {}).get('use_spot'))
        ssh_key_id = client.ensure_ssh_key(utils.SSH_KEY_NAME, public_key)
        location_code = _location_code_for_region(client, region)
        instance = client.create_instance(
            instance_type=str(instance_type),
            image=utils.DEFAULT_IMAGE,
            hostname=cluster_name_on_cloud,
            ssh_key_ids=[ssh_key_id],
            location_code=location_code,
            is_spot=use_spot,
        )
        instance_id = str(instance['id'])
        created = [instance_id]

    # Readiness is `status == running` (ip is non-null only then); the
    # poll fails closed on terminal failures and on the deadline.
    instance = client.wait_until_running(instance_id)

    with _state_lock():
        state = _load_state()
        state[cluster_name_on_cloud] = {
            'instance_id': instance_id,
            'region': region,
            'ssh_user': SSH_USER,
            'instance_type': str(
                (config.node_config or {}).get('InstanceType') or ''),
        }
        _save_state(state)

    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=instance_id,
        resumed_instance_ids=[],
        created_instance_ids=created,
    )


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """No-op: run_instances already blocks until `running`."""
    del region, cluster_name_on_cloud, state


def stop_instances(**kwargs) -> None:
    """STOP is refused, not silently destroy-terminated (the cloud layer
    declares STOP unsupported; this is the belt to that suspenders)."""
    del kwargs
    raise utils.VerdaError(
        'stop is not supported on Verda: the API stop-shaped actions park '
        'a billing box in an unknown state. Use teardown.')


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Terminate the cluster's instance via the action endpoint."""
    del provider_config, worker_only  # single-node: head is the cluster
    if not str(cluster_name_on_cloud or '').strip():
        raise utils.VerdaError(
            'refusing to terminate with an empty cluster_name_on_cloud')
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    if instance_id is None:
        # Unmapped: an identity we cannot resolve must never destroy by
        # guess. Report the project's live instances for the operator.
        live = [
            str(i.get('id')) for i in client.list_instances() if _is_live(i)
        ]
        if not live:
            return  # nothing anywhere: already torn down
        raise utils.VerdaError(
            f'no recorded instance for cluster {cluster_name_on_cloud!r}; '
            f'live instances on the project: {live}. refusing to guess '
            'which is ours — operator decides (console teardown or '
            'state-file restore)')
    # delete_instance is idempotent (404 is success inside the client).
    client.delete_instance(instance_id)
    _forget(cluster_name_on_cloud, instance_id)
    logger.info('verda instance %s deleted', instance_id)


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region, provider_config
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    if instance_id is None:
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)
    try:
        instance = client.get_instance(instance_id)
    except utils.VerdaNotFoundError:
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)
    if not _is_live(instance):
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)

    # A None ip is narrowed to '' the quantacloud way (str(x or '')): a
    # live-but-unaddressed box must not be a TypeError in reconcile.
    ip = client.public_ip(instance) or ''
    stored_user = (state.get(cluster_name_on_cloud) or {}).get('ssh_user')
    return common.ClusterInfo(
        instances={
            instance_id: [
                common.InstanceInfo(
                    instance_id=instance_id,
                    internal_ip=ip,
                    external_ip=ip,
                    ssh_port=22,
                    tags={},
                    node_name=instance_id,
                )
            ]
        },
        head_instance_id=instance_id,
        provider_name=PROVIDER_NAME,
        ssh_user=str(stored_user or SSH_USER),
    )


#: The bounded read-after-write window for `query_instances`'s
#: `retry_if_missing` re-poll: a just-created instance can 404 on the
#: provider's first status poll until the create lands server-side.
_QUERY_RETRY_WINDOW_S = 60.0
_QUERY_RETRY_INTERVAL_S = 5.0


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    del cluster_name, provider_config, non_terminated_only
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    if instance_id is None:
        return {}
    # Bounded re-poll on the read-after-write 404 window (the quantacloud
    # -19 lesson): with retry_if_missing, re-poll before concluding
    # absence; a box that is REALLY gone forgets the mapping.
    deadline = time.monotonic(
    ) + _QUERY_RETRY_WINDOW_S if retry_if_missing else 0.0
    while True:
        try:
            instance = client.get_instance(instance_id)
            break
        except utils.VerdaNotFoundError:
            if time.monotonic() >= deadline:
                _forget(cluster_name_on_cloud, instance_id)
                return {}
            time.sleep(_QUERY_RETRY_INTERVAL_S)
    return {instance_id: (_sky_status(instance), None)}


def open_ports(cluster_name_on_cloud: str,
               ports: list,
               provider_config: Optional[Dict[str, Any]] = None) -> None:
    """No-op: Verda instances have no port firewall in the deploy path
    (direct SSH on port 22)."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(cluster_name_on_cloud: str,
                  ports: list,
                  provider_config: Optional[Dict[str, Any]] = None) -> None:
    """No-op counterpart to open_ports."""
    del cluster_name_on_cloud, ports, provider_config
