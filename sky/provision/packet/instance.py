"""Packet.ai instance provisioning.

Single-pod lifecycle on the Packet.ai API (https://dash.packet.ai/api/v1),
two Packet-specific wrinkles shape it (both MEASURED, ENG-562, 2026-10-15 —
the receipts on the ticket):

* **Instance creation is 501 NOT_IMPLEMENTED for API-key auth** — the
  deploy engine is console-session-path only today (the SPA POSTs the same
  ``/instances`` route with a productId-shaped body under a session Bearer
  token; the API-key path validates the doc shape and then refuses). The
  orchestrator ruling: the lane bridges the seam with **ADOPTION of
  console-launched boxes by instance NAME** (Packet instances carry a
  ``name`` — the console's launch form sets it, and the operator launches
  the lane's box with the cluster name), and ``create_instance`` raises the
  typed :class:`PacketCreateNotImplementedError` (never retried) so a
  create attempt surfaces as the platform-capability refusal it is. The
  adoption is BY EXACT NAME on the cluster-name convention, never by
  guess: an unmapped live box with a foreign name is refused (the operator
  decides), the same rule as QuantaCloud's.
* **Pool/product/region ids are ephemeral.** The catalog's InstanceType is
  the stable ``<accelerator>:<count>`` token; at launch this module
  re-resolves the concrete product (in-stock), pool, and NUMERIC
  ``region_id`` from the LIVE launch-options (the surface flipped its
  product list mid-session on 2026-10-15 — a frozen id is silent rot).

Everything else is the quantacloud shape: a LOCAL cluster-identity store
(``~/.sky/packet-clusters.json``) mapping cluster-name -> instance_id,
fail-loud on corrupt files, refuse-to-guess on unmapped live boxes; STOP is
a REAL stop on Packet (billing stops, box+disk persist) unlike QuantaCloud
whose stop destroys; teardown is DELETE (root disk lost); ``GET
/instances/{full-id}/connection`` returns the PUBLIC ip + port + user +
password (the password is a live secret — this module reads the user and
ip, and NEVER stores the password: key auth is the measured primary path,
the account key registered by material before launch was auto-installed).

The SSH user is MEASURED ``ubuntu`` (2026-10-15: measured on the live box
over password AND key auth; the constant below is only the fallback — the
connection endpoint reports it per instance and is read first).
"""

from __future__ import annotations

import json
import os
import threading
import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.packet import utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    pass

logger = sky_logging.init_logger(__name__)

PROVIDER_NAME = 'packet'

# MEASURED (ENG-562, 2026-10-15): ubuntu — over password AND key auth, sudo
# passwordless. The live connection endpoint's `username` is read first;
# this constant is only the fallback for a missing field.
SSH_USER = 'ubuntu'

# The literal the ray template carries until `configure_ssh_info`
# substitutes the real key into node_config['PublicKey'] (see the check in
# run_instances for why the literal itself must be refused).
_SSH_PUBLIC_KEY_PLACEHOLDER = 'skypilot:ssh_public_key_content'

# Packet.ai instance status -> SkyPilot cluster status.
#
# The docs give a closed set (running | stopped | starting | stopping |
# error, transitional pending/restarting), but the Latitude lesson
# (provisioning slugs returned verbatim, open set) says treat it as open
# anyway: mapped steady states only; anything else maps to None =
# transitional (SkyPilot keeps waiting) and is logged verbatim. Terminal
# states surface as STOPPED (the ClusterStatus enum has no TERMINATED) so
# reconcile reaps them.
_STATUS_MAP: Dict[str, Optional[status_lib.ClusterStatus]] = {
    'running': status_lib.ClusterStatus.UP,
    'stopped': status_lib.ClusterStatus.STOPPED,
    'error': status_lib.ClusterStatus.STOPPED,  # terminal failure -> reap
    'failed': status_lib.ClusterStatus.STOPPED,  # terminal failure -> reap
    'starting': None,  # transitional
    'stopping': None,  # transitional
    'restarting': None,  # transitional
    'pending': None,  # transitional
    'terminated': status_lib.ClusterStatus.STOPPED,
}

# The local cluster-identity store (see module docstring). Packet instances
# carry a name, but the name is CONSOLE-ASSIGNED at launch (the API-key path
# cannot create today), so this module keeps the authoritative mapping
# anyway — a renamed console box must not orphan a running lane box.
_STATE_PATH = os.path.expanduser('~/.sky/packet-clusters.json')

_STATE_LOCK = threading.Lock()


def _client() -> 'utils.PacketClient':
    return utils.client_from_env()


def _instance_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The {id} route wraps the instance at ``data.instance`` (with metadata
    beside it); some shapes return the instance directly. Refuse neither."""
    instance = payload.get('instance') if isinstance(payload, dict) else None
    if isinstance(instance, dict):
        return instance
    if isinstance(payload, dict):
        return payload
    return {}


def _sky_status(instance: Dict[str, Any]):
    """Map one instance's status, refusing to guess at an unknown state."""
    status = utils.instance_status(instance)
    if not status:
        return None
    mapped = _STATUS_MAP.get(status)
    if mapped is None and status not in _STATUS_MAP:
        logger.warning(
            'packet instance %s: unmapped status %r (treated as '
            'transitional)', instance.get('id'), status)
    return mapped


def _is_live(instance: Dict[str, Any]) -> bool:
    """An instance that still exists on the account in a non-final state."""
    status = utils.instance_status(instance)
    return bool(status) and status != 'terminated'


# -- local cluster-identity store -----------------------------------------
#
# Fail-loud rules: a corrupt file raises (never silently forgets a billing
# box); a missing file is an empty map.


def _load_state() -> Dict[str, Dict[str, Any]]:
    with _STATE_LOCK:
        if not os.path.exists(_STATE_PATH):
            return {}
        try:
            with open(_STATE_PATH, encoding='utf-8') as handle:
                state = json.load(handle)
        except (ValueError, OSError) as exc:
            raise utils.PacketError(
                f'packet cluster state file {_STATE_PATH} is corrupt '
                f'({exc}); refusing to silently forget billing instances — '
                'fix or remove the file by hand') from exc
        if not isinstance(state, dict):
            raise utils.PacketError(
                f'packet cluster state file {_STATE_PATH} is corrupt '
                f'(expected an object, got {type(state).__name__})')
        return state


def _save_state(state: Dict[str, Dict[str, Any]]) -> None:
    with _STATE_LOCK:
        os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
        tmp = f'{_STATE_PATH}.tmp.{os.getpid()}'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(state, handle, indent=2)
        os.replace(tmp, _STATE_PATH)


def _forget(state: Dict[str, Dict[str, Any]],
            cluster_name_on_cloud: str) -> None:
    if cluster_name_on_cloud in state:
        del state[cluster_name_on_cloud]
        _save_state(state)


def _mapped_id(state: Dict[str, Dict[str, Any]],
               cluster_name_on_cloud: str) -> Optional[str]:
    entry = state.get(cluster_name_on_cloud)
    if not isinstance(entry, dict):
        return None
    instance_id = entry.get('instance_id')
    return str(instance_id) if instance_id else None


# -- launch-time product re-resolution -------------------------------------


def _resolve_launch_target(
    client: 'utils.PacketClient',
    accelerator: str,
    gpu_count: int,
    region_name: str,
    max_price_per_gpu: Optional[float] = None,
) -> Tuple[str, str, int]:
    """Re-resolve (product id, pool id, numeric region id) from the LIVE
    launch-options for the token's (accelerator, count, region).

    Pool/product/region ids are ephemeral (the surface flipped its product
    list mid-session on 2026-10-15) — a frozen id from catalog time is
    silent rot. Fails closed on no match (the offer behind the catalog row
    went out of stock), never rents a different shape.
    """
    # pylint: disable=import-outside-toplevel
    from sky.catalog.data_fetchers import fetch_packet

    options = client.get_launch_options()
    products = [p for p in options['products'] if isinstance(p, dict)]
    pools = [p for p in options['pools'] if isinstance(p, dict)]
    regions = [r for r in options['regions'] if isinstance(r, dict)]

    region_id = None
    for region in regions:
        if str(region.get('name') or '') == region_name:
            raw = region.get('id')
            if isinstance(raw, int):
                region_id = raw
                break
    if region_id is None:
        raise utils.PacketError(
            f'no Packet.ai region named {region_name!r} in the live '
            'launch-options; the catalog row names a region the provider '
            'no longer serves — fail closed rather than rent elsewhere')

    for product in products:
        name = str(product.get('name') or '')
        if fetch_packet.normalise_accelerator(name) != accelerator.upper():
            continue
        try:
            available = int(product.get('totalAvailableGpus') or 0)
        except (TypeError, ValueError):
            continue
        if available < gpu_count:
            continue
        try:
            cents = int(product.get('pricePerHourCents') or 0)
        except (TypeError, ValueError):
            continue
        per_gpu = cents / 100.0
        if max_price_per_gpu is not None and per_gpu > max_price_per_gpu:
            continue
        product_regions = {
            str(r.get('name') or '')
            for r in (product.get('regions') or [])
            if isinstance(r, dict)
        }
        if region_name not in product_regions:
            continue
        pool_ids = {
            str(p.get('id'))
            for p in pools
            if isinstance(p.get('id'), (int, str)) and str(
                (p.get('region') or {}).get('name') or '') == region_name
        }
        for raw_pool_id in (product.get('poolIds') or []):
            pool_id = str(raw_pool_id)
            if pool_id in pool_ids:
                return str(product.get('id')), pool_id, region_id
    raise utils.PacketError(
        f'no in-stock Packet.ai product for {accelerator!r} x {gpu_count} in '
        f'region {region_name!r}' +
        (f' at <= ${max_price_per_gpu}/GPU-hr' if max_price_per_gpu else '') +
        '; the offer behind the catalog row went out of stock '
        '(pool/product/region ids are ephemeral) — fail closed rather than '
        'rent a different shape')


# -- provisioning ----------------------------------------------------------


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Create (or ADOPT) the cluster's single instance and wait for running.

    Adoption order (the ENG-562 ruling — creation is 501 for API keys):

    1. a recorded mapping (the normal resume path);
    2. a live instance NAMED exactly ``cluster_name_on_cloud`` — the
       console-launched bridge (the operator launched the lane's box from
       the console with the cluster name because the API-key path cannot
       create today);
    3. create — which raises the typed
       :class:`PacketCreateNotImplementedError` today (fail-closed, never
       retried; the message names the console path).
    """
    del cluster_name  # cluster_name_on_cloud is the identity we use
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    instance: Optional[Dict[str, Any]] = None

    if instance_id is not None:
        try:
            instance = _instance_from_payload(client.get_instance(instance_id))
        except utils.PacketNotFoundError:
            # Stale mapping (already terminated server-side): forget it.
            _forget(state, cluster_name_on_cloud)
            instance_id = None
        else:
            if not _is_live(instance):
                _forget(state, cluster_name_on_cloud)
                instance_id = None
            else:
                status = utils.instance_status(instance)
                if status in (utils.STATUS_ERROR, utils.STATUS_FAILED):
                    raise utils.PacketError(
                        f'instance {instance_id} for cluster '
                        f'{cluster_name_on_cloud!r} is {status}; terminate '
                        'it before relaunching (not auto-retried — report '
                        'and decide)')
                # Transitional or running: adopt and wait below.

    if instance_id is None:
        # The console-launched bridge: a live instance carrying the
        # cluster's NAME is ours to adopt (the operator launched it from
        # the console with the cluster name — the only creation path today).
        # Matching is EXACT (never a guess): every other live box is
        # reported and refused below.
        for candidate in client.list_instances():
            if not _is_live(candidate):
                continue
            if utils.instance_name(candidate) == cluster_name_on_cloud:
                instance_id = str(candidate.get('id'))
                instance = candidate
                break

    created: List[str] = []
    if instance_id is None:
        live_others = [
            str(d.get('id')) for d in client.list_instances() if _is_live(d)
        ]
        if live_others:
            # A live instance we cannot attribute is a guess — refuse it (a
            # console experiment or a lost state file; the operator
            # decides). Adoption is by EXACT name only.
            raise utils.PacketError(
                f'cluster {cluster_name_on_cloud!r} has no recorded '
                f'instance and no instance carries its name, but '
                f'{len(live_others)} live instance(s) exist on the account '
                f'({live_others}); refusing to guess which is ours — '
                'terminate the foreign box from the console, or relaunch '
                'the lane\'s box FROM THE CONSOLE with the cluster name '
                f'({cluster_name_on_cloud!r}), then relaunch')

        # No mapping, no name-match, no foreign boxes: CREATE. Today this
        # raises the typed PacketCreateNotImplementedError (fail-closed,
        # never retried); the day the engine is ported for API keys, this
        # exact path provisions with zero changes.
        # pylint: disable=import-outside-toplevel
        from sky.catalog.data_fetchers import fetch_packet

        node_config = config.node_config or {}
        token = node_config.get('InstanceType')
        if not token:
            raise utils.PacketError(
                'node_config is missing InstanceType (the Packet.ai '
                '"<accelerator>:<count>" catalog token, e.g. rtxpro6000:1)')
        accelerator, gpu_count = fetch_packet.parse_instance_type_token(
            str(token))

        # node_config['PublicKey'] is the ONE canonical source:
        # backend_utils puts Packet on the generic `auth.configure_ssh_info`
        # path, which substitutes the real key into the template's
        # `skypilot:ssh_public_key_content` placeholder. The placeholder
        # check is the point: if substitution did NOT run, the literal is
        # still TRUTHY, a bare `if not public_key` waves garbage into
        # ensure_ssh_key, and the deploy fails far from the cause.
        public_key = node_config.get('PublicKey')
        if isinstance(public_key, str):
            public_key = public_key.strip()
        if not public_key or public_key == _SSH_PUBLIC_KEY_PLACEHOLDER:
            raise utils.PacketError(
                'node_config is missing its PublicKey, or the placeholder '
                f'was never substituted (got {public_key!r}); '
                'auth.configure_ssh_info must run before provisioning — '
                'a Packet.ai instance without a real public key is '
                'unreachable')

        # Launch-time re-resolution: the token names the (accelerator,
        # count) pair; the concrete product/pool/region ids come from the
        # LIVE launch-options (ids are ephemeral — a frozen one is silent
        # rot). Keys install at LAUNCH only: ensure the account carries the
        # key material before creating.
        product_id, pool_id, region_id = _resolve_launch_target(
            client, accelerator, gpu_count, region)
        client.ensure_ssh_key(utils.SSH_KEY_NAME, public_key)
        # NOTE: create_instance raises PacketCreateNotImplementedError
        # today (API-key path); the body below is the validated doc shape
        # so the ported engine is a drop-in.
        payload = client.create_instance(
            name=cluster_name_on_cloud,
            pool_id=pool_id,
            region_id=region_id,
            vgpus=gpu_count,
            ssh_key_name=utils.SSH_KEY_NAME,
            public_key=public_key,
        )
        del product_id  # pool_id is what the create body takes
        instance_id = str(payload.get('instance_id') or payload.get('id') or '')
        if not instance_id:
            raise utils.PacketError(f'POST /instances returned no instance id: '
                                    f'{sorted(payload)}')
        created = [instance_id]

    # Readiness is `status == running`; the poll fails closed on
    # error/failed and on the deadline.
    instance = _instance_from_payload(client.get_instance(instance_id))
    client.wait_until_running(instance_id)

    # Record the identity + the provider-reported SSH user for every later
    # operation (query/cluster-info/terminate) — the connection endpoint
    # is the source of truth (MEASURED: it returns the PUBLIC ip; the
    # password half of that response is a live secret and is never
    # stored — key auth is the measured primary path).
    ssh_user = SSH_USER
    try:
        connection = client.get_connection(instance_id)
        ssh_user = utils.ssh_user_from_connection(connection) or SSH_USER
    except utils.PacketError as exc:
        logger.warning(
            'packet instance %s: connection endpoint refused '
            'while recording ssh_user (%s); falling back to the '
            'measured constant %r', instance_id, exc, SSH_USER)
    state = _load_state()
    state[cluster_name_on_cloud] = {
        'instance_id': instance_id,
        'region': region,
        'ssh_user': ssh_user,
        'instance_type': str((config.node_config or {}).get('InstanceType') or
                             ''),
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


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Stop the cluster's instance — a REAL stop on Packet.ai.

    MEASURED semantics (unlike QuantaCloud, whose stop destroys):
    ``POST /instances/{id}/stop`` stops hourly billing while the box and
    its disk persist (``start`` resumes). The dispatch contract
    (sky/provision/__init__.py stop_instances wrapper after ``provider_name``
    is stripped) is this signature — the ``(region, cluster_name, ...)``
    shape copied from vast does not BIND and would TypeError at dispatch.
    """
    del provider_config, worker_only  # single-node: head is the cluster
    if not str(cluster_name_on_cloud or '').strip():
        # An empty identity maps to NOTHING here — but the refusal stays
        # loud so a miswired caller can never act by guess.
        raise utils.PacketError(
            'refusing to stop with an empty cluster_name_on_cloud')
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    if instance_id is None:
        return  # unmapped: nothing of ours to stop (terminate refuses
        # instead — see terminate_instances — because teardown must be
        # loud, but a stop of nothing is a no-op).
    try:
        client.stop_instance(instance_id)
    except utils.PacketNotFoundError:
        pass  # already gone server-side
    logger.info('packet instance %s stopped', instance_id)


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Terminate the cluster's instance (DELETE; root disk lost)."""
    del provider_config, worker_only  # single-node: head is the cluster
    if not str(cluster_name_on_cloud or '').strip():
        # An empty identity filters NOTHING — but the refusal stays loud
        # so a miswired caller can never destroy by guess (CodeRabbit on
        # the Latitude PR, same class).
        raise utils.PacketError(
            'refusing to terminate with an empty cluster_name_on_cloud')
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    if instance_id is None:
        # Unmapped: an identity we cannot resolve must never destroy by
        # guess. Report the account's live instances for the operator.
        live = [
            str(d.get('id')) for d in client.list_instances() if _is_live(d)
        ]
        if not live:
            return  # nothing anywhere: already torn down
        raise utils.PacketError(
            f'no recorded instance for cluster {cluster_name_on_cloud!r}; '
            f'live instances on the account: {live}. refusing to guess '
            'which is ours — operator decides (console teardown or '
            'state-file restore)')
    # delete_instance is idempotent (404 is success inside the client).
    client.delete_instance(instance_id)
    _forget(state, cluster_name_on_cloud)
    logger.info('packet instance %s terminated (deleted)', instance_id)


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
        payload = client.get_instance(instance_id)
    except utils.PacketNotFoundError:
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)
    instance = _instance_from_payload(payload)
    if not _is_live(instance):
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)

    # The connection endpoint is the source of truth for ip/port/user
    # (MEASURED: it returns the PUBLIC ip — the instance record's `ip`
    # array agrees when populated; the {id} route needs the FULL id).
    # The password in that response is a live secret: read the fields we
    # need, never store or log the rest.
    try:
        connection = client.get_connection(instance_id)
        ip = str(connection.get('ip') or '')
        ssh_port = int(connection.get('port') or 22)
        ssh_user = (utils.ssh_user_from_connection(connection) or str(
            (state.get(cluster_name_on_cloud) or {}).get('ssh_user') or '') or
                    SSH_USER)
    except utils.PacketError:
        ip = ''
        ips = instance.get('ip')
        if isinstance(ips, list) and ips:
            ip = str(ips[0])
        ssh_port = 22
        ssh_user = (str(
            (state.get(cluster_name_on_cloud) or {}).get('ssh_user') or '') or
                    SSH_USER)
    if not ip:
        ips = instance.get('ip')
        if isinstance(ips, list) and ips:
            ip = str(ips[0])
    return common.ClusterInfo(
        instances={
            instance_id: [
                common.InstanceInfo(
                    instance_id=instance_id,
                    # A Packet.ai pod exposes one public address; there is
                    # no separate internal address to report.
                    internal_ip=ip,
                    external_ip=ip,
                    ssh_port=ssh_port,
                    tags={},
                    node_name=str(instance.get('name') or instance_id),
                )
            ]
        },
        head_instance_id=instance_id,
        provider_name=PROVIDER_NAME,
        ssh_user=ssh_user,
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    del cluster_name, provider_config, non_terminated_only
    client = _client()
    state = _load_state()
    instance_id = _mapped_id(state, cluster_name_on_cloud)
    if instance_id is None:
        return {}
    try:
        payload = client.get_instance(instance_id)
    except utils.PacketNotFoundError:
        # Gone server-side: forget the mapping so nothing resurrects it,
        # and report absence (the same signal a server list gives when
        # the box was deleted).
        _forget(state, cluster_name_on_cloud)
        return {}
    instance = _instance_from_payload(payload)
    if not _is_live(instance):
        _forget(state, cluster_name_on_cloud)
        return {}
    return {instance_id: (_sky_status(instance), None)}


def open_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op: Packet.ai pods expose direct SSH on a per-instance port with
    no firewall in the deploy path (MEASURED: the connection endpoint
    returns the public ip + port; port exposure beyond SSH rides the
    provider's NodePort surface POST /api/v1/services when a lane needs
    it)."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: list,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op counterpart to open_ports."""
    del cluster_name_on_cloud, ports, provider_config
