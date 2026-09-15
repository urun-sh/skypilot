"""Spheron provisioner for SkyPilot.

Registered via ``sky.provision.register_provisioner`` — SkyPilot's dispatch
explicitly prefers a registered provisioner's methods over the static cloud
module's, so this needs no fork.

Everything here that could have been guessed was instead **measured against the
live API on 2026-09-11**, using a $0.12/hr CPU node. Three findings contradict
the vendor manual and are the reason this file looks the way it does:

1. **``teamId`` is REQUIRED on create.** The manual (section 6) lists it under
   "Optional fields — defaults to the current team". It does not: omitting it
   returns ``400: Team ID is required for authenticated deployments``.
2. **``sshCommand`` does not exist in the response.** The manual (section 8)
   says ``running`` populates ``sshCommand`` and ``ipAddress``. Only
   ``ipAddress`` appears; there is no ``sshCommand`` key at all. The SSH
   command must therefore be composed here, which means the login user has to
   be known rather than read.
3. **``ipAddress`` is populated while still ``deploying``.** It was present on
   the very first poll, ~1s after create, long before ``running``. A caller
   that treats a non-null IP as "ready" will connect too early; readiness is
   the *status*, never the presence of an address.

Measured login user: ``ubuntu``. Connecting as ``root`` is refused outright
with "Please login as the user ubuntu rather than the user root."
"""

from __future__ import annotations

import os
import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.utils import status_lib

from sky.adaptors import spheron as api

if typing.TYPE_CHECKING:
    pass

logger = sky_logging.init_logger(__name__)

PROVIDER_NAME = "spheron"

# Measured 2026-09-11 on massed-compute Ubuntu Server 22.04. Root is refused.
SSH_USER = "ubuntu"
SSH_PORT = 22

# Spheron status -> SkyPilot cluster status.
# `stopped` maps to STOPPED even though the cloud class declares STOP
# unsupported: an instance can still be stopped out-of-band from the dashboard,
# and reporting it as UP would be a lie.
_STATUS_MAP: Dict[str, Optional[status_lib.ClusterStatus]] = {
    "deploying": status_lib.ClusterStatus.INIT,
    "running": status_lib.ClusterStatus.UP,
    "stopped": status_lib.ClusterStatus.STOPPED,
    # Gone for good; represented as absent rather than any live status.
    "failed": None,
    "terminated": None,
    "terminated-provider": None,
}


def _client() -> api.SpheronClient:
    key = os.environ.get("SPHERON_API_KEY", "").strip()
    if not key:
        path = os.path.expanduser("~/.spheron/api_key")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                key = handle.read().strip()
    if not key:
        raise api.SpheronError(
            "SPHERON_API_KEY is not set and ~/.spheron/api_key is absent"
        )
    return api.SpheronClient(key)


def _deployments_for_cluster(
    client: api.SpheronClient, cluster_name_on_cloud: str
) -> List[Dict[str, Any]]:
    """Deployments belonging to this cluster.

    SkyPilot has no tag mechanism here, so the deployment ``name`` carries the
    cluster identity. ``name`` is the only mutable field on a deployment, which
    makes it the correct (and only) place to put it.
    """
    return [
        d
        for d in client.list_deployments()
        if str(d.get("name") or "") == cluster_name_on_cloud
    ]


def _sky_status(deployment: Dict[str, Any]):
    """Map one deployment's status, refusing to guess at an unknown one.

    ``_STATUS_MAP.get(..., None)`` would silently classify a NEW vendor status
    as terminated, and both consequences are expensive: ``run_instances`` would
    create a SECOND deployment alongside a live one, and ``terminate_instances``
    would skip a deployment that is still billing. Same fail-loud rule as
    ``query_instances``. (CodeRabbit on #338.)
    """
    raw = str(deployment.get("status") or "").lower()
    if raw not in _STATUS_MAP:
        raise api.SpheronError(
            f"unknown Spheron status {raw!r} on deployment "
            f"{deployment.get('id')}; refusing to guess whether it is live"
        )
    return _STATUS_MAP[raw]


def _live(deployments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deployments that still exist (running, deploying or stopped).

    `stopped` counts as live on purpose: it still holds the disk and still
    bills, so terminate_instances must be able to clean it up. run_instances
    guards against ADOPTING one separately.
    """
    return [d for d in deployments if _sky_status(d) is not None]


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Create (or adopt) the cluster's single instance and wait for running."""
    del cluster_name  # cluster_name_on_cloud is the identity we use
    client = _client()

    existing = _live(_deployments_for_cluster(client, cluster_name_on_cloud))
    if len(existing) > 1:
        raise api.SpheronError(
            f"cluster {cluster_name_on_cloud!r} has {len(existing)} live "
            "deployments; refusing to guess which is the head"
        )

    if existing:
        deployment = existing[0]
        status = str(deployment.get("status") or "").lower()
        if status == api.STATUS_STOPPED:
            # `stopped` is live (disk kept, still billing) so _live() returns
            # it, but the Spheron client has no start/resume call -- adopting
            # one would hand wait_until_running a deployment that can never
            # reach `running`, and the launch would burn the full timeout
            # before failing. Say what is wrong instead. (CodeRabbit on #338.)
            raise api.SpheronError(
                f"deployment {deployment.get('id')} for cluster "
                f"{cluster_name_on_cloud!r} is STOPPED and Spheron exposes no "
                "resume in this integration; terminate it before relaunching "
                "(note: terminating deletes the disk)"
            )
        created = []
    else:
        node_config = config.node_config
        provider = node_config.get("spheron_provider")
        offer_id = node_config.get("spheron_offer_id")
        if not provider or not offer_id:
            raise api.SpheronError(
                "node_config is missing spheron_provider/spheron_offer_id; "
                'the instance type must be "{provider}_{offerId}"'
            )

        offer = _resolve_offer(client, provider, offer_id)
        ssh_key_id = _ensure_key(client, config)

        request = api.DeploymentRequest.from_offer(
            offer,
            region=region,
            operating_system=offer.pick_os(["cuda"] if offer.gpu_count else None),
            ssh_key_id=ssh_key_id,
            # REQUIRED despite the manual calling it optional -- see module docstring.
            team_id=client.current_team_id(),
            name=cluster_name_on_cloud,
        )
        deployment = client.create_deployment(request)
        created = [str(deployment.get("id"))]

    deployment_id = str(deployment.get("id"))
    # Readiness is the STATUS. ipAddress appears during `deploying` and must
    # never be used as a readiness signal.
    client.wait_until_running(deployment_id)

    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=deployment_id,
        resumed_instance_ids=[],
        created_instance_ids=created,
    )


def _resolve_offer(
    client: api.SpheronClient, provider: str, offer_id: str
) -> api.Offer:
    for offer in client.list_offers():
        if offer.provider == provider and offer.offer_id == offer_id:
            if offer.maintenance:
                raise api.SpheronError(f"offer {offer_id!r} is flagged for maintenance")
            return offer
    raise api.SpheronError(
        f"offer {provider}/{offer_id} is no longer listed; the catalog is "
        "stale relative to live availability"
    )


def _ensure_key(client: api.SpheronClient, config: common.ProvisionConfig) -> str:
    public_key = (config.node_config or {}).get("PublicKey")
    if not public_key:
        path = os.path.expanduser("~/.ssh/sky-key.pub")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                public_key = handle.read().strip()
    if not public_key:
        raise api.SpheronError(
            "no SSH public key available; a Spheron deployment without one is "
            "unreachable"
        )
    return client.ensure_ssh_key("skypilot", public_key)


def wait_instances(
    region: str, cluster_name_on_cloud: str, state: Optional[status_lib.ClusterStatus]
) -> None:
    """No-op: run_instances already blocks until `running`."""
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Not supported.

    Stopping is per-offer (`supportsPause`) and unavailable on much of the
    fleet -- measured on massed-compute: "Massed Compute does not support
    stopping an instance. It keeps billing at the full hourly rate until you
    destroy it." Failing loudly is the only safe answer: a stop that silently
    became a terminate would destroy the disk.
    """
    del provider_config, worker_only
    raise NotImplementedError(
        f"Spheron does not support stopping {cluster_name_on_cloud}; "
        "terminate instead (this destroys the disk)."
    )


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """Terminate the cluster's deployments.

    A minimum runtime applies and is charged in full even on an early
    terminate, so the remaining time is logged rather than silently absorbed.
    """
    del provider_config
    if worker_only:
        return  # single-node only; there are no workers to terminate
    client = _client()
    live = _live(_deployments_for_cluster(client, cluster_name_on_cloud))
    # SAME GUARD AS run_instances, because this path is destructive. `name` is
    # mutable and account-wide, so an unrelated or hand-made deployment sharing
    # it would otherwise be destroyed by `sky down`. Refusing is the only safe
    # answer: terminate deletes the disk and cannot be undone.
    # (Greptile on #338.)
    if len(live) > 1:
        raise api.SpheronError(
            f"cluster {cluster_name_on_cloud!r} matches {len(live)} live "
            f"deployments ({[d.get('id') for d in live]}); refusing to "
            "terminate because `name` is account-wide and mutable, so one of "
            "these may not belong to this cluster"
        )
    for deployment in live:
        deployment_id = str(deployment.get("id"))
        try:
            check = client.can_terminate(deployment_id)
            if not check.get("canTerminate"):
                logger.info(
                    f"spheron: {deployment_id} not yet terminable "
                    f"({check.get('reason')}); minimumRuntime="
                    f"{check.get('minimumRuntime')}min, timeRemaining="
                    f"{check.get('timeRemaining')}min. Issuing terminate anyway "
                    "-- the provider minimum is billed regardless."
                )
        except api.SpheronError as exc:
            logger.info(
                f"spheron: can-terminate check failed for {deployment_id}: {exc}"
            )
        client.terminate_deployment(deployment_id)


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region, provider_config
    client = _client()
    deployments = _live(_deployments_for_cluster(client, cluster_name_on_cloud))
    if not deployments:
        return common.ClusterInfo(
            instances={}, head_instance_id=None, provider_name=PROVIDER_NAME
        )

    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = None
    for deployment in deployments:
        deployment_id = str(deployment.get("id"))
        ip = deployment.get("ipAddress") or ""
        instances[deployment_id] = [
            common.InstanceInfo(
                instance_id=deployment_id,
                # Spheron exposes one public address; there is no separate
                # internal address to report.
                internal_ip=ip,
                external_ip=ip,
                ssh_port=SSH_PORT,
                tags={},
                node_name=str(deployment.get("name") or deployment_id),
            )
        ]
        if head_instance_id is None:
            head_instance_id = deployment_id

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name=PROVIDER_NAME,
        # Composed, not read: the API returns no sshCommand field at all.
        ssh_user=SSH_USER,
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    del cluster_name, provider_config
    client = _client()
    result: Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]] = {}
    for deployment in _deployments_for_cluster(client, cluster_name_on_cloud):
        raw_status = str(deployment.get("status") or "").lower()
        if raw_status not in _STATUS_MAP:
            raise api.SpheronError(
                f"unknown Spheron status {raw_status!r} on deployment "
                f"{deployment.get('id')}; refusing to guess its meaning"
            )
        sky_status = _STATUS_MAP[raw_status]
        if sky_status is None:
            continue  # gone

        result[str(deployment.get("id"))] = (sky_status, None)
    return result


def open_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op: Spheron exposes no port/firewall API (OpenPortsVersion.LAUNCH_ONLY)."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op counterpart to open_ports."""
    del cluster_name_on_cloud, ports, provider_config
