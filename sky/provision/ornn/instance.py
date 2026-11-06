"""Ornn instance provisioning.

The launch path is the **spot bid's auto-launch**: ``run_instances`` places
a MARKET bid (1-8 GPUs on ONE node) and a fill auto-starts a single isolated
VM from the org's first custom image (or Ornn's default) with every active
account SSH key attached. There is no explicit instance-create, no image
argument on the bid, and manual ``ornn_access_activate`` is NOT in the path
(it can refuse ``image_required`` on a spot reservation — reproduced twice
2026-10-08, ENG-541 Q4 — and is carried as a typed, never-retried error).

Attribution lives in the local identity store (``~/.sky/ornn-clusters.json``):
the bid API has NO name/hostname field, so cluster_name_on_cloud ->
reservation id is recorded at bid time and an UNATTRIBUTED reservation is
invisible to terminate/reconcile by design (the ownership gate).

Wait semantics fail CLOSED on: a cancelled/ended reservation (the market
bid was refused — a capacity refusal, surfaced, never blind-retried), a
completed reservation that never exposed SSH machines (the wedge class,
filed with Ornn), and the deadline (Ornn applies Ubuntu security updates
BEFORE the SSH port opens — first boot takes minutes longer, which the
default timeout tolerates, and then fails loud).

Teardown is DELETE-only: ``ornn_spot_cancel`` (orderId == reservationId
while running) ENDS the reservation and destroys the VM + its local disk;
``stop_instances`` refuses (a stop that destroys would be the worst of
both), and 404/already-ended is idempotent success. Local files on the VM
are destroyed with the disk — export before terminating.
"""

from __future__ import annotations

import time

import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.ornn import utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    pass

logger = sky_logging.init_logger(__name__)

PROVIDER_NAME = "ornn"

# The literal the ray template carries until `configure_ssh_info`
# substitutes the real key into node_config['PublicKey'] (the check in
# run_instances refuses the literal unsubstituted — a spot VM without our
# account key material is unreachable).
_SSH_PUBLIC_KEY_PLACEHOLDER = "skypilot:ssh_public_key_content"

# Ornn reservation status -> SkyPilot cluster status. The status set is
# OPEN (receipts 2026-10-08): `active` (running), `completed`/`cancelled`
# (ended). Anything else maps to None = transitional (keep waiting);
# the wait loop fails closed on the deadline and on ended-without-machines.
_STATUS_MAP: Dict[str, Optional[status_lib.ClusterStatus]] = {
    "active": status_lib.ClusterStatus.UP,
    "completed": status_lib.ClusterStatus.STOPPED,
    "cancelled": status_lib.ClusterStatus.STOPPED,
}


def _client() -> utils.OrnnClient:
    return utils.client_from_env()


def _sky_status(
        reservation: Dict[str, Any]) -> Optional[status_lib.ClusterStatus]:
    """Map one reservation's status, refusing to guess at unknown values."""
    raw = str(reservation.get("status") or "").strip().lower()
    return _STATUS_MAP.get(raw)


def _is_live(reservation: Dict[str, Any]) -> bool:
    """A reservation that still exists (and bills) in any state.

    Ornn expresses teardown by the reservation ENDING (``endsAt`` set,
    status completed/cancelled) — so every reservation we can still SHOW is
    live. This predicate exists so adoption/terminate read correctly and
    reconcile can never orphan a billed box.
    """
    status = str(reservation.get("status") or "").strip().lower()
    return status not in (utils.STATUS_COMPLETED, utils.STATUS_CANCELLED)


def _parse_instance_type(node_config: Dict[str, Any]) -> Tuple[str, int]:
    """The (gpu_slug, gpus) pair from the stable ``ornn:<slug>:<gpus>``
    token — raising loudly on garbage (never bidding with a bad slug)."""
    from sky.catalog.data_fetchers import fetch_ornn

    token = str(node_config.get("InstanceType") or "").strip()
    if not token:
        raise utils.OrnnError("node_config is missing InstanceType (the stable "
                              "'ornn:<gpu-slug>:<gpus>' token, e.g. "
                              "ornn:nvidia_rtx_pro_6000:1)")
    return fetch_ornn.parse_instance_type_token(token)


def _ensure_account_ssh_key(client: utils.OrnnClient, public_key: str) -> str:
    """Ensure the deployment's key exists as an ACCOUNT key; return its id.

    Spot VMs auto-attach every active account key at launch, so the
    deployment's public key must be an account key BEFORE the bid. Matched
    on key MATERIAL (not name/label): a re-created controller keypair with
    the same name never selects a stale key, and a rename never mints a
    duplicate.
    """
    wanted = public_key.strip().split()
    wanted_material = wanted[1] if len(wanted) > 1 else public_key.strip()
    for key in client.list_ssh_keys():
        material = str(key.get("public_key") or "").strip().split()
        key_material = material[1] if len(material) > 1 else ""
        if key_material and key_material == wanted_material:
            key_id = key.get("id")
            if key_id:
                return str(key_id)
    created = client.add_ssh_key(public_key.strip(), label=utils.SSH_KEY_LABEL)
    key_id = created.get("id") or (created.get("ssh_key") or {}).get("id")
    if not key_id:
        raise utils.OrnnError("ornn_ssh_keys_add returned no key id: "
                              f"{sorted(created)}")
    return str(key_id)


def _activate_with_retry(client: utils.OrnnClient,
                         reservation_id: str,
                         key_id: str,
                         node_count: int,
                         cluster_name_on_cloud: str) -> None:
    """Boot the reservation's VMs, retrying through the server-side
    materialization delay.

    MEASURED IN THE LIVE E2E (2026-11-06): the reservation record
    materializes server-side with a DELAY after the fill — an activate
    fired +350ms after the fill is refused with "Ornn could not find
    that reservation". Bounded retry (the activate is idempotent by its
    per-cluster request id); any other refusal stays loud and typed.
    Shared by the fresh-bid path and the adoption resume.
    """
    activate_deadline = time.monotonic() + 120.0
    while True:
        try:
            client.activate_access(
                reservation_id,
                [key_id],
                machine_count=node_count,
                network_mode="public",
                request_id=f"skypilot-activate-{cluster_name_on_cloud}",
            )
            return
        except utils.OrnnError as exc:
            if ("could not find" not in str(exc) or
                    time.monotonic() >= activate_deadline):
                raise
            logger.info(
                "ornn: reservation %s not materialized yet; retrying "
                "activate", reservation_id)
            time.sleep(5.0)


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Place (or adopt) the cluster's spot bid and wait for SSH-ready."""
    del cluster_name  # cluster_name_on_cloud is the identity we use
    del region  # the bid API takes no region (gpuSlug + gpus only)
    client = _client()

    gpu_slug, gpus = _parse_instance_type(config.node_config or {})

    # Adoption: the identity store is the ONLY attribution (the bid API has
    # no name field). A recorded reservation that is still live is adopted;
    # one that ENDED fails closed — re-bidding while a stale record exists
    # would mint a second billed box that terminate could never find.
    recorded = utils.reservation_id_for(cluster_name_on_cloud)
    if recorded:
        reservation = client.show_reservation(recorded)
        status = str(reservation.get("status") or "").strip().lower()
        if _is_live(reservation):
            logger.info(
                "ornn: adopting recorded reservation %s (%s) for cluster %s",
                recorded,
                status,
                cluster_name_on_cloud,
            )
            created: List[str] = []
            reservation_id = recorded
            # v2 (CodeRabbit on #90): an adopted reservation may be a
            # PRE-activation relaunch — a first attempt that died between
            # the fill and the activate leaves a live reservation with NO
            # booted node, and waiting on it would just burn the deadline.
            # Adopt the boot too: activate only when no node is exposed
            # yet, idempotently by the same per-cluster request id, machine
            # count from the reservation's own reservedNodeCount.
            adopt_targets = client.list_storage_targets()
            if (utils.OrnnClient.target_for_reservation(
                    adopt_targets, recorded) is None):
                public_key = (config.node_config or {}).get("PublicKey")
                if isinstance(public_key, str):
                    public_key = public_key.strip()
                if (not public_key or
                        public_key == _SSH_PUBLIC_KEY_PLACEHOLDER):
                    raise utils.OrnnError(
                        "node_config['PublicKey'] is missing or was never "
                        f"substituted (got {public_key!r}) — the adopted "
                        "reservation exposes no booted node, and resuming "
                        "its activation needs the account key")
                adopt_key_id = _ensure_account_ssh_key(client,
                                                       str(public_key))
                adopt_node_count = int(
                    reservation.get("reservedNodeCount") or 1)
                _activate_with_retry(client, recorded, adopt_key_id,
                                     adopt_node_count,
                                     cluster_name_on_cloud)
        else:
            raise utils.OrnnError(
                f"cluster {cluster_name_on_cloud!r} is recorded against "
                f"reservation {recorded}, which has ENDED "
                f"(status={status!r}, endReason="
                f"{reservation.get('endReason')!r}); clear the stale record "
                "before relaunching (a re-bid now would bill a box "
                "terminate can no longer find)")
    else:
        # node_config['PublicKey'] is the ONE canonical source:
        # backend_utils puts Ornn on the generic `auth.configure_ssh_info`
        # path, which substitutes the real key into the template's
        # `skypilot:ssh_public_key_content` placeholder. The placeholder
        # check is the point: the unsubstituted literal is TRUTHY, a bare
        # `if not public_key` waves garbage into the account-key register,
        # and the bid fails far from the cause.
        public_key = (config.node_config or {}).get("PublicKey")
        if isinstance(public_key, str):
            public_key = public_key.strip()
        if not public_key or public_key == _SSH_PUBLIC_KEY_PLACEHOLDER:
            raise utils.OrnnError(
                "node_config['PublicKey'] is missing or was never "
                f"substituted (got {public_key!r}); auth.configure_ssh_info "
                "must run before provisioning — a spot bid without a real "
                "account key is an unreachable VM")
        # v2 MEASURED (2026-11-06): a filled bid does NOT auto-launch —
        # activation boots the machine and provisions exactly the
        # sshKeyIds passed THERE. The key must exist BEFORE the bid (the
        # activate call names it by id), so the id is captured here.
        key_id = _ensure_account_ssh_key(client, str(public_key))

        # v2: bids buy NODES — whole machines. The free quote names the
        # box shape (gpusPerNode) at the live on-demand ask; the node
        # count resolves from it so the bid always covers the requested
        # GPU count (a bigger box, never a smaller one — the v1
        # _sellable_gpu_count rule). on-demand fills instantly and runs
        # until market_end; v2's 'spot' tier yields SCHEDULED bookings,
        # useless for an immediate launch, so the lane buys on-demand
        # and the QUOTE is the price truth at launch (catalog rows are
        # advisory admission pricing).
        quote = client.quote_market(gpu_slug, sale_type="on-demand")
        quote_nodes = [
            n for n in (quote.get("nodes") or [])
            if isinstance(n, dict) and int(n.get("gpusPerNode") or 0) >= 1
        ]
        if not quote_nodes:
            raise utils.OrnnCapacityRefusedError(
                f"ornn on-demand quote for {gpu_slug} carries no node "
                f"capacity: {sorted(quote)} — capacity refusal at the "
                "current ask, not retried")
        per_node = int(quote_nodes[0]["gpusPerNode"])
        node_count = -(-int(gpus) // per_node)  # ceil: whole boxes only
        ask_per_gpu = int(
            quote_nodes[0].get("pricePerGpuHourMicroUsd") or 0)

        # MARKET order: fill-now-or-cancelled. A cancelled market bid is a
        # capacity refusal (typed, surfaced, never blind-retried); the
        # idempotency key is stable per cluster so a retried provision
        # re-places the SAME bid.
        bid = client.place_spot_bid(
            gpu_slug,
            nodes=node_count,
            sale_type="on-demand",
            idempotency_key=f"skypilot-{cluster_name_on_cloud}",
            confirm=True,
        )
        # Explicitly narrowed (CodeRabbit on #90 round 1): the response
        # shape is an object under "order", falling back to the bid body
        # itself for a legacy envelope — typed so the fillIds read below
        # is a Dict[str, Any] access for mypy, never a guessing game.
        raw_order = bid.get("order")
        order: Dict[str, Any] = (raw_order if isinstance(raw_order, dict)
                                 else bid)
        order_status = str(order.get("status") or "").strip().lower()
        if order_status == "cancelled":
            raise utils.OrnnCapacityRefusedError(
                f"ornn market bid for {gpu_slug} x{gpus} (cluster "
                f"{cluster_name_on_cloud!r}) was CANCELLED — capacity "
                "refusal at the current ask, not retried (report and decide)")
        fill_ids = [
            str(f).strip() for f in (order.get("fillIds") or [])
            if str(f or "").strip()
        ]
        # v2 MEASURED: the fill ID IS the reservation ID (no orderId on
        # a filled v2 bid).
        reservation_id = fill_ids[0] if fill_ids else ""
        if not reservation_id:
            raise utils.OrnnError(
                f"ornn market bid response carried no fill id: "
                f"{sorted(order)}")
        created = [reservation_id]
        # The identity store is written BEFORE the readiness wait
        # (CodeRabbit on skypilot-controller#529): a reservation that
        # wedges in wait_until_ready must remain attributable to
        # THIS cluster, or terminate/query/reconcile cannot find it and
        # the box keeps billing with no owner able to end it.
        utils.remember_cluster(cluster_name_on_cloud, reservation_id, gpu_slug,
                               gpus)
        logger.info(
            "ornn: market bid %s x%d (%d node[s] of %d GPU[s], ask "
            "$%.2f/GPU-hr) for cluster %s filled as reservation %s",
            gpu_slug,
            gpus,
            node_count,
            per_node,
            ask_per_gpu / 1_000_000.0,
            cluster_name_on_cloud,
            reservation_id,
        )

        # v2: no auto-launch — the ACTIVATE call boots the machine and
        # provisions the key captured above. Idempotent per cluster
        # (the helper retries through the fill->record materialization
        # delay measured in the live e2e).
        _activate_with_retry(client, reservation_id, key_id, node_count,
                             cluster_name_on_cloud)

    # SSH-ready: the node address must surface in storage_targets. Fails
    # CLOSED on cancel/complete-without-node and on the deadline (the
    # security-update first boot takes minutes and is tolerated by the
    # default timeout, then loud). The reservation is ALREADY recorded:
    # a failed wait leaves an attributable, cancellable box — never an
    # orphan billing invisibly to reconcile.
    client.wait_until_ready(reservation_id)

    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=utils.REGION,
        zone=None,
        head_instance_id=reservation_id,
        resumed_instance_ids=[],
        created_instance_ids=created,
    )


def wait_instances(
    region: str,
    cluster_name_on_cloud: str,
    state: Optional[status_lib.ClusterStatus],
) -> None:
    """No-op: run_instances already blocks until SSH-ready."""
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """STOP is refused, not silently destroy-terminated.

    The dispatch contract (sky/provision/__init__.py stop_instances wrapper
    after `provider_name` is stripped) is this signature — a
    `(region, cluster_name, ...)` shape copied from vast does not BIND and
    would TypeError at dispatch. Ornn declares STOP unsupported
    (Ornn._CLOUD_UNSUPPORTED_FEATURES): ``ornn_spot_cancel`` ENDS the
    reservation and destroys the VM + local disk, so SkyPilot never routes
    here; if anything ever does, refuse loudly rather than destroying the
    disk under a caller that believed it was preserved.
    """
    del cluster_name_on_cloud, provider_config, worker_only
    raise NotImplementedError(
        "stop is unsupported on Ornn: the only teardown is "
        "ornn_spot_cancel, which ENDS the reservation and destroys the VM "
        "and its local disk. Use terminate_instances for teardown.")


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """End the cluster's filled bid (v2 market_end; DELETE-only teardown).

    An empty identity is refused (it filters nothing); the reservation is
    resolved ONLY through the local identity store (the ownership gate —
    an unattributed reservation is invisible by design); not-found is
    idempotent success, and the v2 "conflicting operation" refusal (an
    end already in flight) is success inside end_fill.
    """
    del provider_config, worker_only  # single-node lane; the head is the cluster
    if not str(cluster_name_on_cloud or "").strip():
        # An empty identity would look up NOTHING — but refusing loudly is
        # the contract every sibling enforces (the Latitude/Vast class:
        # never a teardown path that could sweep more than asked).
        raise utils.OrnnError(
            "refusing to terminate with an empty cluster_name_on_cloud")
    client = _client()
    reservation_id = utils.reservation_id_for(cluster_name_on_cloud)
    if not reservation_id:
        logger.info(
            "ornn: no recorded reservation for cluster %r; nothing to "
            "terminate (unattributed reservations are invisible by design)",
            cluster_name_on_cloud,
        )
        return
    try:
        # v2 MEASURED (2026-11-06): market_end is the teardown (a fill ID
        # is its reservation ID) — async, and repeats while the end is
        # still processing refuse with "Wait for conflicting operation",
        # which end_fill returns as success-in-flight data.
        client.end_fill(reservation_id)
    except utils.OrnnNotFoundError:
        # Already ended server-side: teardown is idempotent.
        logger.info("ornn: reservation %s already gone on end",
                    reservation_id)
    utils.forget_cluster(cluster_name_on_cloud)


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region, provider_config
    client = _client()
    reservation_id = utils.reservation_id_for(cluster_name_on_cloud)
    if not reservation_id:
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)
    reservation = client.show_reservation(reservation_id)
    if not _is_live(reservation):
        return common.ClusterInfo(instances={},
                                  head_instance_id=None,
                                  provider_name=PROVIDER_NAME)
    # v2 MEASURED (2026-11-06): the node address lives in
    # storage_targets (ip + nodeUsername); access_show stays empty for
    # VM-mode reservations.
    targets = client.list_storage_targets()
    target = utils.OrnnClient.target_for_reservation(targets, reservation_id)
    host, ssh_user = (utils.OrnnClient.target_node_endpoint(target)
                      if target is not None else ("", utils.SSH_USER))
    instances: Dict[str, List[common.InstanceInfo]] = {}
    node_name = str(
        reservation.get("spotNodeId") or reservation.get("nodeId") or
        reservation_id)
    instances[reservation_id] = [
        common.InstanceInfo(
            instance_id=reservation_id,
            # A VM exposes one public SSH endpoint (the storage-target
            # ip); there is no separate internal address and no custom
            # port (v2 carries none — measured: stock 22).
            internal_ip=host,
            external_ip=host,
            ssh_port=22,
            tags={},
            node_name=node_name,
        )
    ]
    return common.ClusterInfo(
        instances=instances,
        head_instance_id=reservation_id,
        provider_name=PROVIDER_NAME,
        # The v2 storage target carries the node's username
        # (MEASURED 2026-11-06: 'ubuntu'); SSH_USER remains the fallback.
        ssh_user=ssh_user or utils.SSH_USER,
    )


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    del cluster_name, provider_config
    client = _client()
    result: Dict[str, Tuple[Optional[status_lib.ClusterStatus],
                            Optional[str]]] = {}
    # The ownership gate: ONLY the recorded reservation is reported. An
    # unattributed reservation (a bid this integration did not place, or a
    # record lost to a wiped store) is invisible — reconcile must never
    # sweep reservations it cannot attribute.
    reservation_id = utils.reservation_id_for(cluster_name_on_cloud)
    if not reservation_id:
        return result
    try:
        reservation = client.show_reservation(reservation_id)
    except utils.OrnnNotFoundError:
        return result
    sky_status = _sky_status(reservation)
    raw = str(reservation.get("status") or "").strip().lower()
    if sky_status is None and raw not in _STATUS_MAP:
        # Unmapped status (open set): log verbatim, treat as in-flight —
        # never crash a reconcile loop over a slug the platform invented.
        logger.info(
            "ornn reservation %s in unmapped status %r (treated as "
            "in-flight)",
            reservation_id,
            raw,
        )
    if non_terminated_only and sky_status is not status_lib.ClusterStatus.UP:
        return result
    result[reservation_id] = (sky_status, None)
    return result


def open_ports(
    region: str,
    cluster_name_on_cloud: str,
    ports: List[int],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op: Ornn spot VMs expose their SSH endpoint directly; there is no
    port firewall in the deploy path (OpenPortsVersion.LAUNCH_ONLY)."""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[int],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op counterpart to open_ports."""
    del cluster_name_on_cloud, ports, provider_config
