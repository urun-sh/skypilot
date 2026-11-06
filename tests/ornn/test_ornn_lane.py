"""Lane tests for the Ornn carry: provisioner lifecycle, fetcher row
selection, and wiring pins for every threading point (the former
lane-killer class), mirroring tests/latitude/test_latitude_lane.py.

The provisioner is exercised through a stubbed client
(sky.provision.ornn.utils.client_from_env is patched) — no network. The
fetcher is exercised from fixture books. The wiring pins ARE
source-inspected (AST) because a missing thread does not fail a behavior
test until a real launch dies — the Shadeform lesson.
"""

import ast
import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

from sky import clouds
from sky.catalog.data_fetchers import fetch_ornn
from sky.provision import common
from sky.provision.ornn import instance as provision_instance
from sky.provision.ornn import utils as provision_utils
from sky.utils import status_lib


def _config(instance_type="ornn:nvidia_rtx_pro_6000:1",
            public_key="ssh-ed25519 AAAAREAL real-deploy-key"):
    node_config = {"InstanceType": instance_type}
    if public_key is not None:
        node_config["PublicKey"] = public_key
    return common.ProvisionConfig(
        provider_config={},
        authentication_config={},
        docker_config={},
        node_config=node_config,
        count=1,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


class _StubClient:
    """Scripted OrnnClient stand-in: records calls, returns canned payloads."""

    def __init__(self):
        self.calls = []
        # Distinct MATERIAL from the deploy keys (material is the identity).
        self.ssh_keys = [{
            "id": "key-1",
            "public_key": "ssh-ed25519 AAAASTUB stub-key",
            "status": "active"
        }]
        self.reservations = {}
        # v2 MEASURED (2026-11-06): a filled market bid's order carries
        # fillIds (the fill ID IS the reservation ID).
        self.bid_response = {
            "order": {
                "orderId": "ord-1",
                "status": "filled",
                "fillIds": ["resv-1"],
                "filledGpus": 1,
                "filledNodes": 1,
                "saleType": "on-demand",
            },
        }
        self.quote_response = {
            "gpuSlug": "nvidia_rtx_pro_6000",
            "nodes": [{
                "gpus": 1,
                "gpusPerNode": 1,
                "pricePerGpuHourMicroUsd": 1500000,
                "listingId": "9704db15-0000-0000-0000-000000000001",
                "region": "us-west-2",
            }],
            "region": None,
            "saleType": "on-demand",
        }
        # v2 MEASURED: the node address lives in storage_targets.
        self.targets = [{
            "id": "resv-1",
            "label": "1 node",
            "nodeCount": 1,
            "gpuCount": 1,
            "gpuType": "nvidia_rtx_pro_6000",
            "nodeUsername": "ubuntu",
            "status": "active",
            "nodes": [{
                "ip": "gpu-1.ornn.com",
                "name": "Node 1",
                "reservationId": "resv-1",
            }],
            "optionKey": "node:resv-1",
            "reservationIds": ["resv-1"],
        }]
        self.cancel_not_found = False

    # -- record + canned behavior ------------------------------------------

    def whoami(self):
        return {"user": {"email": "mo@urun.sh"}}

    def list_ssh_keys(self):
        return list(self.ssh_keys)

    def add_ssh_key(self, public_key, *, label):
        self.calls.append(("add_ssh_key", public_key, label))
        return {"id": "key-2"}

    def list_spot_books(self):
        return [{
            "gpuSlug": "nvidia_rtx_pro_6000",
            "bestAskPricePerGpuHourMicroUsd": 2000000
        }]

    def quote_market(self, gpu_slug, *, nodes=1, sale_type="on-demand",
                     region=None):
        self.calls.append(("quote_market", gpu_slug, nodes, sale_type))
        return json.loads(json.dumps(self.quote_response))

    def place_spot_bid(self,
                       gpu_slug,
                       *,
                       nodes=1,
                       sale_type="on-demand",
                       region=None,
                       idempotency_key=None,
                       confirm=False):
        self.calls.append(("place_spot_bid", gpu_slug, nodes, sale_type,
                           idempotency_key, confirm))
        return json.loads(json.dumps(self.bid_response))

    def show_reservation(self, reservation_id):
        return self.reservations.get(reservation_id, {
            "reservationId": reservation_id,
            "status": "active"
        })

    def end_fill(self, fill_id):
        self.calls.append(("end_fill", fill_id))
        if self.cancel_not_found:
            raise provision_utils.OrnnNotFoundError("not_found")
        return {"status": "ended", "fillId": fill_id}

    def wait_until_ready(self, reservation_id, **kwargs):
        self.calls.append(("wait_until_ready", reservation_id))
        if getattr(self, "fail_wait", False):
            raise provision_utils.OrnnError(
                "reservation resv-1 still 'active' with no node after "
                "1500s; refusing to wait longer")
        return self.show_reservation(reservation_id)

    def activate_access(self, reservation_id, ssh_key_ids, *,
                        access_mode="vm", machine_count=None,
                        network_mode=None, request_id=None):
        self.calls.append(("activate_access", reservation_id,
                           tuple(ssh_key_ids), machine_count, request_id))
        if getattr(self, "activate_not_found_once", False):
            # MEASURED in the live e2e 2026-11-06: the reservation record
            # materializes server-side with a delay after the fill.
            self.activate_not_found_once = False
            raise provision_utils.OrnnError(
                "tools/call access_activate: Ornn could not find that "
                "reservation")
        return {"nodes": []}

    def list_storage_targets(self):
        return json.loads(json.dumps(self.targets))


def _ready_stub():
    stub = _StubClient()
    stub.targets = [{
        "id": "resv-1",
        "reservationIds": ["resv-1"],
        "nodeUsername": "ubuntu",
        "nodeCount": 1,
        "gpuCount": 1,
        "status": "active",
        "nodes": [{"ip": "gpu-1.ornn.com", "name": "Node 1",
                   "reservationId": "resv-1"}],
    }]
    return stub


class TestProvisionerLifecycle(unittest.TestCase):

    def setUp(self):
        tmpdir = tempfile.mkdtemp(prefix="test-ornn-")
        self.addCleanup(lambda: os.path.exists(tmpdir) and _rmtree(tmpdir))
        self._store = os.path.join(tmpdir, "clusters.json")
        patcher = mock.patch.object(provision_utils, "CLUSTERS_FILE",
                                    self._store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_bid_launch_path_and_auto_attach(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            record = provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                      _config())
        self.assertEqual(record.head_instance_id, "resv-1")
        self.assertEqual(record.created_instance_ids, ["resv-1"])
        bid = next(c for c in stub.calls if c[0] == "place_spot_bid")
        # v2 MEASURED (2026-11-06): the launch path QUOTES first (free),
        # then places the on-demand whole-node bid with the stable
        # idempotency key — the quote's gpusPerNode resolves the count.
        self.assertEqual(bid[1], "nvidia_rtx_pro_6000")
        self.assertEqual(bid[2], 1)  # 1 GPU / gpusPerNode=1 -> one node
        self.assertEqual(bid[3], "on-demand")
        self.assertEqual(bid[4], "skypilot-c-on-cloud")
        self.assertIs(bid[5], True)
        quote = next(c for c in stub.calls if c[0] == "quote_market")
        self.assertLess(stub.calls.index(quote), stub.calls.index(bid))
        # v2: no auto-launch — the ACTIVATE call is IN the launch path now,
        # booting the box with the account key captured before the bid.
        activate = [c for c in stub.calls if c[0] == "activate_access"]
        self.assertEqual(len(activate), 1)
        self.assertEqual(activate[0][1], "resv-1")
        # _config's deploy key is NOT on the stub account, so the
        # provisioner registers it (key-2) and activates with THAT id.
        self.assertEqual(activate[0][2], ("key-2",))
        # The reservation is attributed in the local identity store.
        self.assertEqual(provision_utils.reservation_id_for("c-on-cloud"),
                         "resv-1")

    def test_activate_retries_through_materialization_delay(self):
        """MEASURED in the live e2e (2026-11-06): an activate fired
        +350ms after the fill is refused with 'Ornn could not find that
        reservation' — the record materializes server-side with a delay.
        The launch retries (the activate is idempotent by request id),
        bounded; any other refusal stays loud."""
        stub = _ready_stub()
        stub.activate_not_found_once = True
        sleeps = []
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub), \
             mock.patch.object(provision_instance.time, "sleep",
                               side_effect=lambda s: sleeps.append(s)):
            record = provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                      _config())
        self.assertEqual(record.head_instance_id, "resv-1")
        self.assertEqual(len(sleeps), 1)
        activates = [c for c in stub.calls if c[0] == "activate_access"]
        self.assertEqual(len(activates), 2)

    def test_reservation_recorded_before_the_readiness_wait(self):
        """CodeRabbit on skypilot-controller#529: a reservation that WEDGES
        in wait_until_ready (the reproducible auto-launch gap, ENG-541
        Q4) must remain attributable to this cluster, or
        terminate/query/reconcile cannot find it and the box keeps
        billing with no owner able to cancel it. The store is written
        BEFORE the wait — a failed wait leaves a cancellable record."""
        stub = _ready_stub()
        stub.fail_wait = True
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            with self.assertRaises(provision_utils.OrnnError):
                provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                 _config())
        # The reservation IS in the store despite the failed wait, so
        # terminate_instances and query_instances both find it.
        self.assertEqual(provision_utils.reservation_id_for("c-on-cloud"),
                         "resv-1")
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            seen = provision_instance.query_instances("c", "c-on-cloud")
            # And the owner can tear the wedged box down — INSIDE the
            # mock context: a stub-less _client() would build a REAL
            # client from ~/.ornn and call the LIVE market (a stray
            # live-cancel attempt with the stub's non-UUID id was
            # rejected server-side during test development — never
            # again).
            provision_instance.terminate_instances("c-on-cloud")
        self.assertEqual(set(seen), {"resv-1"})
        self.assertEqual([c for c in stub.calls if c[0] == "end_fill"],
                         [("end_fill", "resv-1")])

    def test_terminated_wait_failure_does_not_double_bid(self):
        """A wedged launch leaves the store entry; a RETRY must ADOPT the
        recorded reservation, never place a second bid (two boxes would
        bill with one identity)."""
        stub = _ready_stub()
        stub.fail_wait = True
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            with self.assertRaises(provision_utils.OrnnError):
                provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                 _config())
            stub.fail_wait = False
            record = provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                      _config())
        bids = [c for c in stub.calls if c[0] == "place_spot_bid"]
        self.assertEqual(len(bids), 1)
        self.assertEqual(record.created_instance_ids, [])

    def test_verify_zero_stock_refuses_offered_schedule_without_a_book(self):
        """CodeRabbit on skypilot-controller#529: an offered schedule whose
        slug is MISSING from the books payload is a contradiction too — the
        book-membership filter made an empty books payload accept the
        contradiction and write a false header-only catalog."""
        with self.assertRaises(fetch_ornn.OrnnCatalogError) as ctx:
            fetch_ornn.verify_zero_stock([], [], [{
                "gpuSlug": "nvidia_rtx_pro_6000",
                "gpusPerNode": 1,
                "capacityNodeCount": 8
            }])
        self.assertIn("nvidia_rtx_pro_6000", str(ctx.exception))

    def test_missing_public_key_refused(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            with self.assertRaises(provision_utils.OrnnError):
                provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                 _config(public_key=None))

    def test_placeholder_public_key_refused(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            with self.assertRaises(provision_utils.OrnnError):
                provision_instance.run_instances(
                    "ornn", "c", "c-on-cloud",
                    _config(public_key=provision_instance.
                            _SSH_PUBLIC_KEY_PLACEHOLDER))

    def test_ensure_account_key_by_material_before_bid(self):
        # A deployment key NOT on the account is registered (material is the
        # identity) BEFORE the bid — spot VMs auto-attach account keys.
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances(
                "ornn", "c", "c-on-cloud",
                _config(public_key="ssh-ed25519 AAAAFRESH fresh-key"))
        added = [c for c in stub.calls if c[0] == "add_ssh_key"]
        self.assertEqual(len(added), 1)
        self.assertIn("fresh-key", added[0][1])
        bid_index = next(
            i for i, c in enumerate(stub.calls) if c[0] == "place_spot_bid")
        add_index = next(
            i for i, c in enumerate(stub.calls) if c[0] == "add_ssh_key")
        self.assertLess(add_index, bid_index)

    def test_existing_material_not_re_registered(self):
        stub = _ready_stub()
        stub.ssh_keys = [{
            "id": "key-1",
            "public_key": "ssh-ed25519 AAAAREAL real-deploy-key",
            "status": "active"
        }]
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
        self.assertEqual([c for c in stub.calls if c[0] == "add_ssh_key"], [])

    def test_cancelled_market_bid_surfaces_capacity_refusal(self):
        stub = _ready_stub()
        stub.bid_response = {
            "order": {
                "orderId": "resv-x",
                "status": "cancelled"
            }
        }
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            with self.assertRaises(provision_utils.OrnnCapacityRefusedError):
                provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                 _config())

    def test_adoption_resumes_an_interrupted_activation(self):
        """CodeRabbit on #90: a first attempt that died between the fill
        and the activate leaves a live reservation with NO booted node —
        the adoption must resume the boot (same per-cluster request id,
        node count from the reservation's own reservedNodeCount), never
        wait out the deadline on an unactivated box."""
        stub = _ready_stub()
        stub.targets = []  # live reservation, no booted node
        stub.reservations["resv-1"] = {"reservationId": "resv-1",
                                       "status": "active",
                                       "reservedNodeCount": 1}
        # The store already attributes resv-1 to this cluster (the first
        # attempt recorded it before dying between fill and activate).
        provision_utils.remember_cluster("c-on-cloud", "resv-1",
                                        "nvidia_rtx_pro_6000", 1)
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            record = provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                      _config())
        self.assertEqual(record.head_instance_id, "resv-1")
        self.assertEqual(record.created_instance_ids, [])
        activates = [c for c in stub.calls if c[0] == "activate_access"]
        self.assertEqual(len(activates), 1)
        # machine_count comes from the reservation's reservedNodeCount.
        self.assertEqual(activates[0][3], 1)
        # And the SAME per-cluster request id rides the resume.
        self.assertEqual(activates[0][4],
                         "skypilot-activate-c-on-cloud")

    def test_adoption_of_recorded_live_reservation(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
            record = provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                      _config())
        # The second run ADOPTED resv-1: no second bid was placed.
        bids = [c for c in stub.calls if c[0] == "place_spot_bid"]
        self.assertEqual(len(bids), 1)
        self.assertEqual(record.head_instance_id, "resv-1")
        self.assertEqual(record.created_instance_ids, [])

    def test_ended_recorded_reservation_refuses_rebid(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
            stub.reservations["resv-1"] = {
                "reservationId": "resv-1",
                "status": "completed",
                "endReason": "stopped_by_buyer"
            }
            with self.assertRaises(provision_utils.OrnnError) as ctx:
                provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                                 _config())
        self.assertIn("ENDED", str(ctx.exception))

    def test_terminate_cancels_and_forgets(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
            provision_instance.terminate_instances("c-on-cloud")
        ends = [c for c in stub.calls if c[0] == "end_fill"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0][1], "resv-1")
        self.assertIsNone(provision_utils.reservation_id_for("c-on-cloud"))

    def test_terminate_empty_identity_refused(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            with self.assertRaises(provision_utils.OrnnError):
                provision_instance.terminate_instances("   ")

    def test_terminate_not_found_is_idempotent(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
            stub.cancel_not_found = True
            provision_instance.terminate_instances("c-on-cloud")
        self.assertIsNone(provision_utils.reservation_id_for("c-on-cloud"))

    def test_query_visibility_gate(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
            seen = provision_instance.query_instances("c", "c-on-cloud")
            # Only the recorded reservation is visible; unattributed
            # reservations are invisible by design.
            self.assertEqual(set(seen), {"resv-1"})
            self.assertEqual(seen["resv-1"][0], status_lib.ClusterStatus.UP)
            other = provision_instance.query_instances("c",
                                                       "some-other-cluster")
            self.assertEqual(other, {})

    def test_get_cluster_info_shape(self):
        stub = _ready_stub()
        with mock.patch.object(provision_utils,
                               "client_from_env",
                               return_value=stub):
            provision_instance.run_instances("ornn", "c", "c-on-cloud",
                                             _config())
            info = provision_instance.get_cluster_info("ornn", "c-on-cloud")
        self.assertEqual(info.provider_name, "ornn")
        # v2 MEASURED (2026-11-06): the storage target carries the node's
        # username ('ubuntu') and ip; no custom SSH port in v2 (stock 22).
        self.assertEqual(info.ssh_user, "ubuntu")
        self.assertEqual(info.instances["resv-1"][0].ssh_port, 22)
        self.assertEqual(info.instances["resv-1"][0].external_ip,
                         "gpu-1.ornn.com")

    def test_stop_refused_and_signatures_pinned(self):
        with self.assertRaises(NotImplementedError):
            provision_instance.stop_instances("c-on-cloud")
        # The dispatch contract (sky/provision/__init__.py strips
        # `provider_name`): names AND kinds AND order — a
        # `(region, cluster_name, ...)` shape copied from vast does not
        # BIND and TypeErrors at dispatch.
        for func in (provision_instance.stop_instances,
                     provision_instance.terminate_instances):
            params = inspect.signature(func).parameters
            self.assertEqual(
                list(params),
                ["cluster_name_on_cloud", "provider_config", "worker_only"])
            self.assertTrue(
                all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                    for p in params.values()))


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


class TestFetcherRows(unittest.TestCase):
    BOOKS = [
        {
            "gpuSlug": "nvidia_rtx_pro_6000",
            "bestAskPricePerGpuHourMicroUsd": 2000000
        },
        {
            "gpuSlug": "nvidia_h100_sxm",
            "bestAskPricePerGpuHourMicroUsd": None
        },
        {
            "gpuSlug": "nvidia_h200_sxm",
            "bestAskPricePerGpuHourMicroUsd": 0
        },
    ]

    def test_rows_only_for_live_asks(self):
        rows = list(fetch_ornn.iter_rows(self.BOOKS))
        by_type = {r[0]: r for r in rows}
        # The ask book yields counts 1..8 priced ask x count.
        self.assertEqual(len(rows), 8)
        self.assertIn("ornn:nvidia_rtx_pro_6000:1", by_type)
        self.assertIn("ornn:nvidia_rtx_pro_6000:8", by_type)
        price_i = fetch_ornn.CSV_COLUMNS.index("Price")
        self.assertEqual(by_type["ornn:nvidia_rtx_pro_6000:1"][price_i], 2.0)
        self.assertEqual(by_type["ornn:nvidia_rtx_pro_6000:8"][price_i], 16.0)
        # Null-ask and zero-ask books are never quoted free.
        self.assertNotIn("ornn:nvidia_h100_sxm:1", by_type)
        self.assertNotIn("ornn:nvidia_h200_sxm:1", by_type)

    def test_write_csv_creates_missing_parent_dir(self):
        # get_catalog_path is side-effect free; the writer must create
        # the versioned catalog's ornn/ parent itself.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ornn", "vms.csv")
            written = fetch_ornn.write_csv(
                [next(iter(fetch_ornn.iter_rows(self.BOOKS)))], out)
            self.assertEqual(written, 1)
            self.assertTrue(os.path.exists(out))

    def test_row_shape(self):
        row = next(iter(fetch_ornn.iter_rows(self.BOOKS)))
        record = dict(zip(fetch_ornn.CSV_COLUMNS, row))
        self.assertEqual(record["InstanceType"], "ornn:nvidia_rtx_pro_6000:1")
        self.assertEqual(record["AcceleratorName"], "RTXPRO6000")
        self.assertEqual(record["AcceleratorCount"], 1)
        self.assertEqual(record["Region"], "ornn")
        # GpuInfo parses as JSON after the quote-swap (the CSV convention).
        gpu_info = json.loads(record["GpuInfo"].replace("'", '"'))
        # The legacy 'Gpus' wrapper is mandatory (the catalog/common
        # DeviceMemoryGiB pass reads row['Gpus'][0]['MemoryInfo']
        # ['SizeInMiB'] — the #39/#40 lesson).
        self.assertEqual(gpu_info["Gpus"][0]["Name"], "RTXPRO6000")
        self.assertGreater(gpu_info["Gpus"][0]["MemoryInfo"]["SizeInMiB"], 0)
        self.assertEqual(record["SpotPrice"], "")

    def test_token_roundtrip_and_garbage_refused(self):
        self.assertEqual(
            fetch_ornn.parse_instance_type_token("ornn:nvidia_rtx_pro_6000:8"),
            ("nvidia_rtx_pro_6000", 8))
        for garbage in ("", "ornn:", "nvidia_rtx_pro_6000:1", "ornn:slug:x",
                        "ornn:slug:0", "ornn:slug:9"):
            with self.assertRaises(fetch_ornn.OrnnCatalogError):
                fetch_ornn.parse_instance_type_token(garbage)

    def test_verify_zero_stock_refuses_contradiction(self):
        schedules = [{
            "gpuSlug": "nvidia_rtx_pro_6000",
            "gpusPerNode": 1,
            "capacityNodeCount": 8
        }]
        with self.assertRaises(fetch_ornn.OrnnCatalogError) as ctx:
            fetch_ornn.verify_zero_stock([], self.BOOKS, schedules)
        self.assertIn("nvidia_rtx_pro_6000", str(ctx.exception))

    def test_verify_zero_stock_allows_agreement(self):
        schedules = [{
            "gpuSlug": "nvidia_h100_sxm",
            "gpusPerNode": 4,
            "capacityNodeCount": 0
        }]
        fetch_ornn.verify_zero_stock([], self.BOOKS, schedules)  # no raise

    def test_verify_zero_stock_refuses_listings_without_capacity_fields(self):
        """A listing missing its v2 capacity fields cannot be
        cross-checked — reading it as zero-capacity would let a false
        zero-stock catalog pass the check (CodeRabbit on #88)."""
        with self.assertRaises(fetch_ornn.OrnnCatalogError):
            fetch_ornn.verify_zero_stock([], self.BOOKS,
                                         [{"gpuSlug": "nvidia_h100_sxm"}])
        with self.assertRaises(fetch_ornn.OrnnCatalogError):
            fetch_ornn.verify_zero_stock(
                [], self.BOOKS,
                [{"gpuSlug": "nvidia_h100_sxm", "gpusPerNode": "lots",
                  "capacityNodeCount": 2}])
        # Round 2: int(0.5) truncates to 0 — a fractional count must be
        # refused, not read as zero capacity (and bools are never counts).
        with self.assertRaises(fetch_ornn.OrnnCatalogError):
            fetch_ornn.verify_zero_stock(
                [], self.BOOKS,
                [{"gpuSlug": "nvidia_h100_sxm", "gpusPerNode": 0.5,
                  "capacityNodeCount": 8}])
        with self.assertRaises(fetch_ornn.OrnnCatalogError):
            fetch_ornn.verify_zero_stock(
                [], self.BOOKS,
                [{"gpuSlug": "nvidia_h100_sxm", "gpusPerNode": True,
                  "capacityNodeCount": 8}])
        # Round 3: negative capacity is a data fault, not zero; positive
        # capacity with no gpuSlug cannot be attributed — both refuse.
        with self.assertRaises(fetch_ornn.OrnnCatalogError):
            fetch_ornn.verify_zero_stock(
                [], self.BOOKS,
                [{"gpuSlug": "nvidia_h100_sxm", "gpusPerNode": -1,
                  "capacityNodeCount": 8}])
        with self.assertRaises(fetch_ornn.OrnnCatalogError):
            fetch_ornn.verify_zero_stock(
                [], self.BOOKS,
                [{"gpusPerNode": 4, "capacityNodeCount": 2}])

    def test_unmapped_slug_compacts_and_stays_visible(self):
        rows = list(
            fetch_ornn.iter_rows([{
                "gpuSlug": "nvidia_gb300",
                "bestAskPricePerGpuHourMicroUsd": 5000000
            }]))
        self.assertEqual(rows[0][1], "GB300")
        self.assertEqual(rows[0][0], "ornn:nvidia_gb300:1")


class TestCloudWiring(unittest.TestCase):
    """Source-pinned threading tests (the Shadeform lane-killer class)."""

    def test_repr_and_catalog_resolution(self):
        self.assertEqual(clouds.Ornn._REPR, "Ornn")
        import importlib
        importlib.import_module(
            f"sky.catalog.{clouds.Ornn._REPR.lower()}_catalog")

    def test_all_clouds_contains_ornn(self):
        from sky.skylet import constants
        self.assertIn("ornn", constants.ALL_CLOUDS)

    def test_clouds_all_and_alias(self):
        from sky import clouds as clouds_mod
        self.assertIn("Ornn", clouds_mod.__all__)
        import sky
        self.assertIs(sky.Ornn, clouds.Ornn)

    def test_cluster_config_template_registered(self):
        from sky.backends import cloud_vm_ray_backend
        self.assertEqual(
            cloud_vm_ray_backend._get_cluster_config_template(clouds.Ornn()),
            "ornn-ray.yml.j2")

    def test_auth_dispatch_covers_ornn(self):
        # _add_auth_to_cluster_config is an isinstance chain ending in
        # `assert False, cloud`; Ornn must take the generic
        # configure_ssh_info branch (its key is an ACCOUNT key ensured by
        # material before the bid).
        from sky.backends import backend_utils
        source = inspect.getsource(backend_utils._add_auth_to_cluster_config)
        tree = ast.parse(source)
        names = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn("Ornn", names,
                      "Ornn missing from the auth dispatch chain")

    def test_zones_provision_loop_yields_per_region(self):
        from sky import clouds as clouds_lib
        with mock.patch.object(
                clouds.Ornn,
                "regions_with_offering",
                classmethod(lambda cls, *a, **k: [
                    clouds_lib.Region("ornn"),
                ]),
        ):
            yields = list(
                clouds.Ornn.zones_provision_loop(
                    region="ornn",
                    num_nodes=1,
                    instance_type="ornn:nvidia_rtx_pro_6000:1"))
        # One yield PER region with an offering — a single yield for "any"
        # turned a first-region failure into the end of provisioning
        # (the Spheron lane).
        self.assertEqual(yields, [None])

    def test_ssh_key_file_mount_is_ornn_scoped(self):
        mounts = clouds.Ornn().get_credential_file_mounts()
        self.assertEqual(mounts,
                         {"~/.ornn/mcp-credentials": "~/.ornn/mcp-credentials"})

    def test_dependencies_extra(self):
        from sky.setup_files import dependencies
        self.assertIn("ornn", dependencies.cloud_dependencies)
        self.assertEqual(dependencies.cloud_dependencies["ornn"], [])

    def test_provision_package_exports(self):
        import sky.provision.ornn as pkg
        for name in ("bootstrap_instances", "run_instances", "wait_instances",
                     "stop_instances", "terminate_instances",
                     "get_cluster_info", "query_instances", "open_ports",
                     "cleanup_ports"):
            self.assertTrue(callable(getattr(pkg, name)), name)


if __name__ == "__main__":
    unittest.main()
