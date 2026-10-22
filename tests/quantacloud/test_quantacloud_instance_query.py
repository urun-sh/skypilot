"""Behavioral tests for the quantacloud provision adapter's
``query_instances`` retry contract (sky/provision/quantacloud/instance.py).

The 25th (live receipts: the -17 StopFailoverError and the -19 launch
TypeError, ENG-575, 2026-10-19): a JUST-CREATED deployment can 404 on the
provider's first status poll — the read-after-write window between the
create and the get server-side. The pre-fix adapter TypeErrored on the
``retry_if_missing`` kwarg every other cloud adapter in the tree accepts,
and even without the TypeError it would forget the mapping on that first
404 — treating the just-launched box as deleted and stranding the launch.

Transport-injected: no network. The client is a stub whose
``get_deployment`` raises ``QuantacloudNotFoundError`` a chosen number of
times before succeeding, and the mapping helpers are patched, so each test
pins exactly the retry/forget boundary.
"""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sky.provision.quantacloud import instance as qc_instance
from sky.provision.quantacloud import utils

# The lock file rides beside _STATE_PATH; point it at a tmp dir so the
# suite never touches the real ~/.sky.
_STATE_DIR = Path(tempfile.mkdtemp(prefix="qc-instance-test-"))


class _StubClient:
    """get_deployment raises NotFoundError `failures` times, then succeeds."""

    def __init__(self, failures):
        self._failures = failures
        self.calls = 0

    def get_deployment(self, deployment_id):
        self.calls += 1
        if self.calls <= self._failures:
            raise utils.QuantacloudNotFoundError(f"missing: {deployment_id}")
        return {"deploymentId": deployment_id, "state": "ACTIVE"}


def _patched_adapter(failures,
                     *,
                     retry_if_missing,
                     reload_state=None,
                     mapped_id="dep-123",
                     real_forget=False):
    """Run query_instances against the stub, returning (result, forgot).

    ``reload_state`` (optional) overrides what the SECOND ``_load_state``
    call returns — the expiration path's re-read: pass a fresh state whose
    mapping names a DIFFERENT deployment to pin that the stale snapshot
    never clobbers a concurrent writer's new mapping. ``mapped_id`` is the
    ``_mapped_id`` patch target (a constant, or a per-call sequence).
    With ``real_forget``, the REAL ``_forget`` runs (its lock + re-read +
    deployment guard) and ``_save_state`` records into ``forgot`` (a
    skipped guard never saves); otherwise ``_forget`` itself is a recorder.
    """
    state = {"mappings": {}}
    forgot = []
    load_states = [state, reload_state if reload_state is not None else state]
    if isinstance(mapped_id, (list, tuple)):
        mapped_id_patch = mock.patch.object(qc_instance,
                                            "_mapped_id",
                                            side_effect=list(mapped_id))
    else:
        mapped_id_patch = mock.patch.object(qc_instance,
                                            "_mapped_id",
                                            return_value=mapped_id)
    if real_forget:
        forget_patch = mock.patch.object(
            qc_instance,
            "_save_state",
            side_effect=lambda st: forgot.append(st))
    else:
        forget_patch = mock.patch.object(
            qc_instance,
            "_forget",
            side_effect=lambda name, dep=None: forgot.append(name))
    with mock.patch.object(qc_instance.time, "sleep", lambda s: None), \
         mock.patch.object(qc_instance, "_client",
                           return_value=_StubClient(failures)), \
         mock.patch.object(qc_instance, "_load_state",
                           side_effect=load_states), \
         mapped_id_patch, \
         forget_patch, \
         mock.patch.object(
             qc_instance, "_sky_status", return_value=mock.sentinel.status), \
         mock.patch.object(qc_instance, "_STATE_PATH",
                           str(_STATE_DIR / "clusters.json")):
        result = qc_instance.query_instances("cluster",
                                             "cluster-on-cloud",
                                             retry_if_missing=retry_if_missing)
    return result, forgot


class QueryInstancesRetryContract(unittest.TestCase):
    """The dispatch contract + the read-after-write window."""

    def test_accepts_the_retry_if_missing_kwarg(self):
        """Every other adapter accepts the kwarg; the pre-fix adapter
        TypeErrored here — the exact -19 launch-failure line."""
        # A missing deployment mapping short-circuits before any transport
        # call, so this exercises the signature alone.
        with mock.patch.object(qc_instance, "_client", return_value=_StubClient(0)), \
             mock.patch.object(qc_instance, "_load_state",
                               return_value={"mappings": {}}), \
             mock.patch.object(qc_instance, "_mapped_id", return_value=None):
            self.assertEqual(
                qc_instance.query_instances("c",
                                            "c-on-cloud",
                                            retry_if_missing=True), {})

    def test_first_poll_404_with_retry_repolls_and_keeps_the_mapping(self):
        """The -10-class window: a just-created deployment 404s once, the
        re-poll succeeds, the mapping is NEVER forgotten, and the instance
        reports its status."""
        result, forgot = _patched_adapter(1, retry_if_missing=True)
        self.assertEqual(forgot, [])
        self.assertEqual(result, {"dep-123": (mock.sentinel.status, None)})

    def test_steady_state_404_still_forgets_the_mapping(self):
        """Without the flag (every steady-state poll), a 404 is the REAL
        gone signal: forget the mapping, report absence — unchanged
        pre-fix behavior."""
        result, forgot = _patched_adapter(1, retry_if_missing=False)
        self.assertEqual(result, {})
        self.assertEqual(forgot, ["cluster-on-cloud"])

    def test_retry_window_is_bounded(self):
        """A deployment still 404ing past the window is treated as gone:
        the re-poll never spins forever (the window is 60s at 5s polls —
        shrunken here so the test runs in milliseconds)."""
        with mock.patch.object(qc_instance, "_QUERY_RETRY_WINDOW_S", 0.0):
            result, forgot = _patched_adapter(1, retry_if_missing=True)
        self.assertEqual(result, {})
        self.assertEqual(forgot, ["cluster-on-cloud"])

    def test_expiration_never_clobbers_a_concurrent_writers_new_mapping(self):
        """CodeRabbit (this PR): the retry window is long enough for a
        concurrent writer to have saved a NEW mapping for this cluster
        name. The expiration path must re-read the CURRENT state and
        remove OUR OWN mapping only if it still names THIS deployment —
        never delete-and-save the stale snapshot wholesale."""
        with mock.patch.object(qc_instance, "_QUERY_RETRY_WINDOW_S", 0.0):
            # The re-read state: the cluster name now maps to a NEW
            # deployment (a concurrent writer's replacement box).
            fresh = {"mappings": {}}
            result, forgot = _patched_adapter(1,
                                              retry_if_missing=True,
                                              reload_state=fresh,
                                              mapped_id=["dep-123", "dep-999"],
                                              real_forget=True)
        self.assertEqual(result, {})
        self.assertEqual(forgot, [])


if __name__ == "__main__":
    unittest.main()
