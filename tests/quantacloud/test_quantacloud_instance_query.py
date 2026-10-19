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

import unittest
from unittest import mock

from sky.provision.quantacloud import instance as qc_instance
from sky.provision.quantacloud import utils


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


def _patched_adapter(failures, *, retry_if_missing):
    """Run query_instances against the stub, returning (result, forgot)."""
    state = {"mappings": {}}
    forgot = []
    with mock.patch.object(qc_instance, "_client", return_value=_StubClient(failures)), \
         mock.patch.object(qc_instance, "_load_state", return_value=state), \
         mock.patch.object(
             qc_instance, "_mapped_id", return_value="dep-123"), \
         mock.patch.object(
             qc_instance, "_forget",
             side_effect=lambda st, name: forgot.append(name)), \
         mock.patch.object(
             qc_instance, "_sky_status", return_value=mock.sentinel.status):
        result = qc_instance.query_instances(
            "cluster", "cluster-on-cloud", retry_if_missing=retry_if_missing)
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
                qc_instance.query_instances(
                    "c", "c-on-cloud", retry_if_missing=True), {})

    def test_first_poll_404_with_retry_repolls_and_keeps_the_mapping(self):
        """The -10-class window: a just-created deployment 404s once, the
        re-poll succeeds, the mapping is NEVER forgotten, and the instance
        reports its status."""
        result, forgot = _patched_adapter(1, retry_if_missing=True)
        self.assertEqual(forgot, [])
        self.assertEqual(
            result, {"dep-123": (mock.sentinel.status, None)})

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


if __name__ == "__main__":
    unittest.main()
