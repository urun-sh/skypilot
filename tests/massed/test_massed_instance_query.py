"""Behavioral tests for the MassedCompute provision adapter's
``query_instances`` retry contract (sky/provision/massed/instance.py).

The QuantaCloud ENG-575 lesson, carried into this lane from birth: a
JUST-LAUNCHED instance can be absent from the account list on the
provider's first status poll — the read-after-write window. Without
``retry_if_missing``, a launch-path status read would treat the
just-launched box as deleted and strand the launch; steady-state polls
must never enter the window.

Transport-injected: no network. ``_find_by_name`` is patched to miss a
chosen number of times before the instance appears, so each test pins
exactly the retry/absence boundary.
"""

import unittest
from unittest import mock

from sky.provision.massed import instance as provision


class _AppearAfter:
    """_find_by_name stub: misses `misses` polls, then returns the row."""

    def __init__(self, misses, row):
        self._misses = misses
        self._row = row
        self.calls = 0

    def __call__(self, client, name):
        self.calls += 1
        if self.calls <= self._misses:
            return None
        return self._row


_ROW = {
    'uuid': 'uuid-1',
    'name': 'urun-massed-e2e',
    'ip': '1.2.3.4',
    'status': 'rented',
    'os_booted': 1,
}


class QueryInstancesRetryContract(unittest.TestCase):
    """The read-after-write window + the steady-state absence."""

    def _query(self, misses, retry_if_missing):
        finder = _AppearAfter(misses, dict(_ROW))
        client = mock.Mock()
        with mock.patch.object(provision, '_client',
                               return_value=client), \
             mock.patch.object(provision, '_find_by_name', finder), \
             mock.patch.object(provision, '_QUERY_RETRY_INTERVAL_S', 0.0):
            result = provision.query_instances(
                cluster_name='c',
                cluster_name_on_cloud='urun-massed-e2e',
                retry_if_missing=retry_if_missing,
            )
        return result, finder.calls

    def test_absent_without_retry_reports_absence_immediately(self):
        # Steady state: no retry requested, absence is a fact after ONE
        # poll — the window is never entered.
        result, calls = self._query(misses=5, retry_if_missing=False)
        self.assertEqual(result, {})
        self.assertEqual(calls, 1)

    def test_just_launched_instance_found_within_the_window(self):
        # Launch path: the box appears on the SECOND poll; the retry
        # window must bridge the read-after-write gap.
        result, calls = self._query(misses=1, retry_if_missing=True)
        self.assertEqual(list(result), ['uuid-1'])
        self.assertEqual(calls, 2)

    def test_present_instance_maps_to_running_status(self):
        result, _ = self._query(misses=0, retry_if_missing=False)
        status = list(result.values())[0][0]
        self.assertIsNotNone(status)

    def test_row_without_uuid_reports_absence(self):
        # A row without a uuid cannot be addressed; report absence rather
        # than a keyless mapping.
        finder = _AppearAfter(0, {'name': 'urun-massed-e2e'})
        client = mock.Mock()
        with mock.patch.object(provision, '_client',
                               return_value=client), \
             mock.patch.object(provision, '_find_by_name', finder):
            result = provision.query_instances(
                cluster_name='c',
                cluster_name_on_cloud='urun-massed-e2e',
            )
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()
