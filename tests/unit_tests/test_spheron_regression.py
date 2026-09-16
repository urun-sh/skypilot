"""Regression tests for the four CodeRabbit review findings on Spheron."""
import inspect
import unittest
from unittest import mock


class TestRegionsWithOfferingAcceptsResources(unittest.TestCase):
    def test_accepts_resources_kwarg(self):
        from sky.clouds import spheron
        sig = inspect.signature(spheron.Spheron.regions_with_offering)
        self.assertIn('resources', sig.parameters)


class TestZonesProvisionLoopHonorsNoOffering(unittest.TestCase):
    def test_no_offering_yields_nothing(self):
        from sky.clouds import spheron
        with mock.patch.object(spheron.Spheron, 'regions_with_offering',
                                return_value=[]):
            result = list(spheron.Spheron.zones_provision_loop(
                region='region', num_nodes=1,
                instance_type='instance', accelerators=None,
                use_spot=False))
        self.assertEqual(result, [])


class TestQueryInstancesKeepsStopped(unittest.TestCase):
    def test_stopped_deployment_is_returned_even_when_non_terminated_only(self):
        """BEHAVIOURAL, not a source-text grep.

        The previous version asserted a filter expression was absent from
        `inspect.getsource(...)`, which passes just as happily if the function
        is deleted, renamed, or rewritten to drop STOPPED some other way. This
        drives the real call and asserts the STOPPED deployment comes back,
        which is the property that matters: a stopped deployment still holds
        (and bills for) its disks, so a caller reporting or reaping capacity
        must be able to see it.
        """
        from sky.provision.spheron import instance
        from sky.utils import status_lib

        deployments = [
            {'id': 'dep-stopped', 'status': 'stopped'},
            {'id': 'dep-running', 'status': 'running'},
            {'id': 'dep-failed', 'status': 'failed'},
        ]
        with mock.patch.object(instance, '_client', return_value=mock.MagicMock()), \
             mock.patch.object(instance, '_deployments_for_cluster',
                               return_value=deployments):
            got = instance.query_instances('c', 'c-on-cloud',
                                           non_terminated_only=True)

        self.assertEqual(got['dep-stopped'][0], status_lib.ClusterStatus.STOPPED)
        self.assertEqual(got['dep-running'][0], status_lib.ClusterStatus.UP)
        # A genuinely GONE deployment is still excluded, by the `is None`
        # branch -- so this never returns a terminated instance regardless.
        self.assertNotIn('dep-failed', got)


if __name__ == '__main__':
    unittest.main()
