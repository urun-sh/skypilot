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
    def test_stopped_filter_removed(self):
        from sky.provision.spheron import instance
        source = inspect.getsource(instance)
        self.assertNotIn(
            'non_terminated_only and sky_status == status_lib.ClusterStatus.STOPPED',
            source)


class TestG4FractionalCounts(unittest.TestCase):
    def test_fractional_mapping(self):
        from sky.catalog import gcp_catalog
        d = gcp_catalog._ACC_INSTANCE_TYPE_DICTS['RTXPRO6000']
        self.assertEqual(d[0.125], ['g4-standard-6'])
        self.assertEqual(d[0.25], ['g4-standard-12'])
        self.assertEqual(d[0.5], ['g4-standard-24'])
        self.assertEqual(d[1], ['g4-standard-48'])
        self.assertNotIn('g4-standard-6', d.get(1, []))


if __name__ == '__main__':
    unittest.main()
