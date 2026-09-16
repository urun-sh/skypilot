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


class TestPickOsFallsBackInsteadOfRaising(unittest.TestCase):
    """A GPU offer with only a plain Ubuntu image must still be provisionable.

    Spheron's massed-compute GPU offers list exactly one OS and no cuda-named
    image, so a hard `pick_os(["cuda"])` requirement made EVERY Spheron GPU
    offer unprovisionable — verified live: a launch reached the provisioner and
    died with

        SpheronError: offer 'gpu_1x_pro_6000_blackwell_us-central-9' has no OS
        matching ['cuda']; available: ['Ubuntu Server 22.04']

    The CUDA-less risk is covered for free one layer down by urun's bootstrap
    `runtime-gpu-probe` ($0 `nvidia-smi -L`, refuses before the paid boot).
    """

    def _offer(self, os_options):
        from sky.adaptors import spheron
        return spheron.Offer(
            provider="massed-compute",
            offer_id="gpu_1x_pro_6000_blackwell_us-central-9",
            gpu_type="RTXPRO6000-PCIE",
            gpu_count=1,
            gpu_memory_gb=96,
            vcpus=16.0,
            memory_gb=144.0,
            price_per_hour_usd=2.39,
            regions=["us-central-9"],
            os_options=list(os_options),
            instance_type="massed-compute_gpu_1x_pro_6000_blackwell_us-central-9",
            supports_cloud_init=True,
            maintenance=False,
            minimum_runtime_minutes=None,
        )

    def test_prefers_a_cuda_image_when_one_exists(self):
        from sky.adaptors import spheron
        o = self._offer(["Ubuntu Server 22.04", "Ubuntu 22.04 CUDA 12.4"])
        self.assertIn("CUDA", o.pick_os(["cuda"]))

    def test_falls_back_to_the_only_os_instead_of_raising(self):
        from sky.adaptors import spheron
        o = self._offer(["Ubuntu Server 22.04"])
        # Must NOT raise: this is the exact shape that blocked every launch.
        self.assertEqual(o.pick_os(["cuda"]), "Ubuntu Server 22.04")

    def test_still_fails_hard_when_the_offer_lists_no_os_at_all(self):
        from sky.adaptors import spheron
        o = self._offer([])
        with self.assertRaises(spheron.SpheronError):
            o.pick_os(["cuda"])
