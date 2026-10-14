"""A runtime-refreshed catalog CSV must reach the catalog module's cache.

Regression for prod-usw2 2026-10-14 (3 SkyPilotLaunchFailed workers, e.g.
node-autoscaler-rtx6000-192b4b90b7-worker): shadeform's catalog CSV is
rewritten under the running processes by shadeform_catalog_refresh
(skypilot-controller), but the catalog modules cached the materialized
DataFrame in a module-level global forever. A long-lived interpreter (the
API server's executor pool) answered every request from the snapshot taken
at its FIRST shadeform catalog read, so the runner (a fresh subprocess, on a
fresh CSV) proposed shadeform/kansascity-usa-6 -- an RTXPro6000 x1 offer at
$2.19/GPU-hr the vendor really sells -- and the executor's frozen snapshot
rejected the launch with

    Invalid region 'kansascity-usa-6'
    Did you mean one of these: 'kansascity-usa-1'?

killing the launch before any failover. This file drives that exact
sequence against the real implementation: validate the region against a
stale CSV (ValueError, the prod message), rewrite the CSV the way the
refresher does (write temp + os.replace), and require the SAME process to
now accept the region, see it in the region universe, and price it.
"""

import os
import tempfile
import unittest
from unittest import mock

from sky.catalog import common as catalog_common
from sky.catalog import shadeform_catalog

_HEADER = ('InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,'
           'Price,Region,GpuInfo,SpotPrice')
# The stale prod catalog (vms.csv.prod-hosted-20260930T2051): massedcompute
# rows in kansascity-usa-1, zero kansascity-usa-6 rows.
_STALE_CSV = _HEADER + (
    '\n'
    'massedcompute_L40S,L40S,1.0,12.0,72,0.88,kansascity-usa-1,,\n'
    'massedcompute_RTX6000Ada,RTX6000Ada,1.0,12.0,48,0.97,kansascity-usa-1,,\n')
# The live Shadeform offer (read from the refreshed prod catalog 2026-10-14):
# massedcompute_RTXPro6000,RTXPro6000,1.0,16.0,192,2.19,kansascity-usa-6.
_FRESH_CSV = _HEADER + (
    '\n'
    'massedcompute_L40S,L40S,1.0,12.0,72,0.88,kansascity-usa-1,,\n'
    'massedcompute_RTXPro6000,RTXPro6000,1.0,16.0,192,2.19,'
    'kansascity-usa-6,,\n')


class TestCatalogDfTracksRuntimeRefresh(unittest.TestCase):
    """shadeform_catalog._get_df re-reads when the CSV is replaced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._csv = os.path.join(self._tmp.name, 'vms.csv')
        with open(self._csv, 'w', encoding='utf-8') as f:
            f.write(_STALE_CSV)
        # Both the stat check and read_catalog resolve the file through
        # get_catalog_path; redirecting it here redirects everything.
        patcher = mock.patch.object(catalog_common,
                                    'get_catalog_path',
                                    return_value=self._csv)
        patcher.start()
        self.addCleanup(patcher.stop)
        # These are module globals shared with every other test in the run.
        self._saved = (shadeform_catalog._df, shadeform_catalog._df_stamp)
        shadeform_catalog._df = None
        shadeform_catalog._df_stamp = None
        self.addCleanup(self._restore)

    def _restore(self):
        shadeform_catalog._df, shadeform_catalog._df_stamp = self._saved

    def _replace_csv(self, content: str) -> None:
        """Install a CSV the way shadeform_catalog_refresh does."""
        tmp_path = self._csv + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(tmp_path, self._csv)

    def test_frozen_df_rejects_region_the_refresh_then_adds(self):
        """The prod incident: stale snapshot -> ValueError; fresh CSV -> OK."""
        with self.assertRaises(ValueError) as cm:
            shadeform_catalog.validate_region_zone('kansascity-usa-6', None)
        self.assertIn("Invalid region 'kansascity-usa-6'", str(cm.exception))
        self.assertIn("'kansascity-usa-1'", str(cm.exception))

        self._replace_csv(_FRESH_CSV)

        # Same process, no reimport: the region must now validate...
        self.assertEqual(
            shadeform_catalog.validate_region_zone('kansascity-usa-6', None),
            ('kansascity-usa-6', None),
        )
        # ...appear in the candidate universe the runner enumerates...
        regions = shadeform_catalog.get_region_zones_for_instance_type(
            'massedcompute_RTXPro6000', use_spot=False)
        self.assertEqual([r.name for r in regions], ['kansascity-usa-6'])
        # ...and price the way the offer does ($2.19/GPU-hr on-demand).
        self.assertEqual(
            shadeform_catalog.get_hourly_cost('massedcompute_RTXPro6000',
                                              use_spot=False,
                                              region='kansascity-usa-6'),
            2.19,
        )

    def test_stat_mismatch_re_reads_without_growing_stale(self):
        """Repeated reads keep agreeing with the file, not the first read."""
        self._replace_csv(_FRESH_CSV)
        self.assertIsNotNone(
            shadeform_catalog.validate_region_zone('kansascity-usa-6', None))
        # The vendor withdraws the region (a stockout removes the row).
        self._replace_csv(_STALE_CSV)
        with self.assertRaises(ValueError):
            shadeform_catalog.validate_region_zone('kansascity-usa-6', None)


if __name__ == '__main__':
    unittest.main()
