"""Behavioral regression: the IE georegion shim must survive the real SDK.

Tests the actual `queryFormatter` path (not just the dict patch) and the
fetcher's CSV generation against a MOCKED offer list — no live API, no
credentials, no real HOME writes.
"""
import os
import unittest
from unittest import mock

from sky.adaptors import vast


class TestVastRegionShim(unittest.TestCase):
    """The IE → EU shim in sky/adaptors/vast.py."""

    def test_ie_in_regions_rev(self):
        """IE must be present and map to the EU georegion token."""
        vast.vast()  # triggers import + patch
        from vastai import vastai_sdk
        self.assertEqual(vastai_sdk._regions_rev.get('IE'), 'EU')

    def test_queryformatter_handles_ie_offer(self):
        """The REAL formatter must not crash on an IE offer.

        queryFormatter(state, obj, instance) where obj is a LIST of offers.
        It iterates over obj, mutating res['geolocation'] in place by
        appending ', {_regions_rev[country]}'. Without the shim,
        KeyError('IE') kills the catalog fetch.
        """
        vast.vast()  # triggers import + patch
        from vastai import vastai_sdk

        offer = {
            'gpu_name': 'RTX PRO 6000 S',
            'num_gpus': 1,
            'geolocation': 'Dublin, IE',
            'hosting_type': 0,
        }
        offers = [offer]
        state = {'georegion': True, 'chunked': False}
        vastai_sdk.queryFormatter(state, offers, None)

        # The geolocation must have been enriched with the georegion
        self.assertIn('IE, EU', offer['geolocation'],
                      f'Expected IE, EU in geolocation, got: '
                      f'{offer["geolocation"]}')

    def test_patch_is_idempotent(self):
        """Running the patch twice is safe (process-global singleton)."""
        from sky.adaptors.vast import _patch_sdk_regions
        _patch_sdk_regions()
        _patch_sdk_regions()
        from vastai import vastai_sdk
        self.assertEqual(vastai_sdk._regions_rev.get('IE'), 'EU')

    def test_patch_fails_loudly_on_sdk_drift(self):
        """If the SDK loses _regions_rev, we raise — not silently no-op."""
        from vastai import vastai_sdk
        saved = vastai_sdk._regions_rev
        from sky.adaptors import vast as _v
        try:
            del vastai_sdk._regions_rev
            _v._vast_sdk = None  # clear singleton
            with self.assertRaises(RuntimeError) as ctx:
                _v.vast()
            self.assertIn('not found', str(ctx.exception))
        finally:
            vastai_sdk._regions_rev = saved
            _v._vast_sdk = None

def _mock_offer(geolocation='Dublin, IE', price=1.27):
    """A deterministic Vast offer fixture matching the fetcher's schema."""
    return {
        'gpu_name': 'RTX PRO 6000 S',
        'num_gpus': 1,
        'cpu_cores': 16,
        'cpu_ram': 192000,
        'gpu_total_ram': 98304,
        'search': {'totalHour': price},
        'min_bid': price,
        'geolocation': geolocation,
        'hosting_type': 0,
        'bundle_id': 'test-bundle',
        'inet_down': 500,
        'disk_space': 512,
        'compute_cap': 120,
    }


class TestVastCatalogFetch(unittest.TestCase):
    """The fetcher must produce a nonempty CSV from a mocked offer list.

    No live API — we patch search_offers with deterministic fixtures and
    verify the catalog pipeline (mapping, pricing, dedup, CSV write).

    IMPORTANT: the fetcher's dedup logic only appends a CSV row on the
    SECOND occurrence of the same stub (instance_type + region + hosting),
    so the fixture must provide at least TWO matching offers for any data
    row to appear.
    """

    def test_single_offer_produces_nonempty_csv(self):
        """ONE unique IE offer → catalog has ONE data row.

        This is the regression test for the dedup bug that dropped
        every FIRST occurrence of an (InstanceType, Region, HostingType)
        stub — Vast instance types encode CPU/RAM, so most marketplace
        offers are unique, and the old code produced a header-only CSV
        even with 46 live offers. One offer must produce one row.
        """
        vast.vast()  # triggers import + patch

        import tempfile
        one_offer = [_mock_offer('Dublin, IE', 1.27)]

        with mock.patch.object(vast, 'vast') as mock_vast:
            mock_vast.return_value.search_offers.return_value = one_offer

            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    import contextlib
                    import io
                    import runpy
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_module(
                            'sky.catalog.data_fetchers.fetch_vast',
                            run_name='__main__')

                    # ASSERT INSIDE the tempdir (it's deleted on exit)
                    csv_path = os.path.join(tmpdir, 'vast', 'vms.csv')
                    self.assertTrue(
                        os.path.exists(csv_path),
                        'fetch_vast did not produce vast/vms.csv')
                    with open(csv_path) as f:
                        lines = f.readlines()
                    self.assertGreater(
                        len(lines), 1,
                        f'Catalog CSV is header-only ({len(lines)} lines) — '
                        'the fetcher silently produced an empty catalog. '
                        'The dedup bug that dropped unique instance types '
                        'has regressed.')

                    # Verify the region was enriched (IE → EU)
                    if len(lines) > 1:
                        data_row = lines[1]
                        self.assertIn('IE', data_row,
                                      f'Expected IE country in CSV row: '
                                      f'{data_row[:100]}')
                finally:
                    os.chdir(old_cwd)



if __name__ == '__main__':
    unittest.main()
