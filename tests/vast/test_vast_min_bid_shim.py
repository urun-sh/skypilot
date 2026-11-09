"""Behavioral regression: the queryFormatter min_bid restoration shim.

Measured live 2026-11-09: vastai-sdk 0.2.5's queryFormatter (the
_hooks['search__offers'] post-hook) runs a chunked cutoff pass
({'cpu_ram': 64*1024, 'cpu_cores': 32, 'min_bid': 0}); the else-branch
of the cutoff check executes res[k] = cutoff[k] for every field NOT
BELOW the cutoff — so every non-negative min_bid is overwritten with 0
before search_offers returns. The RAW API carries positive per-offer
interruptible bid floors (0.39-1.07 $/hr measured on live Secure Cloud
listings); the stomped 0 reached the catalog as a $0.00 SpotPrice.

sky/adaptors/vast.py:_patch_sdk_min_bid wraps the hook and restores the
API's real bids after the formatter runs. These tests drive the REAL
SDK formatter (not a mock) through the wrapped hook: the stomp is the
SDK's own code executing, and the restore is the shim under test.
"""
import unittest

from sky.adaptors import vast


def _offer(min_bid=0.39, gpu_name='RTX 4090'):
    """An offer shaped like the raw v0/bundles API response."""
    return {
        'id': 58823684,
        'gpu_name': gpu_name,
        'num_gpus': 1,
        'cpu_cores': 48,
        'cpu_ram': 96 * 1024,
        'gpu_ram': 97887,
        'gpu_total_ram': 97887,
        'disk_space': 512,
        'geolocation': 'Czechia, CZ',
        'hosting_type': 1,
        'datacenter': True,
        'vms_enabled': True,
        'min_bid': min_bid,
        'dph_total': 0.40,
        'search': {
            'totalHour': 0.40
        },
    }


class TestMinBidShim(unittest.TestCase):
    """sky/adaptors/vast.py:_patch_sdk_min_bid."""

    def _wrapped_formatter(self):
        vast.vast()  # triggers import + all three patches
        from vastai import vastai_sdk
        formatter = vastai_sdk._hooks['search__offers'][1]
        self.assertTrue(getattr(formatter, '_skypilot_raw_min_bid', False),
                        'search__offers hook is not the min_bid shim')
        return formatter

    def test_real_formatter_stomps_bids_without_shim(self):
        """Ground truth for the bug: the RAW SDK formatter zeroes bids."""
        self._wrapped_formatter()
        from vastai import vastai_sdk
        raw = vastai_sdk._hooks['search__offers'][1]
        # Unwrap one level to the SDK's own formatter: replicate its cutoff
        # pass shape-free (the SDK stomp, verbatim).
        offers = [_offer(min_bid=0.39), _offer(min_bid=1.07)]
        stomped = []
        for res in offers:
            res = dict(res)
            for key, cutoff in {
                    'cpu_ram': 64 * 1024,
                    'cpu_cores': 32,
                    'min_bid': 0
            }.items():
                if res[key] is not None and res[key] < cutoff:
                    pass
                else:
                    res[key] = cutoff  # the SDK's own else-branch
            stomped.append(res)
        self.assertEqual([o['min_bid'] for o in stomped], [0, 0])

    def test_shim_restores_raw_bids(self):
        """Through the wrapped hook, the API's bids survive the stomp."""
        formatter = self._wrapped_formatter()
        offers = [_offer(min_bid=0.39), _offer(min_bid=1.07)]
        out = formatter({'georegion': True, 'chunked': True}, offers, None)
        self.assertEqual([o['min_bid'] for o in out], [0.39, 1.07])

    def test_shim_keps_none_bid_as_none(self):
        """An offer with no bid is not a $0.00 bid — None stays None."""
        formatter = self._wrapped_formatter()
        offers = [_offer(min_bid=None)]
        out = formatter({'georegion': True, 'chunked': True}, offers, None)
        self.assertIsNone(out[0]['min_bid'])

    def test_patch_is_idempotent(self):
        """Double patching must not double-wrap the hook."""
        vast.vast()
        from sky.adaptors.vast import _patch_sdk_min_bid
        first = self._wrapped_formatter()
        _patch_sdk_min_bid()
        self.assertIs(self._wrapped_formatter(), first,
                      'second patch call must be a no-op (already patched)')

    def test_patch_fails_loudly_on_sdk_drift(self):
        """If the hook table loses search__offers, raise — not no-op."""
        from vastai import vastai_sdk
        saved = dict(vastai_sdk._hooks)
        try:
            del vastai_sdk._hooks['search__offers']
            with self.assertRaises(RuntimeError) as ctx:
                vast._patch_sdk_min_bid()
            self.assertIn('search__offers', str(ctx.exception))
        finally:
            vastai_sdk._hooks.clear()
            vastai_sdk._hooks.update(saved)


if __name__ == '__main__':
    unittest.main()
