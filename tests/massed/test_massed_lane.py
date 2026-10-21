"""Behavioral tests for the MassedCompute lane outside the client: the
catalog fetcher's row selection, the zero-stock cross-check, and the
provisioner's launch decisions (productName passthrough, the image-id
re-resolve, the SSH-key registration by name).

These exercise actual calls with fixtures/mocks — not source-text
inspection — so a regression fails with a behavior mismatch.
"""

import unittest
from unittest import mock

from sky.catalog.data_fetchers import fetch_massed
from sky.provision.massed import utils
from sky.provision.massed import instance as provision


def _product(name='gpu_1x_pro_6000_blackwell',
             description='1x RTX PRO 6000 (96GB)',
             cents=219,
             vcpu=16,
             ram_gib=144,
             storage_gb=725,
             capacity=19,
             regions=('us-east-1', 'us-central-9')):
    """One gpu_inventory entry in the live API's shape (recorded
    2026-10-21: the products' specs carry SYSTEM RAM, not GPU RAM — the
    per-GPU vram is parsed from the description's '(96GB)')."""
    return {
        'instance_type': {
            'name': name,
            'description': description,
            'price_cents_per_hour': cents,
            'specs': {
                'vcpu_count': vcpu,
                'memory_gib': ram_gib,
                'storage_gb': storage_gb,
            },
        },
        'regions_with_capacity_available': [
            {'name': r, 'description': r} for r in regions
        ],
        'capacity_available': capacity,
    }


def _inventory(**overrides):
    base = {'gpu_1x_pro_6000_blackwell': _product()}
    base.update(overrides)
    return base


class TestFetcherRows(unittest.TestCase):
    """iter_rows: the (product, region) rows the lane will actually rent."""

    def _rows(self, inventory):
        # CSV_COLUMNS order: InstanceType, AcceleratorName, AcceleratorCount,
        # vCPUs, MemoryGiB, Price, Region, GpuInfo, SpotPrice, AvailableUnits.
        return {(r[0], r[6]): r for r in fetch_massed.iter_rows(inventory)}

    def test_pro_6000_blackwell_row_matches_live_market(self):
        # The MEASURED bucket (live probe 2026-10-21): $2.19/hr whole box,
        # capacity 19, us-east-1 + us-central-9 — the lane's target SKU.
        rows = self._rows(_inventory())
        self.assertEqual(len(rows), 2)  # one row per region
        us_east = rows[('gpu_1x_pro_6000_blackwell', 'us-east-1')]
        self.assertEqual(us_east[1], 'RTXPRO6000')  # the family token
        self.assertEqual(us_east[2], 1)  # AcceleratorCount
        self.assertEqual(us_east[5], 2.19)  # Price: 219 cents -> $2.19
        self.assertIn('98304', us_east[7])  # 96 GiB vram in MiB (GpuInfo)
        self.assertEqual(us_east[9], 19)  # AvailableUnits
        self.assertEqual(us_east[3], 16)  # vCPUs
        self.assertEqual(us_east[4], 144)  # system RAM GiB

    def test_cpu_products_never_yield_rows(self):
        # gpu_<N>x_<slug> is the GPU grammar; cpu_* products must not
        # match (they carry no accelerator token).
        rows = list(
            fetch_massed.iter_rows({
                'cpu_16_small': {
                    'instance_type': {
                        'name': 'cpu_16_small',
                        'description': '16 vCPU',
                        'price_cents_per_hour': 10,
                        'specs': {'vcpu_count': 16, 'memory_gib': 32,
                                  'storage_gb': 100},
                    },
                    'regions_with_capacity_available': [
                        {'name': 'us-east-1', 'description': 'us-east-1'}],
                    'capacity_available': 5,
                },
            }))
        self.assertEqual(rows, [])

    def test_zero_capacity_product_is_skipped(self):
        # A sold-out product is a fact, not a row: capacity 0 prices nothing.
        rows = list(
            fetch_massed.iter_rows(
                _inventory(**{
                    'gpu_8x_pro_6000_blackwell': _product(
                        name='gpu_8x_pro_6000_blackwell', cents=1752,
                        capacity=0, regions=())
                })))
        product_names = {r[0] for r in rows}
        self.assertNotIn('gpu_8x_pro_6000_blackwell', product_names)
        self.assertEqual(product_names, {'gpu_1x_pro_6000_blackwell'})


class TestZeroStockCrossCheck(unittest.TestCase):
    """verify_zero_stock: the honest-empty vs contradiction ruling."""

    def test_stocked_inventory_with_no_rows_refuses(self):
        # GPU capacity reported but no row renderable: NEVER write
        # "no capacity" from a contradictory fetch.
        with self.assertRaises(fetch_massed.MassedCatalogError):
            fetch_massed.verify_zero_stock(_inventory(), rows=[])

    def test_honest_zero_stock_passes(self):
        empty = _inventory(**{
            'gpu_1x_pro_6000_blackwell': _product(capacity=0, regions=())
        })
        fetch_massed.verify_zero_stock(empty, rows=[])

    def test_rows_present_never_refuses(self):
        rows = list(fetch_massed.iter_rows(_inventory()))
        self.assertTrue(rows)
        fetch_massed.verify_zero_stock(_inventory(), rows=rows)


class TestProvisionerLaunch(unittest.TestCase):
    """run_instances: the productName passthrough and the launch body."""

    def _config(self, instance_type='gpu_1x_pro_6000_blackwell',
                public_key='ssh-ed25519 AAAA real-material'):
        node_config = {'InstanceType': instance_type}
        if public_key is not None:
            node_config['PublicKey'] = public_key
        record = mock.Mock()
        record.node_config = node_config
        return record

    def test_launch_passes_product_name_and_registered_ssh_key(self):
        client = mock.Mock()
        client.list_instances.return_value = []
        client.ensure_ssh_key.return_value = 'urun-skypilot'
        client.default_image_id.return_value = 7
        client.launch_instance.return_value = 'uuid-1'
        client.wait_until_booted.return_value = {
            'uuid': 'uuid-1', 'ip': '1.2.3.4', 'os_booted': 1,
        }
        with mock.patch.object(provision, '_client', return_value=client):
            record = provision.run_instances(
                region='us-east-1',
                cluster_name='c',
                cluster_name_on_cloud='urun-massed-e2e',
                config=self._config(),
            )
        self.assertEqual(record.head_instance_id, 'uuid-1')
        client.launch_instance.assert_called_once_with(
            image_id=7,
            product_name='gpu_1x_pro_6000_blackwell',
            region_name='us-east-1',
            instance_name='urun-massed-e2e',
            ssh_key_names=['urun-skypilot'],
        )

    def test_missing_instance_type_refuses_before_any_api_call(self):
        client = mock.Mock()
        with mock.patch.object(provision, '_client', return_value=client):
            with self.assertRaises(utils.MassedError):
                provision.run_instances(
                    region='us-east-1',
                    cluster_name='c',
                    cluster_name_on_cloud='urun-massed-e2e',
                    config=self._config(instance_type=''),
                )
        client.launch_instance.assert_not_called()

    def test_placeholder_public_key_refuses(self):
        # The template's unsubstituted placeholder is TRUTHY: a bare
        # `if not public_key` would register a garbage key the box can
        # never answer (the Spheron trap).
        client = mock.Mock()
        with mock.patch.object(provision, '_client', return_value=client):
            with self.assertRaises(utils.MassedError):
                provision.run_instances(
                    region='us-east-1',
                    cluster_name='c',
                    cluster_name_on_cloud='urun-massed-e2e',
                    config=self._config(
                        public_key=provision._SSH_PUBLIC_KEY_PLACEHOLDER),
                )
        client.launch_instance.assert_not_called()


if __name__ == '__main__':
    unittest.main()
