"""Behavioral tests for the io.net lane outside the network: the catalog
fetcher's row selection against the REAL hardware fixture (the live
snapshot recorded 2026-11-03 at /tmp/ionet-hw-fixture.json), the client's
wire contract (x-api-key auth, the Cloudflare-mandated custom
User-Agent, the price oracle's two answer shapes, the deploy body), the
provisioner's launch decisions (deploy_id passthrough, gpus_per_vm, the
inline ssh-key material, the ssh_access address parsing), the catalog
module's lookups against an in-memory CSV, and the cloud class's
identity/credential surface.

These exercise actual calls with fixtures/mocks — not source-text
inspection — so a regression fails with a behavior mismatch. NO live
network calls anywhere: the client is transport-injected throughout.
"""

import csv
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from sky.adaptors import ionet as ionet_api
from sky.catalog.data_fetchers import fetch_ionet
from sky.catalog import ionet_catalog
from sky.clouds import ionet as ionet_cloud
from sky.clouds.cloud import CloudCapability
from sky.provision.ionet import instance as provision
from sky.provision.ionet import utils

# The REAL live snapshot of GET /hardware taken 2026-11-03 (48-config
# market, 47 rows in the file): the fetcher's entire input, recorded
# once so the row-emission tests pin the real payload shape. CHECKED IN
# (tests/ionet/hw_fixture.json) so CI exercises the fixture paths
# deterministically — an absolute /tmp path would silently skip them.
_FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'hw_fixture.json')

_HAVE_FIXTURE = os.path.exists(_FIXTURE_PATH)


def _fixture_hardware():
    with open(_FIXTURE_PATH, encoding='utf-8') as handle:
        return json.load(handle)['data']['hardware']


def _row(**overrides):
    """One hardware row in the live API's exact shape (recorded
    2026-11-03: the 8x H100 FI config at $41.2418.../hr for the whole
    box)."""
    base = {
        'id': '8H100.80S.176V__FI',
        'deploy_id': '8H100.80S.176V',
        'name': 'H100',
        'num_cards': 8,
        'supplier': 'external',
        'network_mode': 'open',
        'price': 41.241842465753415,
        'vram_per_card': 80,
        'interconnect': 'sxm5',
        'nvlink': True,
        'storage': 2000,
        'vcpu': 176,
        'memory': 1360,
        'location': 'FI',
    }
    base.update(overrides)
    return base


class _Transport:
    """Recorded transport: answers one scripted (status, body) per call
    and captures every (method, url, headers, payload) it saw."""

    def __init__(self, *answers):
        self._answers = list(answers)
        self.calls = []

    def __call__(self, method, url, headers, payload):
        self.calls.append((method, url, headers, payload))
        status, body = self._answers.pop(0)
        return status, json.dumps(body).encode('utf-8')


_PRICE_OK = (200, {
    'data': {
        'replica_count': 1,
        'gpus_per_vm': 8,
        'available_replica_count': [1],
        'total_cost_usdc': 41.24,
    },
})

_PRICE_SOLD_OUT = (200, {
    'status': 'failed',
    'error': 'No available instances found for deployment',
})


class TestFetcherRows(unittest.TestCase):
    """iter_rows: the (deploy_id, location) rows the lane will rent."""

    def _rows(self, hardware, probe):
        # CSV_COLUMNS order: InstanceType, AcceleratorName,
        # AcceleratorCount, vCPUs, MemoryGiB, Price, Region, GpuInfo,
        # SpotPrice — NO AvailableUnits (in_stock tier).
        return {(r[0], r[6]): r for r in fetch_ionet.iter_rows(hardware,
                                                               probe)}

    def test_h100_row_matches_live_snapshot(self):
        rows = self._rows([_row()], lambda row: True)
        row = rows[('8H100.80S.176V', 'fi')]
        self.assertEqual(row[1], 'H100')  # the family token
        self.assertEqual(row[2], 8)  # AcceleratorCount
        self.assertEqual(row[3], 176.0)  # vCPUs
        self.assertEqual(row[4], 1360.0)  # system RAM GiB
        self.assertAlmostEqual(row[5], 41.241842465753415)  # per-VM $/hr
        self.assertEqual(row[6], 'fi')  # lowercase location token
        # 80 GiB vram/card in MiB, in the single-quoted GpuInfo shape.
        self.assertIn("'SizeInMiB': 81920", row[7])
        self.assertEqual(row[8], '')  # no spot tier

    def test_sold_out_config_is_excluded(self):
        # The in_stock-tier ruling: a row exists only when the price
        # probe succeeds; the sold-out config must not surface (a row
        # the deploy must refuse would let the quote admit candidates
        # the launch cannot serve).
        rows = self._rows([_row(), _row(name='RTX6000Ada',
                                         deploy_id='1RTX6000Ada.16V',
                                         num_cards=1,
                                         location='FI')],
                           lambda row: row['name'] != 'RTX6000Ada')
        self.assertEqual(list(rows), [('8H100.80S.176V', 'fi')])

    def test_probe_exception_fails_closed(self):
        # A probe that blows up is NOT stock: never emit a row on the
        # strength of an error.
        def exploding_probe(row):
            raise RuntimeError('boom')
        rows = list(fetch_ionet.iter_rows([_row()], exploding_probe))
        self.assertEqual(rows, [])

    def test_unpriceable_config_never_quotes_free(self):
        rows = self._rows([_row(price=None)], lambda row: True)
        self.assertEqual(rows, {})

    def test_accelerator_name_normalization(self):
        # The vendor's raw names -> the platform's compact identities
        # (measured live 2026-11-03 + the docs' variant spellings).
        cases = {
            'H100': 'H100',
            'A100': 'A100',
            'DGX A100': 'A100',
            'H200': 'H200',
            'L4': 'L4',
            'A30': 'A30',
            'L40': 'L40',
            'RTX4090': 'RTX4090',
            'RTXPro6000': 'RTXPRO6000',
            'PRO 6000 BLACKWELL': 'RTXPRO6000',
            '6000 ADA': 'RTX6000ADA',
            'RTX6000Ada': 'RTX6000ADA',
            'V100': 'V100',
            'V100_32G': 'V100-32GB',
            'B200': 'B200',
            'NVIDIA GeForce RTX 5090': 'RTX5090',  # unknown: compacted
        }
        for raw, token in cases.items():
            self.assertEqual(fetch_ionet.normalise_accelerator(raw), token)

    @unittest.skipUnless(_HAVE_FIXTURE,
                         'the live hardware snapshot fixture is absent')
    def test_real_fixture_emits_rows_with_lowercase_regions(self):
        hardware = _fixture_hardware()
        regions = {str(h.get('location') or '').lower()
                   for h in hardware}
        rows = list(fetch_ionet.iter_rows(hardware, lambda row: True))
        self.assertTrue(rows)
        self.assertTrue({r[6] for r in rows} <= regions)
        # InstanceType is ALWAYS the deploy_id (the stable identity the
        # deploy API accepts back).
        for row in rows:
            self.assertTrue(row[0])
            self.assertEqual(len(row), len(fetch_ionet.CSV_COLUMNS))


class TestRefuseEmpty(unittest.TestCase):
    """verify_nonempty: the honest-empty vs contradiction ruling."""

    def test_empty_hardware_list_refuses(self):
        # An EMPTY hardware list is never an honest stock-out: it means
        # the API is unreachable, the key is unconfigured, or the
        # payload shape drifted.
        with self.assertRaises(fetch_ionet.IonetCatalogError):
            fetch_ionet.verify_nonempty([], rows=[])

    def test_all_probes_failed_refuses(self):
        with self.assertRaises(fetch_ionet.IonetCatalogError):
            fetch_ionet.verify_nonempty([_row()], rows=[])

    @unittest.skipUnless(_HAVE_FIXTURE,
                         'the live hardware snapshot fixture is absent')
    def test_fixture_with_all_probes_failing_refuses(self):
        hardware = _fixture_hardware()
        with self.assertRaises(fetch_ionet.IonetCatalogError):
            fetch_ionet.verify_nonempty(hardware, rows=[])

    def test_rows_present_never_refuses(self):
        rows = list(fetch_ionet.iter_rows([_row()], lambda row: True))
        fetch_ionet.verify_nonempty([_row()], rows=rows)


class TestClientWireContract(unittest.TestCase):
    """The client's HTTP contract, transport-injected (no network)."""

    def test_empty_key_refuses(self):
        with self.assertRaises(ionet_api.IonetAuthError):
            ionet_api.IonetClient('')

    def test_auth_header_and_custom_user_agent_on_every_request(self):
        # The provider's Cloudflare bans python-urllib's default UA
        # (HTTP 403 Error 1010, measured live 2026-11-03): EVERY request
        # must carry the lane's custom UA plus the x-api-key header.
        transport = _Transport((200, {'data': {'hardware': []}}))
        client = ionet_api.IonetClient('secret-key', transport=transport)
        client.list_hardware()
        method, url, headers, _payload = transport.calls[0]
        self.assertEqual(method, 'GET')
        self.assertTrue(url.startswith(ionet_api.API_BASE))
        self.assertEqual(headers['x-api-key'], 'secret-key')
        self.assertEqual(headers['User-Agent'],
                         'urun-skypilot-ionet/0.1 (+https://urun.sh)')

    def test_price_check_success_returns_the_data_dict(self):
        transport = _Transport(_PRICE_OK)
        client = ionet_api.IonetClient('k', transport=transport)
        data = client.price_check('8H100.80S.176V', 8, 'FI')
        self.assertEqual(data['gpus_per_vm'], 8)
        self.assertAlmostEqual(data['total_cost_usdc'], 41.24)
        # location_ids rides the query as the URL-encoded JSON array of
        # one country code — the provider's documented shape.
        url = transport.calls[0][1]
        self.assertIn('location_ids=%5B%22FI%22%5D', url)
        self.assertIn('hardware_id=8H100.80S.176V', url)
        self.assertIn('gpus_per_vm=8', url)

    def test_price_check_sold_out_envelope_is_none(self):
        # The sold-out answer is HTTP 200 with {"status": "failed"} —
        # NOT an HTTP error; it must read as "not offered right now".
        transport = _Transport(_PRICE_SOLD_OUT)
        client = ionet_api.IonetClient('k', transport=transport)
        self.assertIsNone(
            client.price_check('1RTX6000Ada.16V', 1, 'FI'))

    def test_401_maps_to_auth_error(self):
        transport = _Transport((401, {'error': 'bad key'}))
        client = ionet_api.IonetClient('k', transport=transport)
        with self.assertRaises(ionet_api.IonetAuthError):
            client.list_hardware()

    def test_404_maps_to_not_found(self):
        transport = _Transport((404, {'error': 'gone'}))
        client = ionet_api.IonetClient('k', transport=transport)
        with self.assertRaises(ionet_api.IonetNotFoundError):
            client.get_deployment('dep-1')

    def test_429_is_retried_then_succeeds(self):
        transport = _Transport((429, {}), (429, {}), _PRICE_OK)
        client = ionet_api.IonetClient('k', transport=transport)
        with mock.patch.object(ionet_api.time, 'sleep'):
            data = client.price_check('8H100.80S.176V', 8, 'FI')
        self.assertEqual(data['replica_count'], 1)
        self.assertEqual(len(transport.calls), 3)

    def test_destroy_tolerates_already_gone(self):
        transport = _Transport((404, {'error': 'gone'}))
        client = ionet_api.IonetClient('k', transport=transport)
        client.destroy_deployment('dep-1')  # a 404 is success here

    def test_deployment_list_and_vms_shapes(self):
        deployments = {'data': {'deployments': [
            {'id': 'dep-1', 'name': 'urun-ionet-e2e', 'status':
             'running'},
        ]}}
        workers = {'data': {'workers': [
            {'vm_id': 'vm-1', 'ssh_access':
             'ssh -p 2222 ubuntu@box.example', 'status': 'active'},
        ]}}
        transport = _Transport((200, deployments), (200, workers))
        client = ionet_api.IonetClient('k', transport=transport)
        rows = client.list_deployments()
        self.assertEqual([ionet_api.IonetClient.deployment_id(r)
                          for r in rows], ['dep-1'])
        vms = client.get_deployment_vms('dep-1')
        self.assertEqual(ionet_api.IonetClient.worker_address(vms[0]),
                         ('ubuntu', 'box.example', 2222))


class TestDeploymentListPagination(unittest.TestCase):
    """CodeRabbit round 1: the identity source walks EVERY page.

    list_deployments is the provisioner's cluster identity source — a
    deployment past page 1 must not read as absent (duplicate deploys,
    skipped teardown), so the client follows pages until a short page
    and dedupes by id.
    """

    @staticmethod
    def _page(*ids):
        return (200, {'data': {'deployments': [
            {'id': i, 'name': f'c-{i}', 'status': 'running'} for i in ids
        ]}})

    def test_pages_are_followed_until_a_short_page(self):
        page_size = 10
        page1 = [f'dep-{i}' for i in range(page_size)]
        # Page 2 OVERLAPS page 1 (the API documents no cross-page
        # ordering guarantee) and ends one row short.
        page2 = page1[5:] + [f'dep-{i}' for i in range(page_size,
                                                      page_size + 4)]
        transport = _Transport(self._page(*page1), self._page(*page2))
        client = ionet_api.IonetClient('k', transport=transport)
        rows = client.list_deployments(page_size=page_size)
        ids = [r['id'] for r in rows]
        self.assertEqual(sorted(ids), sorted(set(page1 + page2)))
        self.assertEqual(len(ids), len(set(ids)))  # dedupe held
        self.assertEqual(len(transport.calls), 2)  # stopped at short page

    def test_unbounded_account_stops_at_the_cap(self):
        page_size = 2
        full = self._page('dep-1', 'dep-2')
        transport = _Transport(*([full] * ionet_api.MAX_DEPLOYMENT_PAGES))
        client = ionet_api.IonetClient('k', transport=transport)
        with self.assertRaises(ionet_api.IonetError):
            client.list_deployments(page_size=page_size)
        self.assertEqual(len(transport.calls),
                         ionet_api.MAX_DEPLOYMENT_PAGES)


class TestFixtureModeOffline(unittest.TestCase):
    """--from-fixture is OFFLINE (CodeRabbit round 1): no client is
    constructed, so no key and no network can be required."""

    @unittest.skipUnless(_HAVE_FIXTURE, 'the fixture is checked in')
    def test_from_fixture_needs_neither_key_nor_network(self):
        # Any client construction would raise at the first call with no
        # key (the auth layer) — success therefore proves the offline path.
        env = {k: v for k, v in os.environ.items()
               if k != ionet_api.API_KEY_ENV}
        out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out_dir, True)
        out_path = os.path.join(out_dir, 'vms.csv')
        try:
            saved = os.environ.copy()
            os.environ.clear()
            os.environ.update(env)
            rc = fetch_ionet.main(['--from-fixture', _FIXTURE_PATH,
                                   '--output', out_path])
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.assertEqual(rc, 0)
        with open(out_path, encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        # Every fixture config is emitted as offered (the replay has no
        # stock oracle — in-stock claims it cannot make, offered it can).
        self.assertEqual(len(rows),
                         len(_fixture_hardware()))
        self.assertTrue(all(r['SpotPrice'] == '' for r in rows))


class TestDeployBody(unittest.TestCase):
    """POST /deploy: the documented body, never the deprecated fields."""

    def _deploy(self, **overrides):
        transport = _Transport((200, {'data': {'id': 'dep-1'}}))
        client = ionet_api.IonetClient('k', transport=transport)
        body = {
            'resource_private_name': 'urun-ionet-e2e',
            'duration_hours': 1,
            'hardware_id': '8H100.80S.176V',
            'gpus_per_vm': 8,
            'location_ids': ['US'],
            'ssh_keys': {'skypilot': 'ssh-ed25519 AAAA real-material'},
        }
        body.update(overrides)
        deployment_id = client.deploy(**body)
        sent = json.loads(transport.calls[0][3].decode('utf-8'))
        return deployment_id, sent

    def test_body_shape_matches_the_documented_contract(self):
        deployment_id, sent = self._deploy()
        self.assertEqual(deployment_id, 'dep-1')
        self.assertEqual(sent['resource_private_name'], 'urun-ionet-e2e')
        self.assertEqual(sent['hardware_id'], '8H100.80S.176V')
        self.assertEqual(sent['gpus_per_vm'], 8)
        self.assertEqual(sent['location_ids'], ['US'])
        # The keys ride inline as name -> MATERIAL (no registry).
        self.assertEqual(
            sent['ssh_keys'],
            {'skypilot': 'ssh-ed25519 AAAA real-material'})
        self.assertEqual(sent['vm_image_type'], 'general')
        # vms_qty is deprecated — never sent.
        self.assertNotIn('vms_qty', sent)

    def test_empty_name_refuses_before_any_api_call(self):
        client = ionet_api.IonetClient('k', transport=_Transport())
        with self.assertRaises(ionet_api.IonetError):
            client.deploy(resource_private_name='',
                          duration_hours=1,
                          hardware_id='8H100.80S.176V',
                          gpus_per_vm=8,
                          location_ids=['US'],
                          ssh_keys={'skypilot': 'ssh-ed25519 AAAA'})

    def test_empty_ssh_key_material_refuses(self):
        client = ionet_api.IonetClient('k', transport=_Transport())
        with self.assertRaises(ionet_api.IonetError):
            client.deploy(resource_private_name='urun-ionet-e2e',
                          duration_hours=1,
                          hardware_id='8H100.80S.176V',
                          gpus_per_vm=8,
                          location_ids=['US'],
                          ssh_keys={'skypilot': ''})


class TestSshAccessParsing(unittest.TestCase):
    """parse_ssh_access / worker_address: the UI command string is the
    address source, with the network_services fallback."""

    def test_full_command_with_port_and_user(self):
        address = ionet_api.IonetClient.parse_ssh_access(
            {'ssh_access': 'ssh -p 2222 ubuntu@box.example'})
        self.assertEqual(address, ('ubuntu', 'box.example', 2222))

    def test_command_without_port_defaults(self):
        address = ionet_api.IonetClient.parse_ssh_access(
            {'ssh_access': 'ssh root@5.6.7.8'})
        self.assertEqual(address, ('root', '5.6.7.8', 22))

    def test_command_with_stray_options(self):
        address = ionet_api.IonetClient.parse_ssh_access(
            {'ssh_access':
             'ssh -o StrictHostKeyChecking=no -p 2200 root@h.io'})
        self.assertEqual(address, ('root', 'h.io', 2200))

    def test_non_ssh_shape_refuses(self):
        self.assertIsNone(ionet_api.IonetClient.parse_ssh_access(
            {'ssh_access': 'telnet h.io'}))
        self.assertIsNone(ionet_api.IonetClient.parse_ssh_access(
            {'ssh_access': ''}))

    def test_network_services_fallback(self):
        worker = {
            'ssh_access': '',
            'network_services': {
                'ssh': {'port': 22, 'protocol': 'tcp',
                        'public_ip': '5.6.7.8', 'public_port': 2222},
            },
        }
        self.assertEqual(ionet_api.IonetClient.worker_address(worker),
                         ('root', '5.6.7.8', 2222))

    def test_no_address_at_all_is_none(self):
        self.assertIsNone(ionet_api.IonetClient.worker_address({}))


class TestProvisionerLaunch(unittest.TestCase):
    """run_instances: the deploy passthrough and the refusal guards."""

    def _config(self, instance_type='8H100.80S.176V', gpus_per_vm=8,
                public_key='ssh-ed25519 AAAA real-material'):
        node_config = {'InstanceType': instance_type,
                      'GpusPerVm': gpus_per_vm}
        if public_key is not None:
            node_config['PublicKey'] = public_key
        record = mock.Mock()
        record.node_config = node_config
        return record

    def _client(self):
        client = mock.Mock()
        client.list_deployments.return_value = []
        client.deploy.return_value = 'dep-1'
        client.wait_until_running.return_value = {
            'id': 'dep-1', 'status': 'running'
        }
        return client

    def test_launch_passes_deploy_id_location_and_inline_key(self):
        client = self._client()
        with mock.patch.object(provision, '_client',
                               return_value=client):
            record = provision.run_instances(
                region='us',
                cluster_name='c',
                cluster_name_on_cloud='urun-ionet-e2e',
                config=self._config(),
            )
        self.assertEqual(record.head_instance_id, 'dep-1')
        client.deploy.assert_called_once_with(
            resource_private_name='urun-ionet-e2e',
            duration_hours=provision._DEPLOY_DURATION_HOURS,
            hardware_id='8H100.80S.176V',
            gpus_per_vm=8,
            location_ids=['US'],  # the catalog Region, uppercased
            ssh_keys={'skypilot': 'ssh-ed25519 AAAA real-material'},
            vm_image_type='general',
        )
        client.wait_until_running.assert_called_once_with('dep-1')

    def test_existing_deployment_is_adopted_not_redeployed(self):
        client = self._client()
        existing = {'id': 'dep-9', 'name': 'urun-ionet-e2e',
                    'status': 'running'}
        with mock.patch.object(provision, '_client',
                               return_value=client), \
             mock.patch.object(provision, '_find_by_name',
                               return_value=existing):
            record = provision.run_instances(
                region='us',
                cluster_name='c',
                cluster_name_on_cloud='urun-ionet-e2e',
                config=self._config(),
            )
        client.deploy.assert_not_called()
        self.assertEqual(record.head_instance_id, 'dep-9')

    def test_missing_instance_type_refuses_before_any_api_call(self):
        client = self._client()
        with mock.patch.object(provision, '_client',
                               return_value=client):
            with self.assertRaises(utils.IonetError):
                provision.run_instances(
                    region='us',
                    cluster_name='c',
                    cluster_name_on_cloud='urun-ionet-e2e',
                    config=self._config(instance_type=''),
                )
        client.deploy.assert_not_called()

    def test_missing_gpus_per_vm_refuses(self):
        # The deploy body REQUIRES gpus_per_vm (the config's card
        # count); guessing it would rent a different box than priced.
        client = self._client()
        with mock.patch.object(provision, '_client',
                               return_value=client):
            with self.assertRaises(utils.IonetError):
                provision.run_instances(
                    region='us',
                    cluster_name='c',
                    cluster_name_on_cloud='urun-ionet-e2e',
                    config=self._config(gpus_per_vm='8'),
                )
        client.deploy.assert_not_called()

    def test_placeholder_public_key_refuses(self):
        # The template's unsubstituted placeholder is TRUTHY: a bare
        # `if not public_key` would upload a garbage key the box can
        # never answer (the Spheron/Massed trap).
        client = self._client()
        with mock.patch.object(provision, '_client',
                               return_value=client):
            with self.assertRaises(utils.IonetError):
                provision.run_instances(
                    region='us',
                    cluster_name='c',
                    cluster_name_on_cloud='urun-ionet-e2e',
                    config=self._config(
                        public_key=provision._SSH_PUBLIC_KEY_PLACEHOLDER),
                )
        client.deploy.assert_not_called()


class TestCatalogModule(unittest.TestCase):
    """The catalog module's lookups against a small in-memory CSV."""

    def setUp(self):
        import pandas as pd
        self._df = pd.DataFrame([
            ['8H100.80S.176V', 'H100', 8, 176, 1360, 41.24, 'us',
             "{'Gpus': [{'Name': 'H100', 'Count': 8, 'MemoryInfo': "
             "{'SizeInMiB': 81920}}], 'TotalGpuMemoryInMiB': 655360}",
             ''],
            ['8H100.80S.176V', 'H100', 8, 176, 1360, 41.24, 'fi',
             "{'Gpus': [{'Name': 'H100', 'Count': 8, 'MemoryInfo': "
             "{'SizeInMiB': 81920}}], 'TotalGpuMemoryInMiB': 655360}",
             ''],
        ], columns=ionet_catalog._CSV_COLUMNS)
        patcher = mock.patch.object(ionet_catalog, '_get_df',
                                     return_value=self._df)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_instance_type_exists(self):
        self.assertTrue(ionet_catalog.instance_type_exists(
            '8H100.80S.176V'))
        self.assertFalse(ionet_catalog.instance_type_exists('nope'))

    def test_validate_region_zone_is_zoneless(self):
        self.assertEqual(ionet_catalog.validate_region_zone('us', None),
                         ('us', None))
        with self.assertRaises(ValueError):
            ionet_catalog.validate_region_zone('us', 'zone-a')

    def test_get_hourly_cost_per_vm(self):
        cost = ionet_catalog.get_hourly_cost('8H100.80S.176V',
                                             use_spot=False,
                                             region='us', zone=None)
        self.assertAlmostEqual(cost, 41.24)

    def test_get_accelerators_from_instance_type(self):
        self.assertEqual(
            ionet_catalog.get_accelerators_from_instance_type(
                '8H100.80S.176V'),
            {'H100': 8})

    def test_get_vcpus_mem_from_instance_type(self):
        vcpus, mem = ionet_catalog.get_vcpus_mem_from_instance_type(
            '8H100.80S.176V')
        self.assertAlmostEqual(vcpus, 176.0)
        self.assertAlmostEqual(mem, 1360.0)


class TestCloudClass(unittest.TestCase):
    """The cloud class's identity and credential surface."""

    def test_repr_is_the_catalog_name(self):
        # repr_with_region_zone composes IoNet('8H100.80S.176V',
        # region='us') from this; the catalog module resolves through
        # _REPR.lower().
        self.assertEqual(repr(ionet_cloud.IoNet()), 'IoNet')
        self.assertEqual(ionet_cloud.IoNet._REPR, 'IoNet')
        self.assertEqual(ionet_cloud.IoNet._REPR.lower(), 'ionet')

    def test_check_credentials_without_key_reports_disabled(self):
        env = {ionet_api.API_KEY_ENV: ''}
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch('sky.clouds.ionet.os.path.exists',
                           return_value=False):
            ok, hint = ionet_cloud.IoNet.check_credentials(
                CloudCapability.COMPUTE)
        self.assertFalse(ok)
        self.assertIn(ionet_api.API_KEY_ENV, hint)

    def test_check_credentials_probes_the_api_with_the_key(self):
        # The io.net API has no token-validation endpoint: the cheapest
        # authenticated read (GET /hardware) doubles as the probe.
        client = mock.Mock()
        client.list_hardware.return_value = []
        with mock.patch.dict(os.environ,
                              {ionet_api.API_KEY_ENV: 'live-key'}), \
                mock.patch('sky.adaptors.ionet.IonetClient',
                           return_value=client) as ctor:
            ok, hint = ionet_cloud.IoNet.check_credentials(
                CloudCapability.COMPUTE)
        self.assertTrue(ok)
        self.assertIsNone(hint)
        ctor.assert_called_once_with('live-key')

    def test_check_credentials_maps_provider_error_to_hint(self):
        client = mock.Mock()
        client.list_hardware.side_effect = \
            ionet_api.IonetAuthError('HTTP 401')
        with mock.patch.dict(os.environ,
                             {ionet_api.API_KEY_ENV: 'bad-key'}), \
                mock.patch('sky.adaptors.ionet.IonetClient',
                           return_value=client):
            ok, hint = ionet_cloud.IoNet.check_credentials(
                CloudCapability.COMPUTE)
        self.assertFalse(ok)
        self.assertIn('401', hint)


if __name__ == '__main__':
    unittest.main()
