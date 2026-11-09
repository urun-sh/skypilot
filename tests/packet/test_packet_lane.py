"""Behavioral tests for the Packet.ai lane outside the client: the
catalog fetcher (sky/catalog/data_fetchers/fetch_packet.py), the
provisioner (sky/provision/packet/instance.py), and the registration
points an in-tree carry must not miss.

Every fixture shape is captured from the LIVE v2 API (measured
2026-11-09): the FLAT launch-options surface (no pools, no per-product
stock — availability lives in the category-check freeSlots), the served
POST /instances create (the v1 501 refusal is gone), the console-launched
adoption bridge, the connection endpoint's public-ip + ubuntu +
port-30356 shape, and the Pod objects whose status field is ``state``.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sky.catalog.data_fetchers import fetch_packet
from sky.provision import common as provision_common
from sky.provision.packet import instance as provision


def _product(product_id='cmkjtxmp20000asi3ermfys2t',
             name='NVIDIA RTX PRO 6000 Blackwell - Dynamic',
             cents=66,
             vram=96,
             service_id='srv_rtxpro6000',
             billing_type='hourly',
             gpu_count=1):
    # v2 (measured 2026-11-09): products carry NO stock and NO region
    # list (v1 totalAvailableGpus/poolIds/regions are gone) — the
    # WHOLE-BOX pricePerHourCents, the serviceId the category-check
    # answers for, and the description prose the row parser reads.
    return {
        'id': product_id,
        'name': name,
        'description': '96GB ram',
        'billingType': billing_type,
        'pricePerHourCents': cents,
        'serviceId': service_id,
        'featured': True,
        'vramGb': vram,
        'gpuCount': gpu_count,
    }


def _region(region_id=67, name='us-west-2'):
    # v2 (measured 2026-11-09): region rows are {id: int, name: str}.
    return {
        'id': region_id,
        'name': name,
    }


def _category_check(service_id='srv_rtxpro6000',
                    region_id=67,
                    free_slots=3,
                    check_failed=False):
    # v2 (measured 2026-11-09): GET /instances/category-check answers
    # {categoryId, compatibleServiceIds, serviceRegions, freeSlots,
    # maintenance, checkFailed}; freeSlots values are free POD slots
    # keyed service id -> region id.
    return {
        'categoryId': 'cat_1',
        'compatibleServiceIds': [service_id],
        'serviceRegions': {
            service_id: [region_id]
        },
        'freeSlots': {
            service_id: {
                str(region_id): free_slots
            }
        },
        'maintenance': {},
        'checkFailed': check_failed,
    }


def _checks(products, region_id=67, free_slots=3, check_failed=False):
    """{serviceId: category-check answer} for every product."""
    return {
        str(p.get('serviceId') or ''): _category_check(
            str(p.get('serviceId') or ''),
            region_id=region_id,
            free_slots=free_slots,
            check_failed=check_failed) for p in products
    }


def _launch_options(products=(), regions=()):
    # v2 (measured 2026-11-09): the launch-options answer is FLAT — no
    # pools key, no images/storageBlocks (the v1 arrays are gone).
    return {
        'can_use_premium': True,
        'categories': [],
        'existingSharedVolumes': [],
        'products': list(products),
        'regions': list(regions),
        'sshKeys': [],
        'storagePricePerGBHourCents': 2,
        'teamId': 'team_1',
        'walletBalanceCents': 50000,
    }


class TestFetcherTokens(unittest.TestCase):

    def test_normalise_accelerator_maps_the_target_family(self):
        self.assertEqual(
            fetch_packet.normalise_accelerator(
                'NVIDIA RTX PRO 6000 Blackwell - Dynamic'), 'RTXPRO6000')
        self.assertEqual(
            fetch_packet.normalise_accelerator('1 x NVIDIA B200 Dynamic'),
            'B200')

    def test_normalise_accelerator_rejects_empty(self):
        self.assertEqual(fetch_packet.normalise_accelerator(''), '')
        self.assertEqual(fetch_packet.normalise_accelerator(None), '')

    def test_parse_product_gpu_count(self):
        self.assertEqual(
            fetch_packet.parse_product_gpu_count('2 x NVIDIA RTX 4090 '
                                                 'Dedicated'), 2)
        self.assertEqual(
            fetch_packet.parse_product_gpu_count(
                'NVIDIA RTX PRO 6000 Blackwell - Dynamic'), 1)

    def test_token_round_trip(self):
        token = fetch_packet.instance_type_token('RTXPRO6000', 1)
        self.assertEqual(token, 'rtxpro6000:1')
        accelerator, count = fetch_packet.parse_instance_type_token(token)
        self.assertEqual((accelerator, count), ('rtxpro6000', 1))

    def test_parse_garbage_raises(self):
        for garbage in ('', 'rtxpro6000', 'rtxpro6000:x', 'a:b:1', ':1'):
            with self.assertRaises(fetch_packet.PacketCatalogError):
                fetch_packet.parse_instance_type_token(garbage)


class TestFetcherRows(unittest.TestCase):

    def test_in_stock_product_emits_token_region_wholebox_price(self):
        # The v2 LIVE shape (measured 2026-11-09): the RTX PRO 6000
        # product at 66 WHOLE-BOX cents/hr, 96GB, category-check free
        # slots in us-west-2 — the target the lane rents. Whole-box
        # price for count 1 = 0.66.
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product])
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row[0], 'rtxpro6000:1')
        self.assertEqual(row[1], 'RTXPRO6000')
        self.assertEqual(row[2], 1)
        self.assertEqual(row[5], 0.66)
        self.assertEqual(row[6], 'us-west-2')
        gpu_info = json.loads(str(row[7]).replace("'", '"'))
        # The GpuInfo column carries the 'Gpus'-wrapped shape
        # sky/catalog/common's DeviceMemoryGiB pass reads.
        self.assertEqual(gpu_info['Gpus'][0]['Name'], 'RTXPRO6000')
        self.assertEqual(gpu_info['Gpus'][0]['Count'], 1)

    def test_zero_stock_product_is_not_rentable(self):
        # v2 (measured 2026-11-09): v1's totalAvailableGpus is GONE —
        # rentability is the category-check's freeSlots. A product whose
        # check reports 0 free slots must not reach the catalog (a
        # rentable quote for a box that cannot launch).
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product], free_slots=0)
        self.assertEqual(list(fetch_packet.iter_rows(options, checks)), [])

    def test_monthly_unpriced_product_is_not_rentable(self):
        # Monthly plans never touch the wallet and carry no hourly price
        # (pricePerHourCents 0 measured) — quoting them as free would
        # be a lie.
        product = _product(cents=0, billing_type='monthly')
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product])
        self.assertEqual(list(fetch_packet.iter_rows(options, checks)), [])

    def test_multi_gpu_token_prices_the_whole_box(self):
        # v2 (measured 2026-11-09): pricePerHourCents is the WHOLE-BOX
        # price — a 2-GPU product at 78 cents/hr quotes 0.78/hr for the
        # rtx4090:2 token (the count is inside the box price, nothing is
        # multiplied at row time).
        product = _product(name='2 x NVIDIA RTX 4090 Dedicated',
                           cents=78,
                           service_id='srv_4090',
                           gpu_count=2)
        options = _launch_options(products=[product],
                                  regions=[_region(region_id=78, name='Texas')])
        checks = _checks([product], region_id=78)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 'rtx4090:2')
        self.assertEqual(rows[0][2], 2)
        self.assertEqual(rows[0][5], 0.78)  # whole box, 2 GPUs inside
        self.assertEqual(rows[0][6], 'Texas')

    def test_cheapest_per_token_region_wins(self):
        # The same (accelerator, count, region) at two prices: the row
        # is a rentable quote, so the cheapest wins (the provisioner
        # re-resolves against the live surface at launch anyway).
        products = [
            _product(product_id='p_cheap', cents=66),
            _product(product_id='p_dear', cents=99),
        ]
        options = _launch_options(products=products, regions=[_region()])
        checks = _checks(products)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], 0.66)

    def test_unmappable_family_is_skipped(self):
        # A family fragment the accelerator map does not know is skipped
        # (audit-visible via refusal), never a garbage row.
        product = _product(name='Mystery GPU X')
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product])
        self.assertEqual(list(fetch_packet.iter_rows(options, checks)), [])

    def test_check_failed_answer_refuses(self):
        # v2 (measured 2026-11-09): checkFailed=true means the
        # availability answer may be INCOMPLETE — the fetch refuses
        # rather than write a guessed-around partial catalog.
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product], check_failed=True)
        with self.assertRaises(fetch_packet.PacketCatalogError) as ctx:
            list(fetch_packet.iter_rows(options, checks))
        self.assertIn('checkFailed', str(ctx.exception))

    def test_blank_description_quotes_from_ramMb_field(self):
        # v2 (measured 2026-11-09): the live 5090 Dynamic - Hourly
        # product carries vcpuCount=6, ramMb=49152 and a BLANK
        # description — the structured fields quote what the prose
        # cannot. Field-first, prose fallback.
        product = _product(name='1 x RTX 5090 Dynamic - Hourly',
                           cents=36,
                           service_id='srv_5090',
                           vram=32)
        product['description'] = ''
        product['ramMb'] = 49152
        product['vcpuCount'] = 6
        options = _launch_options(
            products=[product],
            regions=[_region(region_id=100, name='Netherlands')])
        checks = _checks([product], region_id=100)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 'rtx5090:1')
        self.assertEqual(rows[0][3], 6.0)  # vcpuCount field
        self.assertEqual(rows[0][4], 48.0)  # ramMb 49152 / 1024
        self.assertEqual(rows[0][5], 0.36)
        self.assertEqual(rows[0][6], 'Netherlands')

    def test_description_prose_still_quotes_when_fields_absent(self):
        # The 4090 Dedicated carries ramMb=null but its prose carries
        # '64GB ram · 16 vCPU' (measured): the fallback path.
        product = _product(name='1 x NVIDIA RTX 4090 Dedicated',
                           cents=39,
                           service_id='srv_4090',
                           vram=24)
        product['description'] = '64GB ram · 16 vCPU · 250GB Local Storage'
        product['ramMb'] = None
        product['vcpuCount'] = None
        options = _launch_options(products=[product],
                                  regions=[_region(region_id=78, name='Texas')])
        checks = _checks([product], region_id=78)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], 64.0)
        self.assertEqual(rows[0][3], 16.0)


class TestVerifyZeroStock(unittest.TestCase):

    def test_rows_present_passes(self):
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product])
        rows = list(fetch_packet.iter_rows(options, checks))
        fetch_packet.verify_zero_stock(options, checks, rows)  # no raise

    def test_honest_empty_passes(self):
        # Every product's check at 0 free slots: the header-only catalog
        # is the honest write (the Latitude semantics).
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product], free_slots=0)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(rows, [])
        fetch_packet.verify_zero_stock(options, checks, rows)  # no raise

    def test_contradiction_refuses(self):
        # A service reporting a free slot in a region the catalog names
        # while no row will be written is a filter bug or API drift —
        # REFUSE, never write "no capacity".
        product = _product(name='NVIDIA Something Unmappable')
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product], free_slots=5)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(rows, [])
        with self.assertRaises(fetch_packet.PacketCatalogError) as ctx:
            fetch_packet.verify_zero_stock(options, checks, rows)
        self.assertIn('refusing', str(ctx.exception))

    def test_stocked_product_with_no_ram_anywhere_refuses(self):
        # Measured live 2026-11-09: a stocked product whose description
        # is blank AND ramMb is null cannot be quoted (SkyPilot's launch
        # validation rejects memory <= 0) — the zero-row write REFUSES
        # rather than answer "no capacity" while real stock exists.
        product = _product()
        product['description'] = ''
        product['ramMb'] = None
        options = _launch_options(products=[product], regions=[_region()])
        checks = _checks([product], free_slots=8)
        rows = list(fetch_packet.iter_rows(options, checks))
        self.assertEqual(rows, [])
        with self.assertRaises(fetch_packet.PacketCatalogError) as ctx:
            fetch_packet.verify_zero_stock(options, checks, rows)
        self.assertIn('refusing', str(ctx.exception))


class _StubClient:
    """Records calls; answers from canned state."""

    def __init__(self, instances=(), launch_options=None, category_checks=None):
        self.instances = {str(d['id']): d for d in instances}
        product = _product()
        self.launch_options = launch_options or _launch_options(
            products=[product], regions=[_region()])
        # v2 (measured 2026-11-09): rentability answers live in the
        # category-checks, keyed by service id.
        self.category_checks = category_checks or _checks([product])
        self.created = []
        self.stopped = []
        self.deleted = []
        self.ensure_key_calls = []

    # -- catalog -----------------------------------------------------

    def get_launch_options(self):
        return self.launch_options

    def category_check(self, service_id, *, category_id=None):
        # v2 (measured 2026-11-09): GET /instances/category-check.
        del category_id
        return self.category_checks.get(
            str(service_id), _category_check(str(service_id), free_slots=0))

    # -- ssh keys ------------------------------------------------------

    def list_ssh_keys(self):
        return []

    def create_ssh_key(self, name, public_key):
        self.ensure_key_calls.append((name, public_key))
        return {'id': 'key_new'}

    def ensure_ssh_key(self, name, public_key):
        self.ensure_key_calls.append((name, public_key))
        return 'key_1'

    # -- instances -----------------------------------------------------

    def list_instances(self, status=None):
        del status
        return list(self.instances.values())

    def get_instance(self, instance_id):
        # v2 (measured 2026-11-09): the {id} route answers the Pod FLAT
        # (the v1 data.instance wrapper is gone).
        if str(instance_id) not in self.instances:
            raise provision.utils.PacketNotFoundError('gone')
        return dict(self.instances[str(instance_id)])

    def get_connection(self, instance_id):
        if str(instance_id) not in self.instances:
            raise provision.utils.PacketNotFoundError('gone')
        return {
            'ip': '50.35.188.78',
            'port': 30356,
            'username': 'ubuntu',
            'password': 'secret-password',
            'ssh_command': 'ssh -p 30356 ubuntu@50.35.188.78',
        }

    def create_instance(self,
                        *,
                        product_id,
                        region_id,
                        name=None,
                        ssh_key_ids=None):
        # v2 (measured 2026-11-09): the create body is {product_id,
        # region_id, name?, ssh_key_ids?} (v1's pool_id/vgpus kwargs are
        # gone) and the 201 answers {id: ...}.
        self.created.append({
            'product_id': product_id,
            'region_id': region_id,
            'name': name,
            'ssh_key_ids': list(ssh_key_ids or []),
        })
        new = {
            'id': 'i-new',
            'name': name,
            'state': 'running',
        }
        self.instances['i-new'] = new
        return {'id': 'i-new'}

    def delete_instance(self, instance_id):
        self.deleted.append(str(instance_id))
        self.instances.pop(str(instance_id), None)

    def stop_instance(self, instance_id):
        self.stopped.append(str(instance_id))

    def wait_until_running(self, instance_id, **kwargs):
        # v2: the payload is the FLAT pod; the client's poll reads
        # ``state``.
        del kwargs
        payload = self.get_instance(instance_id)
        status = provision.utils.instance_status(payload)
        if status != 'running':
            raise provision.utils.PacketError(
                f'stub saw {status!r} while waiting for running')
        return payload


def _config(node_config=None):
    return provision_common.ProvisionConfig(
        provider_config={},
        authentication_config={},
        docker_config={},
        node_config=node_config or {},
        count=1,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


def _launchable_config():
    return _config({
        'InstanceType': 'rtxpro6000:1',
        'PublicKey': 'ssh-rsa AAAAcanned test@host',
    })


def _instance_row(instance_id, state, name='rtxpro6000blackwell-1x-dynamic'):
    # v2 (measured 2026-11-09): the Pod names its status field ``state``
    # (v1 ``status``); the enum is starting | running | stopping |
    # stopped | failed | terminating | terminated | unknown.
    return {'id': instance_id, 'name': name, 'state': state}


class _StateIsolated(unittest.TestCase):
    """Runs each test with the provisioner's identity store in a tmp dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._state_path = os.path.join(self._tmp.name, 'clusters.json')
        patcher = mock.patch.object(provision, '_STATE_PATH', self._state_path)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestProvisioner(_StateIsolated):

    def _run(self, stub, config=None, region='us-west-2'):
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            return provision.run_instances(region, 'cluster',
                                           'sky-test-cluster', config or
                                           _launchable_config())

    def test_adopts_the_mapped_instance(self):
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running', name='sky-test-cluster'),
        ])
        # Seed the mapping (a prior run recorded it).
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-1',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        record = self._run(stub)
        self.assertEqual(record.head_instance_id, 'i-1')
        self.assertEqual(record.created_instance_ids, [])
        self.assertEqual(stub.created, [])

    def test_adopts_console_launched_box_by_name(self):
        # THE CONSOLE BRIDGE: the operator launched the lane's box FROM
        # THE CONSOLE with the cluster name. No mapping exists — the
        # adoption is by EXACT NAME. (v2 serves API-key creates now,
        # measured 2026-11-09, but a console-launched box still adopts
        # here rather than minting a second box.)
        stub = _StubClient(instances=[
            _instance_row('i-console', 'running', name='sky-test-cluster'),
        ])
        record = self._run(stub)
        self.assertEqual(record.head_instance_id, 'i-console')
        self.assertEqual(record.created_instance_ids, [])
        state = provision._load_state()
        self.assertEqual(state['sky-test-cluster']['instance_id'], 'i-console')
        # The measured SSH user is recorded from the connection endpoint.
        self.assertEqual(state['sky-test-cluster']['ssh_user'], 'ubuntu')

    def test_refuses_foreign_live_box(self):
        # An unmapped live box with a foreign name is a guess — refuse.
        stub = _StubClient(instances=[
            _instance_row('i-foreign', 'running', name='someone-elses-box'),
        ])
        with self.assertRaises(provision.utils.PacketError) as ctx:
            self._run(stub)
        self.assertIn('refusing to guess', str(ctx.exception))
        self.assertIn('console', str(ctx.exception))

    def test_create_path_provisions_with_live_resolved_ids(self):
        # v2 (measured 2026-11-09): the v1 501 create refusal is GONE —
        # POST /instances SERVES creates for API keys. No mapping, no
        # name-match, no foreign boxes: the create path provisions, and
        # the recorded body proves the launch-time re-resolution RAN —
        # the LIVE product id and the NUMERIC region id (67), the
        # cluster name, and the ensured key's id (never a guess, never a
        # frozen id from catalog time).
        stub = _StubClient(instances=[])
        record = self._run(stub)
        self.assertEqual(record.head_instance_id, 'i-new')
        self.assertEqual(record.created_instance_ids, ['i-new'])
        self.assertEqual(stub.created, [{
            'product_id': 'cmkjtxmp20000asi3ermfys2t',
            'region_id': 67,
            'name': 'sky-test-cluster',
            'ssh_key_ids': ['key_1'],
        }])
        self.assertEqual(stub.ensure_key_calls,
                         [('skypilot', 'ssh-rsa AAAAcanned test@host')])

    def test_missing_instance_type_token_refuses(self):
        stub = _StubClient(instances=[])
        with self.assertRaises(provision.utils.PacketError) as ctx:
            self._run(stub,
                      config=_config({
                          'InstanceType': '',
                          'PublicKey': 'ssh-rsa AAAAcanned test@host',
                      }))
        self.assertIn('InstanceType', str(ctx.exception))

    def test_placeholder_public_key_refused(self):
        # configure_ssh_info substitutes the real key into the template's
        # placeholder; if it did NOT run, the literal is truthy garbage —
        # refuse it (the Spheron trap).
        stub = _StubClient(instances=[])
        with self.assertRaises(provision.utils.PacketError) as ctx:
            self._run(stub,
                      config=_config({
                          'InstanceType': 'rtxpro6000:1',
                          'PublicKey': provision._SSH_PUBLIC_KEY_PLACEHOLDER,
                      }))
        self.assertIn('placeholder', str(ctx.exception))

    def test_failed_state_box_refuses_with_report(self):
        # v2 (measured 2026-11-09): v1's ``error`` is GONE — a mapped
        # box in ``failed`` (or terminating/terminated) must not be
        # silently retried.
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-err',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-err', 'failed', name='sky-test-cluster'),
        ])
        with self.assertRaises(provision.utils.PacketError) as ctx:
            self._run(stub)
        self.assertIn('failed', str(ctx.exception))
        self.assertIn('not auto-retried', str(ctx.exception))

    def test_corrupt_state_file_raises(self):
        with open(provision._STATE_PATH, 'w', encoding='utf-8') as handle:
            handle.write('not json')
        with self.assertRaises(provision.utils.PacketError) as ctx:
            provision._load_state()
        self.assertIn('corrupt', str(ctx.exception))

    def test_terminate_deletes_and_forgets(self):
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-1',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            provision.terminate_instances('sky-test-cluster')
        self.assertEqual(stub.deleted, ['i-1'])
        self.assertEqual(provision._load_state().get('sky-test-cluster'), None)

    def test_terminate_refuses_empty_identity(self):
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            with self.assertRaises(provision.utils.PacketError) as ctx:
                provision.terminate_instances('')
        self.assertIn('empty', str(ctx.exception))

    def test_terminate_unmapped_with_live_boxes_refuses(self):
        stub = _StubClient(instances=[
            _instance_row('i-foreign', 'running', name='not-ours'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            with self.assertRaises(provision.utils.PacketError) as ctx:
                provision.terminate_instances('sky-test-cluster')
        self.assertIn('refusing to guess', str(ctx.exception))
        self.assertEqual(stub.deleted, [])

    def test_stop_is_a_real_stop_not_destroy(self):
        # v2 (measured 2026-11-09): a REAL stop persists the box+disk
        # (start resumes) — but it does NOT pause the bill: a stopped pod
        # KEEPS its GPU and bills at the FULL running rate, so only
        # terminating ends the charges. Stop is still not a destroy.
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-1',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            provision.stop_instances('sky-test-cluster')
        self.assertEqual(stub.stopped, ['i-1'])
        self.assertEqual(stub.deleted, [])  # stop is NOT terminate
        self.assertIn('i-1', stub.instances)  # the box+disk persist

    def test_get_cluster_info_reads_public_connection(self):
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-1',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            info = provision.get_cluster_info('us-west-2', 'sky-test-cluster')
        self.assertEqual(info.head_instance_id, 'i-1')
        self.assertEqual(info.ssh_user, 'ubuntu')
        instance_info = info.instances['i-1'][0]
        self.assertEqual(instance_info.external_ip, '50.35.188.78')
        self.assertEqual(instance_info.ssh_port, 30356)

    def test_query_instances_maps_status(self):
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-1',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            queried = provision.query_instances('cluster', 'sky-test-cluster')
        self.assertIn('i-1', queried)

    def test_query_instances_404_forgets(self):
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-gone',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[])  # i-gone is not on the account
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            queried = provision.query_instances('cluster', 'sky-test-cluster')
        self.assertEqual(queried, {})
        self.assertEqual(provision._load_state().get('sky-test-cluster'), None)


class TestLaunchResolution(_StateIsolated):
    """The launch-time product/region re-resolution (v2: pools GONE)."""

    def _run_create(self, stub):
        return provision.run_instances('us-west-2', 'cluster',
                                       'sky-test-cluster', _launchable_config())

    def test_resolves_product_region_from_live_options(self):
        # v2 (measured 2026-11-09): pools are GONE — the create body
        # carries the LIVE product id and the NUMERIC region id from the
        # re-resolution (the ids are ephemeral; a frozen one from
        # catalog time would be silent rot), and the category-check must
        # answer free slots in the region before the create runs.
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        stub = _StubClient(
            launch_options=options,
            category_checks=_checks([product]),
            instances=[],
        )
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            record = self._run_create(stub)
        self.assertEqual(record.created_instance_ids, ['i-new'])
        self.assertEqual(stub.created, [{
            'product_id': 'cmkjtxmp20000asi3ermfys2t',
            'region_id': 67,
            'name': 'sky-test-cluster',
            'ssh_key_ids': ['key_1'],
        }])

    def test_unknown_region_fails_closed(self):
        product = _product()
        options = _launch_options(
            products=[product],
            regions=[],  # no regions served at all
        )
        stub = _StubClient(launch_options=options,
                           category_checks=_checks([product]),
                           instances=[])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            with self.assertRaises(provision.utils.PacketError) as ctx:
                self._run_create(stub)
        self.assertIn('fail closed', str(ctx.exception))

    def test_no_in_stock_product_fails_closed(self):
        # v2 (measured 2026-11-09): stock is the category-check's
        # freeSlots — 0 free slots behind the catalog row means the
        # offer went out of stock; fail closed rather than rent a
        # different shape.
        product = _product()
        options = _launch_options(products=[product], regions=[_region()])
        stub = _StubClient(launch_options=options,
                           category_checks=_checks([product], free_slots=0),
                           instances=[])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            with self.assertRaises(provision.utils.PacketError) as ctx:
                self._run_create(stub)
        self.assertIn('out of stock', str(ctx.exception))


class TestCloudWiring(unittest.TestCase):
    """The registration points an out-of-tree/in-tree cloud must not miss —
    each of these was a real lane-killing bug on a previous carry
    (Shadeform's _REPR, Spheron's auth branch, Vast's template map)."""

    def test_repr_is_the_catalog_name(self):
        from sky.clouds import Packet
        self.assertEqual(Packet._REPR, 'Packet')

    def test_all_clouds_contains_packet(self):
        from sky.skylet import constants
        self.assertIn('packet', constants.ALL_CLOUDS)

    def test_cluster_config_template_registered(self):
        from sky.backends import cloud_vm_ray_backend
        from sky.clouds import Packet
        self.assertEqual(
            cloud_vm_ray_backend._get_cluster_config_template(Packet()),
            'packet-ray.yml.j2',
        )

    def test_auth_dispatch_covers_packet(self):
        # _add_auth_to_cluster_config is an isinstance chain ending in
        # `assert False, cloud`; Packet must take the generic
        # configure_ssh_info branch (its key is registered per-launch).
        import ast
        import inspect

        from sky.backends import backend_utils
        source = inspect.getsource(backend_utils._add_auth_to_cluster_config)
        tree = ast.parse(source)
        names = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn('Packet', names,
                      'Packet missing from the auth dispatch chain')

    def test_zones_provision_loop_yields_per_region(self):
        from sky import clouds as clouds_lib
        from sky.clouds import Packet
        with mock.patch.object(
                Packet,
                'regions_with_offering',
                classmethod(lambda cls, *a, **k: [
                    clouds_lib.Region('us-west-2'),
                    clouds_lib.Region('us-west-1'),
                ]),
        ):
            yields = list(
                Packet.zones_provision_loop(region='us-west-2',
                                            num_nodes=1,
                                            instance_type='rtxpro6000:1'))
        # One yield PER region with an offering — a single yield for "any"
        # turned a first-region failure into the end of provisioning on
        # the Spheron lane.
        self.assertEqual(yields, [None, None])

    def test_ssh_key_file_mount_is_packet_scoped(self):
        from sky.clouds import Packet
        mounts = Packet().get_credential_file_mounts()
        self.assertEqual(mounts,
                         {'~/.packet/credentials': '~/.packet/credentials'})

    def test_dependency_extra_registered(self):
        # The 'packet' extra must exist so `pip install skypilot[packet]`
        # works and the wheel build stays coherent.
        from sky.setup_files import dependencies
        self.assertIn('packet', dependencies.cloud_dependencies)
        self.assertEqual(dependencies.cloud_dependencies['packet'], [])

    def test_provisioner_package_registered(self):
        # The dispatch wrapper resolves `sky.provision.<cloud>` by name;
        # a missing import is a dead lane at first launch.
        from sky.provision import packet
        self.assertTrue(hasattr(packet, 'run_instances'))


if __name__ == '__main__':
    unittest.main()
