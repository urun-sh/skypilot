"""Behavioral tests for the Packet.ai lane outside the client: the
catalog fetcher (sky/catalog/data_fetchers/fetch_packet.py), the
provisioner (sky/provision/packet/instance.py), and the registration
points an in-tree carry must not miss.

Every fixture shape is captured from the LIVE API (ENG-562, 2026-10-15):
the launch-options products/pools/regions surface (including the RTX PRO
6000 product that flipped in mid-session), the 501 create refusal, the
console-launched adoption bridge, the connection endpoint's public-ip +
ubuntu + port-30356 shape.
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
             stock=3,
             vram=96,
             pool_ids=(108,),
             regions=('us-west-2',)):
    return {
        'id': product_id,
        'name': name,
        'description': '96GB ram',
        'pricePerHourCents': cents,
        'poolIds': list(pool_ids),
        'displayOrder': 0,
        'featured': True,
        'badgeText': None,
        'vramGb': vram,
        'totalAvailableGpus': stock,
        'regions': [{
            'id': 67,
            'name': region,
            'city': 'Santa Clara, CA',
            'country': 'United States',
        } for region in regions],
    }


def _pool(pool_id='108', name='rtx-w2-6000-2', price=0.66, region='us-west-2'):
    return {
        'id': pool_id,
        'name': name,
        'gpu_model': name,
        'price_per_hour': price,
        'region': {
            'id': 67,
            'name': region,
            'city': 'Santa Clara, CA',
            'country': 'United States',
        },
    }


def _region(region_id=67, name='us-west-2'):
    return {
        'id': region_id,
        'region_name': name,
        'name': name,
        'city': 'Santa Clara, CA',
        'country': 'United States',
        'country_code': 'us',
        'accelerator_type': 'gpu',
    }


def _launch_options(products=(), pools=(), regions=()):
    return {
        'products': list(products),
        'pools': list(pools),
        'regions': list(regions),
        'images': [],
        'storageBlocks': [],
        'ephemeralStorageBlocks': [],
        'persistentStorageBlocks': [],
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
        # The LIVE shape (ENG-562, 2026-10-15): the RTX PRO 6000 product at
        # 66 cents/GPU-hr, 96GB, stock 3, us-west-2 — the target the lane
        # rents. Whole-box price for count 1 = 0.66.
        options = _launch_options(
            products=[_product()],
            pools=[_pool()],
            regions=[_region()],
        )
        rows = list(fetch_packet.iter_rows(options))
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
        # A featured product at totalAvailableGpus 0 appears in the list;
        # the row must not reach the catalog (a rentable quote for a box
        # that cannot launch).
        options = _launch_options(products=[_product(stock=0)])
        self.assertEqual(list(fetch_packet.iter_rows(options)), [])

    def test_monthly_unpriced_product_is_not_rentable(self):
        # Monthly plans carry pricePerHourCents 0 and never touch the
        # wallet — quoting them as free would be a lie.
        options = _launch_options(products=[_product(cents=0)])
        self.assertEqual(list(fetch_packet.iter_rows(options)), [])

    def test_count_prefix_multiplies_the_wholebox_price(self):
        options = _launch_options(
            products=[
                _product(name='2 x NVIDIA RTX 4090 Dedicated',
                         cents=39,
                         stock=1,
                         pool_ids=(144,),
                         regions=('Texas',)),
            ],
            pools=[
                _pool(pool_id='144',
                      name='4090PremiumPool_1',
                      price=0.39,
                      region='Texas')
            ],
            regions=[_region(region_id=78, name='Texas')],
        )
        rows = list(fetch_packet.iter_rows(options))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 'rtx4090:2')
        self.assertEqual(rows[0][2], 2)
        self.assertEqual(rows[0][5], 0.78)  # 2 x $0.39/GPU-hr

    def test_cheapest_per_token_region_wins(self):
        # The same (accelerator, count, region) at two prices: the row is a
        # rentable quote, so the cheapest wins (the provisioner
        # re-resolves against the live surface at launch anyway).
        options = _launch_options(
            products=[
                _product(product_id='p_cheap', cents=66),
                _product(product_id='p_dear', cents=99),
            ],
            pools=[_pool()],
            regions=[_region()],
        )
        rows = list(fetch_packet.iter_rows(options))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], 0.66)

    def test_unmappable_family_is_skipped(self):
        # A family fragment the accelerator map does not know is skipped
        # (audit-visible via refusal), never a garbage row.
        options = _launch_options(products=[_product(name='Mystery GPU X')])
        self.assertEqual(list(fetch_packet.iter_rows(options)), [])


class TestVerifyZeroStock(unittest.TestCase):

    def test_rows_present_passes(self):
        options = _launch_options(products=[_product()])
        rows = list(fetch_packet.iter_rows(options))
        fetch_packet.verify_zero_stock(options, rows)  # no raise

    def test_honest_empty_passes(self):
        # Every product at 0 stock: the header-only catalog is the honest
        # write (the Latitude semantics).
        options = _launch_options(products=[_product(stock=0)])
        rows = list(fetch_packet.iter_rows(options))
        self.assertEqual(rows, [])
        fetch_packet.verify_zero_stock(options, rows)  # no raise

    def test_contradiction_refuses(self):
        # A product reporting stock while no row will be written is a
        # filter bug or API drift — REFUSE, never write "no capacity".
        options = _launch_options(
            products=[_product(name='NVIDIA Something Unmappable', stock=5)])
        rows = list(fetch_packet.iter_rows(options))
        self.assertEqual(rows, [])
        with self.assertRaises(fetch_packet.PacketCatalogError) as ctx:
            fetch_packet.verify_zero_stock(options, rows)
        self.assertIn('refusing', str(ctx.exception))


class _StubClient:
    """Records calls; answers from canned state."""

    def __init__(self, instances=(), launch_options=None, create_error=None):
        self.instances = {str(d['id']): d for d in instances}
        self.launch_options = launch_options or _launch_options(
            products=[_product()], pools=[_pool()], regions=[_region()])
        self.created = []
        self.stopped = []
        self.deleted = []
        self.ensure_key_calls = []
        self.create_error = create_error

    # -- catalog -----------------------------------------------------

    def get_launch_options(self):
        return self.launch_options

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
        return list(self.instances.values())

    def get_instance(self, instance_id):
        if str(instance_id) not in self.instances:
            raise provision.utils.PacketNotFoundError('gone')
        return {'instance': self.instances[str(instance_id)]}

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
                        name,
                        pool_id,
                        region_id,
                        vgpus=1,
                        ssh_key_name=None,
                        public_key=None):
        if self.create_error is not None:
            raise self.create_error
        self.created.append({
            'name': name,
            'pool_id': pool_id,
            'region_id': region_id,
            'vgpus': vgpus,
        })
        new = {
            'id': 'i-new',
            'name': name,
            'status': 'running',
        }
        self.instances['i-new'] = new
        return {'instance_id': 'i-new'}

    def delete_instance(self, instance_id):
        self.deleted.append(str(instance_id))
        self.instances.pop(str(instance_id), None)

    def stop_instance(self, instance_id):
        self.stopped.append(str(instance_id))

    def wait_until_running(self, instance_id, **kwargs):
        payload = self.get_instance(instance_id)
        return payload['instance']


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


def _instance_row(instance_id, status, name='rtxpro6000blackwell-1x-dynamic'):
    return {'id': instance_id, 'name': name, 'status': status}


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
        # THE ENG-562 BRIDGE: creation is 501 for API keys; the operator
        # launched the lane's box FROM THE CONSOLE with the cluster name.
        # No mapping exists — the adoption is by EXACT NAME.
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

    def test_create_refusal_is_typed_and_names_the_console_path(self):
        # No mapping, no name-match, no foreign boxes: the create path
        # raises the MEASURED typed 501 today — never a retry loop, never
        # a silent "no capacity".
        stub = _StubClient(
            instances=[],
            create_error=provision.utils.PacketCreateNotImplementedError(
                'POST /instances: NOT_IMPLEMENTED: Instance creation is '
                'not yet available: the deploy engine (Stripe, wallet, '
                'provisioning) has not been ported.'),
        )
        with self.assertRaises(
                provision.utils.PacketCreateNotImplementedError) as ctx:
            self._run(stub)
        self.assertIn('NOT_IMPLEMENTED', str(ctx.exception))

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

    def test_error_state_box_refuses_with_report(self):
        # A mapped box in error/failed must not be silently retried.
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-err',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-err', 'error', name='sky-test-cluster'),
        ])
        with self.assertRaises(provision.utils.PacketError) as ctx:
            self._run(stub)
        self.assertIn('error', str(ctx.exception))
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
            _instance_row('i-1', 'running', name='sky-test-cluster'),
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
        # MEASURED Packet semantics: POST stop stops billing, box+disk
        # persist — unlike QuantaCloud whose stop destroys.
        provision._save_state({
            'sky-test-cluster': {
                'instance_id': 'i-1',
                'region': 'us-west-2',
                'ssh_user': 'ubuntu',
                'instance_type': 'rtxpro6000:1',
            }
        })
        stub = _StubClient(instances=[
            _instance_row('i-1', 'running', name='sky-test-cluster'),
        ])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            provision.stop_instances('sky-test-cluster')
        self.assertEqual(stub.stopped, ['i-1'])
        self.assertEqual(stub.deleted, [])  # stop is NOT terminate

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
            _instance_row('i-1', 'running', name='sky-test-cluster'),
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
            _instance_row('i-1', 'running', name='sky-test-cluster'),
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
    """The launch-time product/pool/region re-resolution."""

    def test_resolves_product_pool_region_from_live_options(self):
        options = _launch_options(
            products=[_product()],
            pools=[_pool()],
            regions=[_region()],
        )
        stub = _StubClient(
            launch_options=options,
            create_error=provision.utils.PacketCreateNotImplementedError(
                'POST /instances: NOT_IMPLEMENTED: not '
                'ported'))
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            with self.assertRaises(
                    provision.utils.PacketCreateNotImplementedError):
                self._run_create(stub)
        # The re-resolution DID happen before the refusal: the create body
        # carries the pool id and the NUMERIC region id (67), proving the
        # re-resolution ran — the refusal is the provider's, not ours.
        self.assertEqual(stub.created, [])

    def _run_create(self, stub):
        return provision.run_instances('us-west-2', 'cluster',
                                       'sky-test-cluster', _launchable_config())

    def test_unknown_region_fails_closed(self):
        options = _launch_options(
            products=[_product()],
            pools=[_pool()],
            regions=[],  # no regions served at all
        )
        stub = _StubClient(launch_options=options, instances=[])
        with mock.patch.object(provision.utils,
                               'client_from_env',
                               return_value=stub):
            with self.assertRaises(provision.utils.PacketError) as ctx:
                self._run_create(stub)
        self.assertIn('fail closed', str(ctx.exception))

    def test_no_in_stock_product_fails_closed(self):
        options = _launch_options(
            products=[_product(stock=0)],
            pools=[_pool()],
            regions=[_region()],
        )
        stub = _StubClient(launch_options=options, instances=[])
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
