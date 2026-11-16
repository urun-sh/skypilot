"""Lane tests for the Verda fetcher + cloud + provisioner wiring.

Fixture-based (derived from the OpenAPI spec's examples at
/tmp/verda-openapi.json, reviewed 2026-11-16): NO live calls anywhere —
credentials do not exist yet.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sky import clouds
from sky.adaptors import verda as verda_api
from sky.catalog.data_fetchers import fetch_verda
from sky.clouds import verda as verda_cloud
from sky.provision.verda import instance as verda_instance
from sky.provision.verda import utils

# -- fixtures (OpenAPI examples) -------------------------------------------


def _type_row(code='1H100.80S.22V',
              model='H100',
              currency='usd',
              location_prices=(('FIN-02', '2.90', '0.29'),),
              deprecated_price='999.99'):
    return {
        'id': 'c01dd00d-0000-4972-ae4e-d429115d055b',
        'instance_type': code,
        'model': model,
        'name': 'H100 SXM5 80GB',
        'gpu': {
            'description': '1x H100 SXM5 80GB',
            'number_of_gpus': 1
        },
        'cpu': {
            'description': '22 CPU',
            'number_of_cores': 22,
            'model': 'AMD Genoa'
        },
        'memory': {
            'description': '187GB RAM',
            'size_in_gigabytes': 187
        },
        'gpu_memory': {
            'description': '80GB GPU RAM',
            'size_in_gigabytes': 80
        },
        # DEPRECATED fields: garbage values prove they are never read.
        'price_per_hour': deprecated_price,
        'spot_price': '888.88',
        'currency': currency,
        'location_prices': [{
            'location_code': code_,
            'price_per_hour': on_demand,
            'spot_price': spot,
        } for (code_, on_demand, spot) in location_prices],
    }


_LOCATIONS = [{
    'code': 'FIN-01',
    'name': 'Finland 1',
    'country_code': 'FI'
}, {
    'code': 'FIN-02',
    'name': 'Finland 2',
    'country_code': 'FI'
}]

_AVAILABILITY = [{
    'location_code': 'FIN-01',
    'availabilities': ['1H100.80S.22V'],
}, {
    'location_code': 'FIN-02',
    'availabilities': ['1H100.80S.22V', '2H100.80S.60V'],
}]

# The SPOT tier's stock: spot-available at FIN-01 only.
_AVAILABILITY_SPOT = [{
    'location_code': 'FIN-01',
    'availabilities': ['1H100.80S.22V'],
}, {
    'location_code': 'FIN-02',
    'availabilities': [],
}]


class TestFetcherJoin(unittest.TestCase):

    def test_in_stock_pair_emits_row_with_location_prices(self):
        types = [
            _type_row(location_prices=(('FIN-01', '3.17', '0.31'),
                                       ('FIN-02', '2.90', '0.29')))
        ]
        availability = [{
            'location_code': 'FIN-01',
            'availabilities': ['1H100.80S.22V'],
        }, {
            'location_code': 'FIN-02',
            'availabilities': ['1H100.80S.22V'],
        }]
        rows, skips = fetch_verda.rows_and_skips(types, _LOCATIONS,
                                                 availability,
                                                 _AVAILABILITY_SPOT)
        self.assertEqual(skips, {})
        # FIN-01 price, NOT the deprecated top-level 999.99.
        self.assertEqual([r[0] for r in rows], ['1H100.80S.22V'] * 2)
        prices = {(r[6], float(r[5])) for r in rows}
        self.assertIn(('Finland 1', 3.17), prices)
        self.assertIn(('Finland 2', 2.90), prices)
        # SpotPrice carries the location's spot price ONLY where the
        # SPOT tier reports stock: FIN-01 has spot stock, FIN-02 does
        # NOT, so its SpotPrice cell stays EMPTY (never the on-demand
        # price, never an unavailable spot's price).
        rows_by_region = {r[6]: r for r in rows}
        self.assertEqual(float(rows_by_region['Finland 1'][8]), 0.31)
        self.assertEqual(rows_by_region['Finland 2'][8], '')

    def test_spot_only_pair_kept_with_empty_on_demand_price(self):
        # In the SPOT list but NOT the on-demand list: spot-only
        # capacity must not vanish, and its on-demand price must stay
        # EMPTY (the on-demand stock signal is absent).
        types = [_type_row(location_prices=(('FIN-01', '3.17', '0.31'),))]
        rows, _ = fetch_verda.rows_and_skips(types, _LOCATIONS, [{
            'location_code': 'FIN-01',
            'availabilities': [],
        }], [{
            'location_code': 'FIN-01',
            'availabilities': ['1H100.80S.22V'],
        }])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], '')  # no on-demand stock
        self.assertEqual(float(rows[0][8]), 0.31)

    def test_on_demand_only_pair_has_empty_spot_price(self):
        types = [
            _type_row(location_prices=(('FIN-01', '3.17', '0.31'),
                                       ('FIN-02', '2.90', '0.29')))
        ]
        rows, _ = fetch_verda.rows_and_skips(types, _LOCATIONS, _AVAILABILITY,
                                             [])
        self.assertEqual(len(rows), 2)  # both locations, on-demand stock
        self.assertTrue(all(r[8] == '' for r in rows))

    def test_non_usd_type_skipped_entirely(self):
        types = [_type_row(currency='eur')]
        rows, skips = fetch_verda.rows_and_skips(types, _LOCATIONS,
                                                 _AVAILABILITY,
                                                 _AVAILABILITY_SPOT)
        self.assertEqual(rows, [])
        self.assertIn('non-usd', skips['1H100.80S.22V'][0])

    def test_stock_without_price_is_skipped_with_reason(self):
        # In stock at FIN-01 but priced only for FIN-02: the FIN-01 row
        # is skipped, never fabricated.
        types = [_type_row(location_prices=(('FIN-02', '2.90', '0.29'),))]
        rows, skips = fetch_verda.rows_and_skips(types, _LOCATIONS,
                                                 _AVAILABILITY,
                                                 _AVAILABILITY_SPOT)
        self.assertEqual([r[6] for r in rows], ['Finland 2'])
        self.assertTrue(skips['1H100.80S.22V'])
        self.assertIn('refusing to fabricate',
                      '; '.join(skips['1H100.80S.22V']))

    def test_region_column_is_location_name(self):
        types = [_type_row(location_prices=(('FIN-01', '3.17', '0.31'),))]
        rows, _ = fetch_verda.rows_and_skips(types, _LOCATIONS, _AVAILABILITY,
                                             _AVAILABILITY_SPOT)
        self.assertEqual(rows[0][6], 'Finland 1')

    def test_accelerator_derived_from_model(self):
        rows, _ = fetch_verda.rows_and_skips(
            [_type_row(location_prices=(('FIN-01', '3.17', '0.31'),))],
            _LOCATIONS, [{
                'location_code': 'FIN-01',
                'availabilities': ['1H100.80S.22V']
            }], [])
        self.assertEqual(rows[0][1], 'H100')
        self.assertEqual(rows[0][2], 1)  # gpu.number_of_gpus

    def test_availability_code_missing_from_types_is_recorded(self):
        rows, skips = fetch_verda.rows_and_skips([], _LOCATIONS, _AVAILABILITY,
                                                 _AVAILABILITY_SPOT)
        self.assertEqual(rows, [])
        self.assertTrue(skips.get('1H100.80S.22V'))

    def test_string_prices_parse_and_garbage_does_not(self):
        # Prices arrive as STRINGS per the spec; garbage never parses.
        # pylint: disable=protected-access
        # (_as_float is the module's parse helper; testing it directly is
        # the point.)
        self.assertEqual(fetch_verda._as_float('2.90'), 2.90)
        self.assertEqual(fetch_verda._as_float(3), 3.0)
        self.assertIsNone(fetch_verda._as_float('free'))
        self.assertIsNone(fetch_verda._as_float(None))

    def test_deprecated_fields_never_read(self):
        # Bogus deprecated top-level prices: a row priced from them
        # would crash float(); the emitted price must come from
        # location_prices.
        types = [
            _type_row(deprecated_price='NOT-A-NUMBER',
                      location_prices=(('FIN-01', '3.17', '0.31'),))
        ]
        rows, _ = fetch_verda.rows_and_skips(types, _LOCATIONS, _AVAILABILITY,
                                             _AVAILABILITY_SPOT)
        self.assertEqual(float(rows[0][5]), 3.17)

    def test_fetcher_fails_closed_on_missing_credentials(self):
        env = {
            verda_api.CLIENT_ID_ENV: '',
            verda_api.CLIENT_SECRET_ENV: '',
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch.object(utils,
                                       'resolve_credentials',
                                       return_value=None):
                    with self.assertRaises(verda_api.VerdaAuthError):
                        fetch_verda.main(
                            ['--output', tmp + '/should-not-exist.csv'])


# -- cloud wiring ----------------------------------------------------------


class TestCloudWiring(unittest.TestCase):

    def test_repr_is_the_catalog_name(self):
        # pylint: disable=protected-access
        # _REPR IS the catalog name; the value is the assertion.
        self.assertEqual(verda_cloud.Verda()._REPR, 'Verda')

    def test_check_credentials_fails_closed_without_credentials(self):
        env = {
            verda_api.CLIENT_ID_ENV: '',
            verda_api.CLIENT_SECRET_ENV: '',
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(utils,
                                   'resolve_credentials',
                                   return_value=None):
                ok, err = verda_cloud.Verda.check_credentials(
                    clouds.CloudCapability.COMPUTE)
        self.assertFalse(ok)
        self.assertIn('VERDA_CLIENT_ID', err)

    def test_check_credentials_probes_balance(self):
        fake_client = mock.Mock()
        fake_client.get_balance.return_value = {'amount': 10, 'currency': 'usd'}
        with mock.patch.object(utils,
                               'resolve_credentials',
                               return_value=('cid', 'csecret')):
            with mock.patch.object(utils,
                                   'VerdaClient',
                                   return_value=fake_client) as ctor:
                ok, err = verda_cloud.Verda.check_credentials(
                    clouds.CloudCapability.COMPUTE)
        self.assertTrue(ok, err)
        self.assertIsNone(err)
        ctor.assert_called_once_with('cid', 'csecret')
        fake_client.get_balance.assert_called_once_with()

    def test_spot_is_supported(self):
        # Verda HAS a spot tier; SPOT_INSTANCE must NOT be in the
        # unsupported set (the quantacloud difference). pylint: disable=protected-access
        # (The features map is a class-level protected constant; the
        # set membership IS the assertion.)

        self.assertNotIn(clouds.CloudImplementationFeatures.SPOT_INSTANCE,
                         verda_cloud.Verda._CLOUD_UNSUPPORTED_FEATURES)
        self.assertIn(clouds.CloudImplementationFeatures.STOP,
                      verda_cloud.Verda._CLOUD_UNSUPPORTED_FEATURES)


# -- provisioner -----------------------------------------------------------


class _StubClient:

    def __init__(self, *, locations=(), instances=(), created=None):
        self.locations = list(locations)
        self.instances = list(instances)
        self.created = created or {
            'id': 'inst-1',
            'status': 'running',
            'ip': '198.51.100.7'
        }
        self.calls = []

    def list_locations(self):
        return self.locations

    def list_instances(self):
        return self.instances

    def get_instance(self, instance_id):
        self.calls.append(('get', instance_id))
        if self.instances:
            return self.instances[0]
        raise utils.VerdaNotFoundError('missing')

    def ensure_ssh_key(self, name, public_key):
        self.calls.append(('ensure_ssh_key', name, public_key))
        return 'key-1'

    def create_instance(self, **body):
        self.calls.append(('create', body))
        return self.created

    def delete_instance(self, instance_id):
        self.calls.append(('delete', instance_id))

    def wait_until_running(self, instance_id):
        return self.created


def _config(instance_type='1H100.80S.22V', public_key='ssh-ed25519 REAL'):
    return verda_instance.common.ProvisionConfig(
        provider_config={},
        authentication_config={},
        docker_config={},
        node_config={
            'InstanceType': instance_type,
            'PublicKey': public_key,
        },
        count=1,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=[],
    )


class TestProvisioner(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='verda-lane-test-')
        self._state_patcher = mock.patch.object(verda_instance, '_STATE_PATH',
                                                self._tmp + '/state.json')
        self._state_patcher.start()
        self.addCleanup(self._state_patcher.stop)

    def test_run_instances_maps_name_to_location_code(self):
        client = _StubClient(locations=[
            {
                'code': 'FIN-01',
                'name': 'Finland 1',
                'country_code': 'FI'
            },
        ])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            record = verda_instance.run_instances('Finland 1', 'cluster',
                                                  'cluster-on-verda', _config())
        create = [c for c in client.calls if c[0] == 'create'][0][1]
        self.assertEqual(create['location_code'], 'FIN-01')
        self.assertEqual(create['instance_type'], '1H100.80S.22V')
        self.assertEqual(create['is_spot'], False)
        self.assertEqual(record.head_instance_id, 'inst-1')

    def test_rendered_preemptible_flag_drives_is_spot(self):
        # The vast #92 lesson (CodeRabbit PR #103): the template renders
        # `Preemptible: {{use_spot}}` into node_config; the provisioner
        # must read THAT flag or spot requests deploy ON-DEMAND.
        client = _StubClient(locations=[
            {
                'code': 'FIN-01',
                'name': 'Finland 1',
                'country_code': 'FI'
            },
        ])
        config = _config()
        config.node_config['Preemptible'] = True
        with mock.patch.object(verda_instance, '_client', return_value=client):
            verda_instance.run_instances('Finland 1', 'cluster',
                                         'cluster-on-verda', config)
        create = [c for c in client.calls if c[0] == 'create'][0][1]
        self.assertTrue(create['is_spot'])

    def test_unknown_region_name_refuses_to_guess(self):
        client = _StubClient(locations=[])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            with self.assertRaises(utils.VerdaError):
                verda_instance.run_instances('Nowhere', 'cluster',
                                             'cluster-on-verda', _config())

    def test_placeholder_public_key_refused(self):
        client = _StubClient()
        with mock.patch.object(verda_instance, '_client', return_value=client):
            with self.assertRaises(utils.VerdaError):
                verda_instance.run_instances(
                    'Finland 1', 'cluster', 'cluster-on-verda',
                    _config(public_key='skypilot:ssh_public_key_content'))

    def test_foreign_live_instance_refused(self):
        client = _StubClient(instances=[{'id': 'other', 'status': 'running'}])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            with self.assertRaises(utils.VerdaError) as ctx:
                verda_instance.run_instances('Finland 1', 'cluster',
                                             'cluster-on-verda', _config())
            self.assertIn('refusing to guess', str(ctx.exception))

    def test_terminate_unmapped_with_live_boxes_refuses(self):
        client = _StubClient(instances=[{'id': 'other', 'status': 'running'}])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            with self.assertRaises(utils.VerdaError):
                verda_instance.terminate_instances('unknown-cluster')

    def test_terminate_unmapped_and_empty_is_silent_success(self):
        client = _StubClient(instances=[])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            verda_instance.terminate_instances('unknown-cluster')
        self.assertEqual(client.calls, [])

    def test_identity_persisted_before_readiness_wait(self):
        # pylint: disable=protected-access
        # CodeRabbit PR #103: the ID is saved BEFORE wait_until_running,
        # so a wait timeout leaves a resolvable mapping, not an orphan
        # (reading the patched _STATE_PATH directly is the assertion).

        class _FailWaitClient(_StubClient):

            def wait_until_running(self, instance_id):
                raise utils.VerdaError('wait failed (injected)')

        client = _FailWaitClient(locations=[
            {
                'code': 'FIN-01',
                'name': 'Finland 1',
                'country_code': 'FI'
            },
        ])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            with self.assertRaises(utils.VerdaError):
                verda_instance.run_instances('Finland 1', 'cluster',
                                             'cluster-on-verda', _config())
        with open(verda_instance._STATE_PATH, encoding='utf-8') as h:
            state = json.load(h)
        # The mapping survived the wait failure: teardown can resolve.
        self.assertEqual(state['cluster-on-verda']['instance_id'], 'inst-1')

    def test_terminate_unmapped_resolves_by_hostname_on_cluster_head(self):
        # AUTODOWN (CodeRabbit PR #103): teardown runs on the CLUSTER
        # HEAD, which has no ~/.sky/verda-clusters.json. The deploy set
        # hostname = cluster_name_on_cloud, so the head resolves OUR box
        # by hostname from the live instance list.
        client = _StubClient(instances=[{
            'id': 'inst-77',
            'status': 'running',
            'hostname': 'c',
        }, {
            'id': 'other',
            'status': 'running',
            'hostname': 'someone-elses-box',
        }])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            verda_instance.terminate_instances('c')
        self.assertEqual(client.calls, [('delete', 'inst-77')])

    def test_terminate_hostname_ambiguity_refuses(self):
        # Two live instances share our hostname: never guess between
        # them.
        client = _StubClient(instances=[{
            'id': 'a',
            'status': 'running',
            'hostname': 'c',
        }, {
            'id': 'b',
            'status': 'running',
            'hostname': 'c',
        }])
        with mock.patch.object(verda_instance, '_client', return_value=client):
            with self.assertRaises(utils.VerdaError) as ctx:
                verda_instance.terminate_instances('c')
        self.assertIn('refusing to guess', str(ctx.exception))

    def test_terminate_mapped_deletes_and_forgets(self):
        # pylint: disable=protected-access
        # _STATE_PATH is patched to the tmp dir in setUp; reading/writing
        # the state store directly is the point of the forget assertion.
        with open(verda_instance._STATE_PATH, 'w', encoding='utf-8') as h:
            json.dump({'c': {'instance_id': 'inst-9'}}, h)
        client = _StubClient()
        with mock.patch.object(verda_instance, '_client', return_value=client):
            verda_instance.terminate_instances('c')
        self.assertEqual(client.calls, [('delete', 'inst-9')])
        with open(verda_instance._STATE_PATH, encoding='utf-8') as h:
            self.assertEqual(json.load(h), {})


if __name__ == '__main__':
    unittest.main()
