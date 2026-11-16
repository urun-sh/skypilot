"""Behavioral tests for the Verda REST client (sky/adaptors/verda.py).

Transport-injected: no network. Every canned response is derived from
the OpenAPI 3.1 spec's examples at /tmp/verda-openapi.json (reviewed
2026-11-16), so a regression here is a wire-protocol drift.
"""

import json
import time
import unittest
from unittest import mock

from sky.adaptors import verda as api


def _json(payload):
    return json.dumps(payload).encode('utf-8')


def _error(code, message):
    return _json({'code': code, 'message': message})


class _Transport:
    """Scripted transport: pops one canned response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, payload):
        self.calls.append((method, url, dict(headers), payload))
        return self.responses.pop(0)

    def push(self, response):
        self.responses.append(response)


def _token_response():
    # GetAccessTokenResponseDto example from the OpenAPI spec.
    return (200,
            _json({
                'access_token': 'jwt-token-1',
                'token_type': 'Bearer',
                'expires_in': 3600,
                'refresh_token': 'r-tok',
                'scope': 'cloud-api-v1',
            }))


def _authed_client(transport, responses):
    transport.responses = [_token_response()] + list(transport.responses)
    return api.VerdaClient('cid', 'csecret', transport=transport)


class TestAuth(unittest.TestCase):

    def test_empty_credentials_refused(self):
        for client_id, client_secret in [('', 's'), ('c', ''), (None, None)]:
            with self.assertRaises(api.VerdaAuthError):
                api.VerdaClient(client_id, client_secret)

    def test_token_mint_body_and_bearer_header(self):
        transport = _Transport([
            _token_response(),
            (200, _json({
                'amount': 10,
                'currency': 'usd'
            })),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        client.get_balance()
        mint = transport.calls[0]
        self.assertEqual(mint[0], 'POST')
        self.assertTrue(mint[1].endswith('/v1/oauth2/token'))
        self.assertEqual(
            json.loads(mint[3]), {
                'grant_type': 'client_credentials',
                'client_id': 'cid',
                'client_secret': 'csecret',
            })
        balance = transport.calls[1]
        self.assertEqual(balance[2]['Authorization'], 'Bearer jwt-token-1')
        self.assertEqual(balance[2]['User-Agent'],
                         'urun-skypilot-verda/0.1 (+https://urun.sh)')

    def test_token_is_cached_not_re_minted(self):
        transport = _Transport([
            _token_response(), (200, _json([])), (200, _json([])),
            (200, _json([]))
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        client.get_balance()
        client.list_locations()
        client.list_ssh_keys()
        # One mint + three authenticated calls, all with the same token.
        self.assertEqual(len(transport.calls), 4)
        mints = [c for c in transport.calls if 'oauth2/token' in c[1]]
        self.assertEqual(len(mints), 1)

    def test_401_re_mints_exactly_once(self):
        transport = _Transport([
            _token_response(),  # initial mint
            (401, _error('unauthorized_request', 'expired')),
            _token_response(),  # re-mint
            (200, _json([])),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        client.list_locations()
        self.assertEqual(len(transport.calls), 4)
        self.assertTrue(transport.calls[3][1].endswith('/v1/locations'))

    def test_double_401_raises_auth_error(self):
        transport = _Transport([
            _token_response(),
            (401, _error('unauthorized_request', 'bad creds')),
            _token_response(),
            (401, _error('unauthorized_request', 'still bad')),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        with self.assertRaises(api.VerdaAuthError):
            client.list_locations()

    def test_token_expiry_re_mints(self):
        transport = _Transport([
            _token_response(),
            (200, _json([])),  # first call: token still fresh (margin)
            _token_response(),  # re-mint after forced expiry
            (200, _json([])),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        client.list_locations()
        # Force the cached token past its expiry.
        client._token = (client._token[0], time.monotonic() - 1)
        client.list_locations()
        self.assertEqual(
            len([c for c in transport.calls if 'oauth2/token' in c[1]]), 2)


class TestErrors(unittest.TestCase):

    def _client(self, status, body):
        transport = _Transport([_token_response(), (status, body)])
        return api.VerdaClient('cid', 'SECRET', transport=transport)

    def test_401_is_auth_error(self):
        # The client re-mints once on 401; a SECOND 401 raises AuthError.
        client = self._client(401, _error('unauthorized_request', 'nope'))
        client._transport.push(_token_response())
        client._transport.push((401, _error('unauthorized_request',
                                            'still nope')))
        with self.assertRaises(api.VerdaAuthError):
            client.get_balance()

    def test_403_insufficient_funds_is_typed(self):
        client = self._client(403, _error('insufficient_funds', 'top up'))
        with self.assertRaises(api.VerdaInsufficientFundsError) as ctx:
            client.get_balance()
        # The stockout-vs-funds distinction lives on the exception: a
        # lane that mistakes this for capacity hunts forever on a
        # billing problem.
        self.assertEqual(ctx.exception.code, 'insufficient_funds')

    def test_404_is_not_found(self):
        client = self._client(404, _error('not_found', 'gone'))
        with self.assertRaises(api.VerdaNotFoundError):
            client.get_instance('gone-id')

    def test_429_backs_off_bounded_then_raises(self):
        transport = _Transport([_token_response()] +
                               [(429, _error('server_error', 'slow'))] *
                               (api.MAX_RATE_LIMIT_RETRIES + 1))
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        with mock.patch.object(api.time, 'sleep') as slept:
            with self.assertRaises(api.VerdaRateLimited):
                client.get_balance()
        self.assertEqual(len(slept.call_args_list), api.MAX_RATE_LIMIT_RETRIES)

    def test_generic_error_preserves_code(self):
        client = self._client(500, _error('server_error', 'boom'))
        with self.assertRaises(api.VerdaError) as ctx:
            client.get_balance()
        self.assertEqual(ctx.exception.code, 'server_error')
        self.assertIn('boom', str(ctx.exception))

    def test_no_secret_leak_in_errors(self):
        client = self._client(401, _error('unauthorized_request', 'x'))
        client._transport.push(_token_response())
        client._transport.push((401, _error('unauthorized_request', 'x')))
        with self.assertRaises(api.VerdaAuthError) as ctx:
            client.get_balance()
        self.assertNotIn('csecret', str(ctx.exception))

    def test_non_json_is_typed_error(self):
        client = self._client(200, b'\xff\xfe not json')
        with self.assertRaises(api.VerdaError):
            client.get_balance()

    def test_non_list_where_array_expected(self):
        transport = _Transport([_token_response(), (200, _json({'a': 1}))])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        with self.assertRaises(api.VerdaError):
            client.list_locations()


class TestSSHKeys(unittest.TestCase):

    def test_create_key_posts_add_key_dto_and_returns_id(self):
        transport = _Transport([
            _token_response(),
            (201, _json('f7b3b3b3-1b3b-4b3b-8b3b-3b3b3b3b3b3b')),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        key_id = client.create_ssh_key('skypilot', 'ssh-ed25519 AAAA k')
        self.assertEqual(key_id, 'f7b3b3b3-1b3b-4b3b-8b3b-3b3b3b3b3b3b')
        method, url, headers, payload = transport.calls[1]
        self.assertEqual(method, 'POST')
        self.assertTrue(url.endswith('/v1/ssh-keys'))
        self.assertEqual(json.loads(payload), {
            'name': 'skypilot',
            'key': 'ssh-ed25519 AAAA k'
        })

    def test_ensure_matches_on_material(self):
        # GetKeysResponseDto example shape: the key list returns the key
        # MATERIAL itself, so matching is exact string equality.
        material = 'ssh-ed25519 AAAA-material test@host'
        transport = _Transport([
            _token_response(),
            (200,
             _json([{
                 'id': 'k1',
                 'name': 'other-name',
                 'key': material,
                 'fingerprint': '9b:11:74:af',
                 'created_by_user_id': None,
             }])),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        self.assertEqual(client.ensure_ssh_key('skypilot', material), 'k1')
        self.assertEqual(len(transport.calls), 2)  # no duplicate minted

    def test_ensure_creates_when_no_material_match(self):
        transport = _Transport([
            _token_response(),
            (200,
             _json([{
                 'id': 'k-stale',
                 'name': 'skypilot',
                 'key': 'ssh-ed25519 OTHER',
                 'fingerprint': 'ff',
                 'created_by_user_id': None,
             }])),
            (201, _json('k-new')),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        self.assertEqual(client.ensure_ssh_key('skypilot', 'ssh-ed25519 MINE'),
                         'k-new')
        self.assertEqual(len(transport.calls), 3)


class TestInstances(unittest.TestCase):

    def test_create_instance_posts_deploy_dto(self):
        transport = _Transport([
            _token_response(),
            (200, _json({
                'id': 'inst-1',
                'status': 'provisioning'
            })),
        ])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        instance = client.create_instance(
            instance_type='1H100.80S.22V',
            image='ubuntu-24.04-cuda-12.8-open-docker',
            hostname='my-cluster',
            ssh_key_ids=['k1'],
            location_code='FIN-01',
            is_spot=True)
        self.assertEqual(instance['id'], 'inst-1')
        method, url, _, payload = transport.calls[1]
        self.assertEqual(method, 'POST')
        self.assertTrue(url.endswith('/v1/instances'))
        self.assertEqual(
            json.loads(payload), {
                'hostname': 'my-cluster',
                'instance_type': '1H100.80S.22V',
                'image': 'ubuntu-24.04-cuda-12.8-open-docker',
                'ssh_key_ids': ['k1'],
                'location_code': 'FIN-01',
                'is_spot': True,
            })

    def test_delete_posts_action_dto(self):
        transport = _Transport([_token_response(), (202, _json({}))])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        client.delete_instance('inst-1')
        method, url, _, payload = transport.calls[1]
        self.assertEqual((method, url),
                         ('POST', f'{api.API_BASE}/v1/instances/action'))
        self.assertEqual(json.loads(payload), {
            'action': 'delete',
            'id': 'inst-1'
        })

    def test_delete_404_is_success(self):
        transport = _Transport(
            [_token_response(), (404, _error('not_found', 'gone'))])
        client = api.VerdaClient('cid', 'csecret', transport=transport)
        client.delete_instance('gone')  # must not raise


class _FakeClock:

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _instance(status, ip=None):
    instance = {'id': 'i1', 'status': status}
    if ip:
        instance['ip'] = ip
    return instance


class TestWaitUntilRunning(unittest.TestCase):

    def _client(self, statuses):
        canned = [_token_response()
                 ] + [(200, _json(_instance(s))) for s in statuses]
        transport = _Transport(canned)
        return api.VerdaClient('cid', 'csecret', transport=transport)

    def test_running_returns(self):
        client = self._client(['provisioning', 'validating', 'running'])
        clock = _FakeClock()
        instance = client.wait_until_running('i1',
                                             poll_interval_s=10,
                                             sleep=clock.sleep,
                                             now=clock.now)
        self.assertEqual(instance['status'], 'running')

    def test_terminal_failure_raises(self):
        for status in ('error', 'installation_failed', 'no_capacity'):
            client = self._client(['provisioning', status])
            clock = _FakeClock()
            with self.assertRaises(api.VerdaError) as ctx:
                client.wait_until_running('i1',
                                          poll_interval_s=10,
                                          sleep=clock.sleep,
                                          now=clock.now)
            self.assertIn('not auto-retrying', str(ctx.exception))

    def test_deadline_fails_closed(self):
        client = self._client(['provisioning'] * 50)
        clock = _FakeClock()
        with self.assertRaises(api.VerdaError) as ctx:
            client.wait_until_running('i1',
                                      timeout_s=25,
                                      poll_interval_s=10,
                                      sleep=clock.sleep,
                                      now=clock.now)
        self.assertIn('refusing to wait longer', str(ctx.exception))

    def test_poll_floor_enforced(self):
        client = self._client(['running'])
        with self.assertRaises(api.VerdaError):
            client.wait_until_running('i1', poll_interval_s=1)


if __name__ == '__main__':
    unittest.main()
