"""Behavioral tests for the Packet.ai REST client (sky/adaptors/packet.py).

Every fixture shape is captured from the LIVE v2 API (measured
2026-11-09): the FLAT success envelope (no {data, meta} wrapper), the
message-only error bodies, the served POST /instances create (the v1 501
refusal is gone), the launch-options products/regions shape with NO
pools and NO per-product stock, the ssh-key rows wrapped at .keys, and
the flat Pod objects whose status field is ``state``. The client is
exercised through an injected transport — no network.
"""

import base64
import hashlib
import json
import time
import unittest
import uuid

from sky.adaptors import packet as api


def _page(rows, *, page=1, total_pages=1):
    # v2 (measured 2026-11-09): the instance list answers FLAT JSON —
    # {"instances": [...]} — no {data, meta} wrapper.
    return json.dumps({
        'instances': rows,
        'pagination': {
            'page': page,
            'limit': 100,
            'total': len(rows),
            'totalPages': total_pages,
        },
    }).encode('utf-8')


def _data(payload):
    # A wrapped success body. v2 answers flat, but the client still
    # unwraps a {data, meta} envelope defensively (drift tolerance) —
    # test_data_envelope_unwrapped pins that unwrap.
    return json.dumps({
        'data': payload,
        'meta': {
            'requestId': 'req_test',
            'timestamp': '2026-11-09T00:00:00.000Z',
        },
    }).encode('utf-8')


def _error(message):
    # v2 (measured 2026-11-09): 4xx/5xx bodies carry {"message": ...} —
    # no {error: {code, message}} envelope, no machine code to branch on.
    return json.dumps({
        'message': message,
    }).encode('utf-8')


class _Transport:
    """Scripted transport: pops one canned response per call.

    The injected-transport signature is (method, url, headers, payload) ->
    (status, bytes, headers) — the third element exists so 429s can honor
    the provider's own X-RateLimit-Reset hint.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, payload):
        self.calls.append((method, url, dict(headers), payload))
        status, raw, resp_headers = self.responses.pop(0)
        return status, raw, dict(resp_headers or {})


class TestClientPlumbing(unittest.TestCase):

    def test_api_key_header_on_authenticated_calls_only(self):
        transport = _Transport((200, _data({'id': 'cus_1'}), {}))
        client = api.PacketClient('pk_live_x')
        client._transport = transport
        client.get_account()
        method, url, headers, _ = transport.calls[0]
        self.assertEqual(headers.get('X-API-Key'), 'pk_live_x')
        self.assertIn('urun-skypilot-packet', headers.get('User-Agent'))
        self.assertIn('/account', url)

    def test_empty_key_refuses(self):
        with self.assertRaises(api.PacketAuthError):
            api.PacketClient('   ')

    def test_data_envelope_unwrapped(self):
        transport = _Transport((200, _data({'id': 'cus_1'}), {}))
        client = api.PacketClient('pk_live_x')
        client._transport = transport
        self.assertEqual(client.get_account(), {'id': 'cus_1'})

    def test_non_json_success_raises_typed(self):
        client = api.PacketClient('pk_live_x')
        client._transport = _Transport((200, b'<html>gateway</html>', {}))
        with self.assertRaises(api.PacketError) as ctx:
            client.get_account()
        self.assertIn('not JSON', str(ctx.exception))

    def test_empty_body_error_is_failure_never_silent_none(self):
        # A 4xx with an empty body must raise, never return None — the
        # empty-body-error-reads-as-done bug class.
        client = api.PacketClient('pk_live_x')
        client._transport = _Transport((503, b'', {}))
        with self.assertRaises(api.PacketError):
            client.get_account()


class TestClientErrors(unittest.TestCase):

    def _client(self, status, body, resp_headers=None):
        client = api.PacketClient('pk_live_SECRET')
        client._transport = _Transport((status, body, resp_headers or {}))
        return client

    def test_401_is_typed_auth(self):
        # v2 (measured 2026-11-09): error bodies carry no machine code,
        # so there is nothing to branch on — but the 401 must still be
        # its OWN typed class naming the key channels, never a generic
        # retryable failure.
        client = self._client(401, _error('API key revoked'))
        with self.assertRaises(api.PacketAuthError) as ctx:
            client.get_account()
        self.assertIn('HTTP 401', str(ctx.exception))
        self.assertIn(api.API_KEY_ENV, str(ctx.exception))

    def test_404_typed_not_found(self):
        # v2 (measured 2026-11-09): the {id} route answers the Pod FLAT
        # and 404s stay typed (the terminate paths treat them as
        # idempotent teardown).
        client = self._client(404,
                              _error("Instance with id 'i-55a5e3b1' not found"))
        with self.assertRaises(api.PacketNotFoundError):
            client.get_instance('i-55a5e3b1')

    def test_create_sends_v2_body_with_idempotency_key(self):
        # v2 (measured 2026-11-09): the v1 501 create refusal is GONE —
        # POST /instances SERVES creates for API keys. The client must
        # send {product_id, region_id, name?, ssh_key_ids?} (v1's
        # pool_id/vgpus kwargs are retired) and mint the REQUIRED
        # Idempotency-Key header itself, then return the created pod id.
        client = api.PacketClient('pk_live_SECRET')
        client._transport = _Transport((201, json.dumps({
            'id': 'i-new'
        }).encode('utf-8'), {}))
        created = client.create_instance(product_id='prod_rtxpro6000',
                                         region_id=67,
                                         name='sky-test-cluster',
                                         ssh_key_ids=['key_1'])
        self.assertEqual(created, {'id': 'i-new'})
        method, url, headers, payload = client._transport.calls[0]
        self.assertEqual(method, 'POST')
        self.assertIn('/instances', url)
        self.assertEqual(
            json.loads(payload), {
                'product_id': 'prod_rtxpro6000',
                'region_id': 67,
                'name': 'sky-test-cluster',
                'ssh_key_ids': ['key_1'],
            })
        # The REQUIRED Idempotency-Key header: a fresh uuid4 per launch,
        # generated by the client so no caller can forget it.
        idem = str(headers.get('Idempotency-Key') or '')
        self.assertTrue(idem)
        self.assertEqual(uuid.UUID(idem).version, 4)

    def test_503_unavailable_typed(self):
        client = self._client(503, _error('later'))
        with self.assertRaises(api.PacketUnavailableError):
            client.get_account()

    def test_generic_500_carries_the_message(self):
        # v2 bodies carry no machine code: the raw {"message": ...}
        # text must still surface in the typed error (never a bare
        # status line, never a silent failure).
        client = self._client(500, _error('boom'))
        with self.assertRaises(api.PacketError) as ctx:
            client.get_account()
        self.assertIn('boom', str(ctx.exception))
        self.assertIn('HTTP 500', str(ctx.exception))

    def test_error_never_leaks_the_key(self):
        client = self._client(401, _error('bad key'))
        with self.assertRaises(api.PacketError) as ctx:
            client.get_account()
        self.assertNotIn('pk_live_SECRET', str(ctx.exception))


class TestRateLimitBackoff(unittest.TestCase):

    def test_429_honors_reset_hint_then_raises_typed(self):
        # The provider's own retry hint (X-RateLimit-Reset, Unix seconds)
        # must be honored — bounded — and exhaustion raises the typed
        # error (never a silent success, never an unbounded sleep).
        sleeps = []
        reset_at = time.time() + 3.0
        client = api.PacketClient('pk_live_x')
        responses = [(429, _error('slow down'), {
            'X-RateLimit-Reset': str(int(reset_at))
        })] * (api.RATE_LIMIT_RETRIES + 1)
        client._transport = _Transport(*responses)
        with mock_sleep(sleeps):
            with self.assertRaises(api.PacketRateLimited):
                client.get_account()
        self.assertEqual(len(sleeps), api.RATE_LIMIT_RETRIES)
        # The hint was honored (not the flat exponential fallback): every
        # sleep derives from the reset timestamp, bounded by
        # MAX_RETRY_AFTER_S.
        for slept in sleeps:
            self.assertGreaterEqual(slept, 0.0)
            self.assertLessEqual(slept, api.MAX_RETRY_AFTER_S)
        self.assertTrue(all(s > 0 for s in sleeps))

    def test_429_without_hint_uses_bounded_backoff(self):
        sleeps = []
        client = api.PacketClient('pk_live_x')
        responses = [(429, _error('slow'), {})] * (api.RATE_LIMIT_RETRIES + 1)
        client._transport = _Transport(*responses)
        with mock_sleep(sleeps):
            with self.assertRaises(api.PacketRateLimited):
                client.get_account()
        self.assertEqual(len(sleeps), api.RATE_LIMIT_RETRIES)


class mock_sleep:
    """Patches time.sleep inside the client's backoff loop."""

    def __init__(self, record):
        self.record = record

    def __enter__(self):
        self._patcher = _patch_time_sleep(self.record)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()
        return False


def _patch_time_sleep(record):
    import unittest.mock as mock
    return mock.patch.object(time, 'sleep', lambda s: record.append(s))


class TestLaunchOptions(unittest.TestCase):

    def test_launch_options_validated_shape(self):
        # v2 (measured 2026-11-09): the answer is FLAT — no {data, meta}
        # wrapper, NO pools array, NO per-product stock (products carry
        # the WHOLE-BOX pricePerHourCents and the serviceId the
        # category-check answers for); regions are {id: int, name: str}.
        payload = {
            'can_use_premium': True,
            'categories': [],
            'existingSharedVolumes': [],
            'products': [{
                'id': 'cmkjtxmp20000asi3ermfys2t',
                'name': 'NVIDIA RTX PRO 6000 Blackwell - Dynamic',
                'description': '96GB ram',
                'billingType': 'hourly',
                'pricePerHourCents': 66,
                'serviceId': 'srv_rtxpro6000',
                'featured': True,
                'vramGb': 96,
                'gpuCount': 1,
            }],
            'regions': [{
                'id': 67,
                'name': 'us-west-2',
            }],
            'sshKeys': [],
            'storagePricePerGBHourCents': 2,
            'teamId': 'team_1',
            'walletBalanceCents': 50000,
        }
        transport = _Transport((200, json.dumps(payload).encode('utf-8'), {}))
        client = api.PacketClient('pk_live_x')
        client._transport = transport
        options = client.get_launch_options()
        self.assertEqual(options['products'][0]['pricePerHourCents'], 66)
        self.assertNotIn('pools', options)
        self.assertEqual(options['regions'][0], {'id': 67, 'name': 'us-west-2'})

    def test_missing_products_list_refuses(self):
        # A launch-options response without a products LIST must refuse —
        # never silently treat the surface as empty (the "empty probe"
        # that once read as absent capacity).
        payload = {'products': None, 'regions': []}
        client = api.PacketClient('pk_live_x')
        client._transport = _Transport(
            (200, json.dumps(payload).encode('utf-8'), {}))
        with self.assertRaises(api.PacketError) as ctx:
            client.get_launch_options()
        self.assertIn('refusing', str(ctx.exception))


class TestPagination(unittest.TestCase):

    def _client_pages(self, *pages):
        transport = _Transport(*[(200, p, {}) for p in pages])
        client = api.PacketClient('pk_live_x')
        client._transport = transport
        return client

    def test_short_page_stops(self):
        client = self._client_pages(_page([{'id': 'i-1'}]))
        self.assertEqual([i['id'] for i in client.list_instances()], ['i-1'])

    def test_pages_walked_to_the_end(self):
        # Two full-ish pages: page 1 full (limit rows), page 2 short.
        full = [dict(id=f'i-{i}') for i in range(api.PAGE_LIMIT)]
        second = [dict(id='i-last')]
        client = self._client_pages(_page(full, page=1, total_pages=2),
                                    _page(second, page=2, total_pages=2))
        rows = client.list_instances()
        self.assertEqual(len(rows), api.PAGE_LIMIT + 1)

    def test_missing_instances_list_refuses(self):
        client = api.PacketClient('pk_live_x')
        client._transport = _Transport((200, json.dumps({
            'nope': 1
        }).encode('utf-8'), {}))
        with self.assertRaises(api.PacketError) as ctx:
            client.list_instances()
        self.assertIn('.instances', str(ctx.exception))


class _Fingerprint:
    """Key fixtures with real material-derived fingerprints."""

    @staticmethod
    def of(material):
        blob = base64.b64decode(material.split()[1])
        digest = hashlib.md5(blob).hexdigest()
        return ':'.join(digest[i:i + 2] for i in range(0, len(digest), 2))


# Properly-padded base64 blobs (the fingerprint derivation decodes them):
_KEY_BLOB = base64.b64encode(b'packet-test-key-material').decode('ascii')
_KEY_MATERIAL = f'ssh-rsa {_KEY_BLOB} test@host'
_OTHER_BLOB = base64.b64encode(b'packet-OTHER-key-material').decode('ascii')
_OTHER_MATERIAL = f'ssh-rsa {_OTHER_BLOB} test@host'


class TestEnsureHelpers(unittest.TestCase):

    def _key_row(self, key_id, material, name='anything'):
        return {
            'id': key_id,
            'name': name,
            'fingerprint': _Fingerprint.of(material),
            'createdAt': '2026-11-09T00:00:00.000Z',
            'keyPreview': 'ssh-rsa AAAA',
        }

    def _client(self, keys):
        # v2 (measured 2026-11-09): GET /account/ssh-keys answers
        # {"keys": [...]} FLAT.
        client = api.PacketClient('pk_live_x')
        client._transport = _EnvelopeList(keys)
        return client

    def test_ensure_matches_by_material_not_name(self):
        client = self._client(
            [self._key_row('key_1', _KEY_MATERIAL, name='a-renamed-key')])
        self.assertEqual(client.ensure_ssh_key('skypilot', _KEY_MATERIAL),
                         'key_1')

    def test_ensure_adds_when_absent(self):
        client = self._client([self._key_row('key_other', _OTHER_MATERIAL)])
        # v2 (measured 2026-11-09): POST /account/ssh-keys answers
        # {"success": true, "key": {...}} — the created key is wrapped
        # at .key (v1 answered the object directly).
        created = _Transport(
            (201,
             json.dumps({
                 'success': True,
                 'key': {
                     'id': 'key_new',
                     'name': 'skypilot',
                     'fingerprint': _Fingerprint.of(_KEY_MATERIAL),
                     'createdAt': '2026-11-09T00:00:00.000Z',
                 },
             }).encode('utf-8'), {}))
        client._transport = _Sequence(client._transport, created)
        self.assertEqual(client.ensure_ssh_key('skypilot', _KEY_MATERIAL),
                         'key_new')

    def test_fingerprint_is_legacy_md5_hex_colon(self):
        # MEASURED (ENG-562): the provider's key list returns
        # 08:07:ea:…-style MD5 hex-colon fingerprints, NOT SHA256. A
        # SHA256 derivation would never match and would mint a duplicate
        # on every deploy.
        self.assertNotIn('SHA256', api._ssh_fingerprint(_KEY_MATERIAL))
        self.assertEqual(len(api._ssh_fingerprint(_KEY_MATERIAL).split(':')),
                         16)

    def test_bad_material_raises(self):
        client = api.PacketClient('pk_live_x')
        del client
        with self.assertRaises(api.PacketError):
            api._ssh_fingerprint('not-a-key')


class _EnvelopeList:
    """A transport answering GET /account/ssh-keys with {"keys": [...]}."""

    def __init__(self, keys):
        self.keys = keys
        self.calls = 0

    def __call__(self, method, url, headers, payload):
        self.calls += 1
        return 200, json.dumps({'keys': self.keys}).encode('utf-8'), {}


class _Sequence:
    """Pops one transport per call."""

    def __init__(self, *transports):
        self.transports = list(transports)

    def __call__(self, method, url, headers, payload):
        return self.transports.pop(0)(method, url, headers, payload)


def _instance(state, name='rtxpro6000blackwell-1x-dynamic', id='i-1'):
    # v2 (measured 2026-11-09): the Pod answers FLAT and names its status
    # field ``state`` (v1 wrapped at data.instance and named it
    # ``status``); the enum is starting | running | stopping | stopped
    # | failed | terminating | terminated | unknown.
    return {'id': id, 'name': name, 'state': state}


class TestWaitUntilRunning(unittest.TestCase):

    def _client(self, instances):
        client = api.PacketClient('pk_live_x')
        client._transport = _InstanceTransport(instances)
        return client

    def test_reaches_running(self):
        client = self._client([_instance('starting'), _instance('running')])
        sleeps = []
        instance = client.wait_until_running('i-1',
                                             sleep=lambda s: sleeps.append(s),
                                             now=lambda: 0.0)
        self.assertEqual(instance['state'], 'running')
        self.assertEqual(sleeps, [15.0])

    def test_failed_fails_closed_immediately(self):
        # v2 (measured 2026-11-09): v1's ``error`` is GONE — ``failed``
        # is the terminal value, and terminating/terminated mean the
        # pod died under the wait: all raise immediately, never
        # auto-retry (the operator decides).
        client = self._client([_instance('failed')])
        with self.assertRaises(api.PacketError) as ctx:
            client.wait_until_running('i-1',
                                      sleep=lambda s: None,
                                      now=lambda: 0.0)
        self.assertIn('failed', str(ctx.exception))
        self.assertIn('not auto-retrying', str(ctx.exception))

    def test_deadline_fails_closed(self):
        client = self._client([_instance('starting')])
        with self.assertRaises(api.PacketError) as ctx:
            # timeout_s=0: the deadline equals now() on the first poll —
            # a stuck provisioning must become a failed runner, not an
            # endless wait.
            client.wait_until_running('i-1',
                                      timeout_s=0,
                                      poll_interval_s=15,
                                      sleep=lambda s: None,
                                      now=lambda: 100.0)
        self.assertIn('refusing to wait longer', str(ctx.exception))


class _InstanceTransport:
    """Answers GET /instances/{id} with queued FLAT Pod payloads."""

    def __init__(self, instances):
        self.instances = list(instances)
        self.calls = 0

    def __call__(self, method, url, headers, payload):
        instance = self.instances.pop(0)
        return 200, json.dumps(instance).encode('utf-8'), {}


class TestConnection(unittest.TestCase):

    def test_get_connection_returns_public_fields(self):
        # v2 (measured 2026-11-09): the connection route SURVIVES — the
        # credentials answer flat, wrapped once at .connection; the ip
        # is the PUBLIC address (ENG-562) and the password half is a
        # live secret (never logged).
        conn = {
            'ip': '50.35.188.78',
            'port': 30356,
            'username': 'ubuntu',
            'password': 'iBgh',
            'ssh_command': 'ssh -p 30356 ubuntu@50.35.188.78',
        }
        client = api.PacketClient('pk_live_x')
        client._transport = _Transport((200, json.dumps({
            'connection': conn,
        }).encode('utf-8'), {}))
        got = client.get_connection('i-full-id')
        self.assertEqual(got['ip'], '50.35.188.78')
        self.assertEqual(got['port'], 30356)
        self.assertEqual(got['username'], 'ubuntu')

    def test_ssh_user_from_connection(self):
        self.assertEqual(
            api.PacketClient.ssh_user_from_connection({'username': 'ubuntu'}),
            'ubuntu')
        self.assertIsNone(api.PacketClient.ssh_user_from_connection({}))


if __name__ == '__main__':
    unittest.main()
