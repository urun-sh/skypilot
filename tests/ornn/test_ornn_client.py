"""Behavioral tests for the Ornn MCP client (sky/adaptors/ornn.py).

These exercise actual client calls through the injectable transport — not
source-text inspection — so a regression fails with a behavior mismatch.
The wire contract asserted here is the provider's own (JSON-RPC 2.0 over
HTTP, SSE-or-JSON envelopes, ``content[0].text`` inner JSON, confirm-gated
mutations, OAuth ``refresh_token`` grant with client_secret_post and BOTH
tokens rotating) as receipted 2026-10-08 on ENG-541.

The first client call consumes, in order: one OAuth refresh (form-encoded
POST against the token endpoint), one ``initialize`` result, and one
``notifications/initialized`` response (ignored) — ``_tool_script`` builds
that prefix so the scripted tool responses start after the handshake.
"""

import json
import unittest
from unittest import mock

from sky.adaptors import ornn as api


def _token_response(status=200,
                    access="acc-1",
                    refresh="ref-2",
                    expires_in=3600):
    return status, json.dumps({
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": expires_in,
        "token_type": "Bearer",
    }).encode("utf-8")


def _mcp_result(payload, is_error=False):
    """A tools/call result envelope whose content[0].text is inner JSON
    (or a bare refusal token)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return 200, json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{
                "type": "text",
                "text": text
            }],
            "isError": is_error,
        },
    }).encode("utf-8")


def _sse_mcp_result(payload):
    """The same tools/call result delivered as SSE ``data:`` lines."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    envelope = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{
                "type": "text",
                "text": text
            }]
        },
    }
    return 200, ("data: " + json.dumps(envelope) + "\n\n").encode("utf-8")


def _tool_script(*tool_responses, access="acc-1", refresh="ref-2"):
    """[refresh, initialize, initialized-notification] + tool responses."""
    return ([_token_response(access=access, refresh=refresh)] +
            [_mcp_result({}), _mcp_result({})] + list(tool_responses))


class _Transport:
    """Scripted transport: pops one canned response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, payload):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "payload": payload,
        })
        return self.responses.pop(0)


def _client(responses, **kwargs):
    transport = _Transport(responses)
    credentials = api.OrnnCredentials("cid-1", "sec-1", "ref-1")
    client = api.OrnnClient(credentials, transport=transport, **kwargs)
    return client, transport


class TestHandshakeAndAuth(unittest.TestCase):

    def test_refresh_grant_form_and_rotation(self):
        client, transport = _client(
            _tool_script(_mcp_result({"orderBooks": []}),
                         access="acc-new",
                         refresh="ref-new"))
        client.list_spot_books()
        # 1) The FIRST call is the OAuth refresh against the token endpoint
        #    (not the MCP endpoint), form-encoded, client_secret_post.
        first = transport.calls[0]
        self.assertEqual(first["method"], "POST")
        self.assertTrue(first["url"].startswith(api.TOKEN_URL))
        body = dict(
            pair.split("=")
            for pair in first["payload"].decode("utf-8").split("&"))
        self.assertEqual(body["grant_type"], "refresh_token")
        self.assertEqual(body["client_id"], "cid-1")
        self.assertEqual(body["client_secret"], "sec-1")
        self.assertEqual(body["refresh_token"], "ref-1")
        # 2) BOTH tokens rotated in the credential object (the old refresh
        #    token is single-use by contract; never reused).
        self.assertEqual(client._credentials._refresh_token, "ref-new")
        # 3) The tool call carries the NEW access token; the secret and the
        #    spent refresh token never ride the URL or the JSON-RPC body.
        tool_call = transport.calls[-1]
        self.assertEqual(tool_call["url"], api.MCP_BASE)
        self.assertEqual(tool_call["headers"]["Authorization"],
                         "Bearer acc-new")
        self.assertNotIn("sec-1", tool_call["url"])
        self.assertNotIn("ref-1", tool_call["url"])
        self.assertNotIn("sec-1", tool_call["payload"].decode("utf-8"))

    def test_refresh_refusal_is_auth_error(self):
        # A broken refresh chain (HTTP 401/400 from the token endpoint) is
        # an auth failure naming the rotation, never a token value.
        client, _ = _client([(401, b'{"error": "invalid_client"}')])
        with self.assertRaises(api.OrnnAuthError) as ctx:
            client.list_spot_books()
        self.assertIn("refresh token chain", str(ctx.exception))
        self.assertNotIn("ref-1", str(ctx.exception))

    def test_credential_blob_from_env(self):
        blob = json.dumps({
            "client_id": "c",
            "client_secret": "s",
            "refresh_token": "r"
        })
        with mock.patch.dict(api.os.environ, {api.CREDENTIALS_ENV: blob}):
            credentials = api.OrnnCredentials.from_env()
        self.assertEqual(credentials._client_id, "c")
        self.assertEqual(credentials._client_secret, "s")
        self.assertEqual(credentials._refresh_token, "r")

    def test_incomplete_credential_refused(self):
        # A refresh-token-only env CANNOT refresh (the exchange requires
        # client_id + client_secret) — refuse at construction, never at
        # 401 o'clock.
        blob = json.dumps({"refresh_token": "r"})
        with mock.patch.dict(api.os.environ, {api.CREDENTIALS_ENV: blob}):
            with self.assertRaises(api.OrnnAuthError):
                api.OrnnCredentials.from_env()

    def test_missing_credential_refused(self):
        env = {
            k: v for k, v in api.os.environ.items() if k != api.CREDENTIALS_ENV
        }
        with mock.patch.dict(api.os.environ, env, clear=True):
            with mock.patch("pathlib.Path.is_file", return_value=False):
                with self.assertRaises(api.OrnnAuthError) as ctx:
                    api.OrnnCredentials.from_env()
                self.assertIn(api.CREDENTIALS_ENV, str(ctx.exception))


class TestToolCalls(unittest.TestCase):

    def test_camelcase_args_and_confirm_gating(self):
        client, transport = _client(
            _tool_script(
                _mcp_result({"preview": True}),
                _mcp_result({"order": {
                    "orderId": "o1",
                    "status": "running"
                }}),
            ))
        client.place_spot_bid("nvidia_rtx_pro_6000",
                              1,
                              order_type="market",
                              idempotency_key="k",
                              confirm=False)
        preview_args = json.loads(
            transport.calls[-1]["payload"])["params"]["arguments"]
        # camelCase, and NO confirm key without a confirm (the free preview).
        self.assertEqual(
            preview_args, {
                "gpuSlug": "nvidia_rtx_pro_6000",
                "gpus": 1,
                "orderType": "market",
                "idempotencyKey": "k"
            })
        client.place_spot_bid("nvidia_rtx_pro_6000",
                              1,
                              order_type="market",
                              idempotency_key="k",
                              confirm=True)
        confirmed_args = json.loads(
            transport.calls[-1]["payload"])["params"]["arguments"]
        self.assertIs(confirmed_args["confirm"], True)

    def test_json_payload_parsed_from_content_text(self):
        client, _ = _client(
            _tool_script(
                _mcp_result({
                    "reservations": [{
                        "reservationId": "r-1",
                        "status": "active"
                    }]
                })))
        rows = client.list_reservations()
        self.assertEqual(rows[0]["reservationId"], "r-1")

    def test_sse_envelope_parsed(self):
        client, _ = _client(
            _tool_script(
                _sse_mcp_result({"machines": [{
                    "host": "h",
                    "port": 2222
                }]})))
        machines = client.show_access("r-1")
        self.assertEqual(machines[0]["port"], 2222)

    def test_list_shapes_guarded(self):
        # A wrong-shaped payload is a protocol error, never a silently
        # empty list that reads as "no capacity".
        client, _ = _client(_tool_script(_mcp_result({"reservations": {}})))
        with self.assertRaises(api.OrnnMcpProtocolError):
            client.list_reservations()


class TestRefusals(unittest.TestCase):
    """A refusal body must reach the consumer verbatim and typed."""

    def _refusal(self, text):
        return _client(_tool_script(_mcp_result(text, is_error=True)))

    def test_image_required_is_typed_and_never_retried(self):
        client, transport = self._refusal("image_required")
        with self.assertRaises(api.OrnnImageBlockedError) as ctx:
            client.activate_access("r-1", ["k-1"])
        self.assertIn("image_required", str(ctx.exception))

        # Never retried: exactly one activate call hit the wire (the
        # form-encoded OAuth refresh payload is not JSON-RPC — skip it).
        def _jsonrpc(call):
            try:
                body = json.loads(call["payload"])
            except (ValueError, TypeError):
                return None
            return body if isinstance(body, dict) else None

        activates = [
            c for c in transport.calls
            if (_jsonrpc(c) or {}).get("method") == "tools/call" and
            (_jsonrpc(c) or {}
            ).get("params", {}).get("name") == "ornn_access_activate"
        ]
        self.assertEqual(len(activates), 1)

    def test_not_found_is_typed(self):
        client, _ = self._refusal("not_found")
        with self.assertRaises(api.OrnnNotFoundError):
            client.show_reservation("r-gone")

    def test_input_validation_is_loud(self):
        client, _ = self._refusal(
            "Input validation error: gpuSlug: expected string")
        with self.assertRaises(api.OrnnInputError) as ctx:
            client.get_spot_book("bad-slug")
        self.assertIn("expected string", str(ctx.exception))

    def test_unmapped_body_preserved_verbatim(self):
        # Funding refusals, cli_route_not_allowed and every unmapped body
        # stay distinguishable verbatim.
        client, _ = self._refusal("insufficient_credits")
        with self.assertRaises(api.OrnnError) as ctx:
            client.place_spot_bid("nvidia_rtx_pro_6000", 1, confirm=True)
        self.assertIn("insufficient_credits", str(ctx.exception))

    def test_jsonrpc_error_level_is_protocol_error(self):
        # The JSON-RPC envelope's own error (below the tool's isError)
        # carries the provider body verbatim, typed.
        client, _ = _client([
            _token_response(),
            (200,
             json.dumps({
                 "jsonrpc": "2.0",
                 "id": 1,
                 "error": {
                     "code": -32000,
                     "message": "cli_route_not_allowed"
                 },
             }).encode("utf-8")),
        ])
        with self.assertRaises(api.OrnnMcpProtocolError) as ctx:
            client.list_spot_books()
        self.assertIn("cli_route_not_allowed", str(ctx.exception))


class TestHTTPLayer(unittest.TestCase):

    def test_401_is_auth_error_without_token(self):
        client, _ = _client(_tool_script((401, b'{"error": "invalid"}')))
        with self.assertRaises(api.OrnnAuthError) as ctx:
            client.list_spot_books()
        self.assertNotIn("Bearer", str(ctx.exception))
        self.assertNotIn("acc-1", str(ctx.exception))

    def test_429_backs_off_bounded_then_raises(self):
        client, _ = _client(_tool_script(*[(429, b"slow down")] * 4),
                            sleep=mock.Mock())
        with self.assertRaises(api.OrnnRateLimitedError):
            client.list_spot_books()
        # Three bounded backoffs (no documented limit on the MCP surface),
        # then a loud failure — never a silent empty answer.
        self.assertEqual(client._sleep.call_count, api.MAX_RATE_LIMIT_RETRIES)
        for call in client._sleep.call_args_list:
            self.assertLessEqual(call.args[0], api.MAX_RETRY_AFTER_S)

    def test_empty_body_on_500_still_raises(self):
        # A 5xx with an empty body must raise, never read as "done" while
        # the server keeps billing (CodeRabbit on the Latitude PR).
        client, _ = _client(_tool_script((500, b"")))
        with self.assertRaises(api.OrnnError):
            client.list_spot_books()

    def test_non_json_success_body_is_protocol_error(self):
        client, _ = _client(_tool_script((200, b"<html>gateway</html>")))
        with self.assertRaises(api.OrnnError):
            client.list_spot_books()


class TestWaitUntilReady(unittest.TestCase):

    def _poll_pair(self, reservation_status="active", machines=None):
        return [
            _mcp_result({
                "reservationId": "r1",
                "status": reservation_status
            }),
            _mcp_result({"machines": [] if machines is None else machines}),
        ]

    def test_ready_when_machines_materialize(self):
        script = _tool_script(
            *(self._poll_pair("active") +
              self._poll_pair("active", machines=[{
                  "host": "h",
                  "port": 22
              }])))
        client, _ = _client(script)
        reservation = client.wait_until_ready("r1",
                                              timeout_s=60,
                                              poll_interval_s=5,
                                              sleep=lambda s: None)
        self.assertEqual(reservation["status"], "active")

    def test_cancelled_reservation_fails_closed(self):
        client, _ = _client(_tool_script(*self._poll_pair("cancelled")))
        with self.assertRaises(api.OrnnCapacityRefusedError):
            client.wait_until_ready("r1",
                                    timeout_s=60,
                                    poll_interval_s=5,
                                    sleep=lambda s: None)

    def test_completed_without_machines_fails_closed(self):
        # The 2026-10-08 wedge class: completed + empty access records is a
        # dead reservation, never an endless wait.
        client, _ = _client(
            _tool_script(*self._poll_pair("completed", machines=None)))
        with self.assertRaises(api.OrnnError) as ctx:
            client.wait_until_ready("r1",
                                    timeout_s=60,
                                    poll_interval_s=5,
                                    sleep=lambda s: None)
        self.assertIn("wedge", str(ctx.exception).lower())

    def test_deadline_fails_closed(self):
        client, _ = _client(_tool_script(*self._poll_pair("active") * 10))
        clock = {"t": 0.0}
        with self.assertRaises(api.OrnnError) as ctx:
            client.wait_until_ready(
                "r1",
                timeout_s=30,
                poll_interval_s=5,
                sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                now=lambda: clock["t"])
        self.assertIn("refusing to wait longer", str(ctx.exception))


class TestMachineEndpoint(unittest.TestCase):

    def test_host_port_parsed(self):
        host, port = api.OrnnClient.machine_endpoint({
            "host": "gpu-1.ornn.com",
            "port": 2222
        })
        self.assertEqual(host, "gpu-1.ornn.com")
        self.assertEqual(port, 2222)

    def test_hostless_entry_is_protocol_error(self):
        with self.assertRaises(api.OrnnMcpProtocolError):
            api.OrnnClient.machine_endpoint({"port": 2222})


if __name__ == "__main__":
    unittest.main()
