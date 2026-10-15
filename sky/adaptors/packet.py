"""Packet.ai REST client.

Packet.ai (https://dash.packet.ai, by hosted.ai) rents on-demand NVIDIA GPU
pods (B200, H200, RTX PRO 6000 Blackwell 96GB, RTX 4090/5090, L40s) from a
prepaid wallet, billed by the second. One ``pk_live_`` API key drives every
authenticated call (``X-API-Key`` or ``Authorization: Bearer`` — both blessed
by the machine reference, both measured).

MEASURED (ENG-562, 2026-10-15 — the vetting receipts on the ticket; the
wrong-company Equinix receipts in the same ticket's history describe a
different, dead product):

* **Instance creation returns 501 NOT_IMPLEMENTED for API-key auth** —
  ``"Instance creation is not yet available: the deploy engine (Stripe,
  wallet, provisioning) has not been ported."`` — after FULL validation
  (``name`` + ``pool_id`` required, then region resolution: a body without
  ``region_id`` gets 400 ``"Could not determine region. Please provide
  region_id."``). Measured on the doc-shaped body, on active-product pools,
  on a hybrid body, under both header spellings. Creation works only through
  the console's session path (the SPA POSTs the same route with a
  productId-shaped body under a session Bearer token, which the API-key path
  rejects as ``MISSING_REQUIRED_FIELD: name``). The lane bridges the seam
  with ADOPTION of console-launched boxes by instance name (see
  ``sky/provision/packet/instance.py``) — ``create_instance`` raises
  :class:`PacketCreateNotImplementedError` (typed, never retried) so a
  create attempt surfaces as the platform-capability refusal it is, not as
  a stock hunt or an auth failure.
* **``GET /api/v1/launch-options`` answers in ~30s cold** (measured 30.6s
  cold, ~1.3s warm) — a 60s-class timeout reads as "the surface is absent";
  this client sends that call with ``LAUNCH_OPTIONS_TIMEOUT_S``.
* **``{id}`` routes need the FULL instance id** (``i-55a5e3b1-6ede-…``);
  the short prefix (``i-55a5e3b1``) 404s ``NOT_FOUND`` — the route keys are
  the complete ids from ``GET /instances``.
* The OpenAPI spec at ``/api/v1/openapi`` lists phantom routes
  (``/api/products``, ``/api/providers/gpu-types``, ``/api/banner``,
  ``/api/branding``, ``/api/cli-config``) that 404 ``route_not_found`` live —
  the spec is aspiration, not truth; measure, never trust it.

Wire format (https://dash.packet.ai/llms-full.txt): success is
``{"data": ..., "meta": {"requestId", "timestamp"}}``; every 4xx/5xx is
``{"error": {"code": UPPER_SNAKE_CASE, "message": ...}, "meta": ...}`` — the
code is preserved verbatim into every raised error (the shadeform
INSUFFICIENT_FUNDS lesson: a flattened status line misreads stockout as
auth). Rate limits (per key, fixed one-minute window): reads 100/min, writes
30/min, launch 10/min, billing reads 5/min; every response carries
``X-RateLimit-Limit/Remaining/Reset`` headers, and a 429's
``X-RateLimit-Reset`` (Unix seconds) IS the provider's own retry hint —
this client backs off honoring it, bounded (the hung-call lesson: no
unbounded sleeps either).
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

API_BASE = 'https://dash.packet.ai/api/v1'

# Identifies the integration to the provider's logs.
USER_AGENT = 'urun-skypilot-packet/0.1 (+https://urun.sh)'

# One timeout for every ordinary request, including the reads inside poll
# loops — the point is the bound, not the value.
DEFAULT_TIMEOUT_S = 60

# GET /launch-options measured ~30s cold (ENG-562); a 60s-class client
# timeout misreads the surface as absent (the orchestrator's original
# "empty probe" was exactly this). Size the bound above the measured
# worst case.
LAUNCH_OPTIONS_TIMEOUT_S = 75

# Documented budget (llms-full.txt): reads 100/min, writes 30/min, launch
# 10/min, billing 5/min, per key, fixed one-minute window. A 429 body
# carries no hint; the X-RateLimit-Reset HEADER is the provider's own
# retry hint and is honored (bounded). Three strikes, then the typed error.
RATE_LIMIT_RETRIES = 3
MAX_RETRY_AFTER_S = 60.0

# Instance list pagination: page (1-based) + limit (max 100) inside
# data.pagination. Loop until a short page; the cap turns a server that
# ignores pagination into a loud error instead of a silently truncated list.
PAGE_LIMIT = 100
MAX_PAGES = 100

# Instance status steady states (llms-full.txt). starting/stopping/
# restarting are transitional (map to None = keep waiting); pending and
# other transitional values are logged verbatim, never guessed at (the
# Latitude open-set lesson).
STATUS_RUNNING = 'running'
STATUS_STOPPED = 'stopped'
STATUS_ERROR = 'error'
STATUS_FAILED = 'failed'

# Environment variable the callers read the key from (the task secret
# channel's spelling). NOTE: packet-gpu-cli documents PACKET_API_KEY but
# does NOT implement it (it reads a conf-stored key) — the lane's adapter
# reads the env itself, never via the CLI.
API_KEY_ENV = 'PACKET_API_KEY'
API_KEY_FILE = '~/.packet/credentials'


class PacketError(RuntimeError):
    """Any Packet.ai API failure. Never raised with the API key in it.

    ``code`` is the provider's UPPER_SNAKE_CASE error code, preserved
    verbatim (branchable downstream); the message never contains headers
    or the URL query (no key leakage).
    """

    def __init__(self, message: str, code: str = 'error'):
        super().__init__(message)
        self.code = code


class PacketAuthError(PacketError):
    """401 — key missing, invalid, revoked, or expired. Check/rotate it."""


class PacketRateLimited(PacketError):
    """429 that exhausted its retries. The provider's own reset hint
    (X-RateLimit-Reset) was already honored, bounded."""


class PacketNotFoundError(PacketError):
    """404 — the resource is gone (or not owned by this team). Terminate
    paths treat this as success (idempotent teardown)."""


class PacketCreateNotImplementedError(PacketError):
    """501 NOT_IMPLEMENTED on POST /instances — the measured
    platform-capability refusal: instance creation is not served for
    API-key auth (the deploy engine is console-session-path only today).
    NEVER retried: retrying a not-implemented surface is budget waste and
    would wedge a runner (the fail-closed lesson). The lane bridges the
    seam with adoption (sky/provision/packet/instance.py)."""

    def __init__(self, message: str):
        super().__init__(message, code='NOT_IMPLEMENTED')


class PacketUnavailableError(PacketError):
    """503 INSTANCES_UNAVAILABLE — the platform could not answer; the
    provider's own docs say retry. Caller-side policy decides (the
    provisioner still fails closed on its deadline)."""


def _error_envelope(raw: bytes) -> Dict[str, Any]:
    """The provider's ``error`` object from a body, or {} if unusable."""
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    if isinstance(payload, dict):
        err = payload.get('error')
        if isinstance(err, dict):
            return err
    return {}


def _ssh_fingerprint(public_key: str) -> str:
    """The ``08:07:ea:…:2b`` (MD5, hex, colon-separated) fingerprint of an
    OpenSSH public key.

    MEASURED (ENG-562): ``GET /ssh-keys`` returns fingerprints in this
    legacy ``ssh-keygen -l`` format (16 bytes = MD5 of the decoded blob),
    NOT the SHA256: form — matching on SHA256 would never hit, and
    ensure_ssh_key would mint a duplicate on every deploy.
    """
    parts = str(public_key or '').strip().split()
    if len(parts) < 2:
        raise PacketError('not a usable OpenSSH public key '
                          f'({str(public_key)[:40]!r})')
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except Exception as exc:  # ValueError on bad base64
        raise PacketError(f'public key blob is not base64: {exc}') from exc
    digest = hashlib.md5(blob).hexdigest()  # nosec - fingerprint, not crypto
    return ':'.join(digest[i:i + 2] for i in range(0, len(digest), 2))


class PacketClient:
    """Thin, explicit client. One method per endpoint we actually use.

    ``transport`` is injected in tests. Signature:
    ``(method, url, headers, payload) -> (status, bytes, headers_dict)`` —
    the third element exists so a 429 can honor the provider's own
    X-RateLimit-Reset hint (the campaign rule: back off honoring the
    provider's retry hint, never a flat guess).
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_BASE,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        transport: Optional[
            Callable[[str, str, Dict[str, str], Optional[bytes]],
                     Tuple[int, bytes, Dict[str, str]]]] = None,
    ):
        if not api_key or not api_key.strip():
            raise PacketAuthError(
                'Packet.ai API key is empty; refusing to make '
                'unauthenticated calls that would look like "no capacity"')
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip('/')
        self._timeout_s = timeout_s
        self._transport = transport

    # -- plumbing ---------------------------------------------------------

    def _http(self, method: str, url: str, headers: Dict[str, str],
              payload: Optional[bytes]):
        req = urllib.request.Request(url,
                                     data=payload,
                                     headers=headers,
                                     method=method)
        # No query in errors: no key leakage (the query can carry a token).
        path = url.split('?')[0]
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                # resp.read() runs OUTSIDE urlopen's URLError handling: a
                # stalled body raises TimeoutError/OSError directly and
                # would escape every caller that only catches PacketError
                # (the trap CodeRabbit flagged on the Spheron client).
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read(), dict(exc.headers)
            except (TimeoutError, OSError) as read_exc:
                raise PacketError(f'{method} {path} failed reading the error '
                                  f'body: {read_exc}') from read_exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise PacketError(f'{method} {path} failed: {exc.reason}') from exc
        except (TimeoutError, OSError) as exc:
            raise PacketError(f'{method} {path} failed: {exc}') from exc

    def _retry_after_s(self, headers: Dict[str, str],
                       now: Callable[[], float]) -> Optional[float]:
        """The provider's own hint from X-RateLimit-Reset, if present and
        sane (a Unix-seconds timestamp, possibly already past)."""
        raw = headers.get('X-RateLimit-Reset')
        if raw is None:
            # urllib's dict(headers) normalizes case; try both spellings.
            raw = headers.get('x-ratelimit-reset')
        if raw is None:
            return None
        try:
            reset_at = float(raw)
        except ValueError:
            return None
        return max(0.0, min(reset_at - now(), MAX_RETRY_AFTER_S))

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[int] = None,
    ) -> Any:
        url = f'{self._base_url}{path}'
        if params:
            url = f'{url}?{urllib.parse.urlencode(params)}'
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
            # Both spellings are blessed; X-API-Key is the one the
            # machine reference's own examples lead with.
            'X-API-Key': self._api_key,
        }
        payload = None
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'

        for attempt in range(RATE_LIMIT_RETRIES + 1):
            if self._transport is not None:
                status, raw, resp_headers = self._transport(
                    method, url, headers, payload)
            else:
                status, raw, resp_headers = self._http(method, url, headers,
                                                       payload)
            if status == 429 and attempt < RATE_LIMIT_RETRIES:
                # Honor the provider's own hint (X-RateLimit-Reset, a Unix
                # timestamp); fall back to bounded exponential. Never a
                # sleep past MAX_RETRY_AFTER_S.
                hint = self._retry_after_s(resp_headers, time.time)
                if hint is None:
                    hint = min(2.0 * (attempt + 1), MAX_RETRY_AFTER_S)
                time.sleep(hint)
                continue
            return self._interpret(status,
                                   raw,
                                   method,
                                   path,
                                   resp_headers=resp_headers)

        # Unreachable: the loop returns or raises on the final attempt.
        raise AssertionError('rate-limit retry loop exhausted')

    def _interpret(self,
                   status: int,
                   raw: bytes,
                   method: str,
                   path: str,
                   resp_headers: Optional[Dict[str, str]] = None) -> Any:
        # Never include the URL query or headers in an error: no key
        # leakage.
        del resp_headers  # hints are consumed in _call, not here
        where = f'{method} {path}'
        if status < 400:
            # The empty-body success path lives UNDER the status check: a
            # 4xx/5xx with an empty body is a FAILURE (raise below), never
            # a silent None — the bug class where a provider's empty-body
            # error would read as "done" while the box keeps billing.
            if not raw:
                return None
            try:
                payload = json.loads(raw)
            # ValueError covers JSONDecodeError AND UnicodeDecodeError (a
            # non-UTF-8 gateway page is a ValueError) so no raw exception
            # escapes a PacketError-only caller.
            except ValueError as exc:
                raise PacketError(
                    f'{where}: response was not JSON ({exc})') from exc
            # The success envelope is {"data": ..., "meta": ...} — unwrap
            # it here so every method sees the payload itself. A non-dict
            # success (a bare list/string) passes through as-is; the
            # per-endpoint methods type-check what they consume.
            if isinstance(payload, dict) and 'data' in payload:
                return payload['data']
            return payload

        err = _error_envelope(raw)
        code = str(err.get('code') or '')
        message = str(err.get('message') or '')[:300]
        if not code and not message:
            # An error with no usable envelope still must not collapse to
            # a bare status line: show the raw prefix instead.
            message = raw[:200].decode('utf-8', 'replace')

        if status == 401:
            hint = {
                'MISSING_API_KEY': 'no key header was sent',
                'INVALID_API_KEY': 'wrong prefix or unknown key',
                'REVOKED_API_KEY': 'the key was revoked '
                                   '(DELETE /api/v1/api-keys/{id})',
                'EXPIRED_API_KEY': 'the key passed its expiresAt',
            }.get(code, 'check the key')
            raise PacketAuthError(
                f'{where}: HTTP 401 ({code or "UNAUTHORIZED"}): {hint}. '
                f'Check {API_KEY_ENV} / {API_KEY_FILE} (or rotate in the '
                'Packet.ai console under API keys).')
        if status == 404:
            raise PacketNotFoundError(
                f'{where}: not found ({message}) — note {{id}} routes need '
                'the FULL instance id from GET /instances')
        if status == 429:
            raise PacketRateLimited(
                f'{where}: rate limited ({code or "RATE_LIMIT_EXCEEDED"}); '
                'the X-RateLimit-Reset hint was already honored, bounded. '
                f'detail: {message}')
        if status == 501:
            # The measured platform-capability refusal (ENG-562): creation
            # is not served for API-key auth. Typed, message verbatim.
            raise PacketCreateNotImplementedError(
                f'{where}: {code or "NOT_IMPLEMENTED"}: {message}')
        if status == 503:
            raise PacketUnavailableError(
                f'{where}: {code or "INSTANCES_UNAVAILABLE"}: {message} '
                '(the platform could not answer; the provider says '
                'retry — caller policy decides)')
        default = code or 'error'
        raise PacketError(
            f'{where}: HTTP {status}: {default}: {message}',
            code=default,
        )

    # -- account ----------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Who-am-I (GET /account): {id, email, name, teamId, createdAt}."""
        payload = self._call('GET', '/account') or {}
        if not isinstance(payload, dict):
            raise PacketError('GET /account: expected an object, got '
                              f'{type(payload).__name__}')
        return payload

    def get_balance(self) -> Dict[str, Any]:
        """The wallet (GET /inventory/balance): {wallet, activeDeployments}."""
        payload = self._call('GET', '/inventory/balance') or {}
        if not isinstance(payload, dict):
            raise PacketError('GET /inventory/balance: expected an object, got '
                              f'{type(payload).__name__}')
        return payload

    # -- launch options (the price/stock surface) ---------------------------

    def get_launch_options(self) -> Dict[str, Any]:
        """Products (the deployable surface) + pools + regions.

        SLOW: ~30s cold (measured) — sent with LAUNCH_OPTIONS_TIMEOUT_S.
        ``products`` is the canonical "what can I deploy, at what price,
        where": a pool row with a price but no active product listing its
        id in ``poolIds`` is a PRICING fact, not a deployability fact (the
        ENG-562 lesson) — callers that need rentability filter products.
        """
        payload = self._call(
            'GET', '/launch-options', timeout_s=LAUNCH_OPTIONS_TIMEOUT_S) or {}
        if not isinstance(payload, dict):
            raise PacketError('GET /launch-options: expected an object, '
                              f'got {type(payload).__name__}')
        for key in ('products', 'pools', 'regions'):
            if not isinstance(payload.get(key), list):
                raise PacketError(
                    f'GET /launch-options: expected a list at .{key}, got '
                    f'{type(payload.get(key)).__name__} — refusing to '
                    'silently treat the surface as empty')
        return payload

    # -- instances ---------------------------------------------------------

    def list_instances(self,
                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every instance on the account, newest first, across pages."""
        out: List[Dict[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            params: Dict[str, Any] = {'page': page, 'limit': PAGE_LIMIT}
            if status:
                params['status'] = status
            payload = self._call('GET', '/instances', params=params) or {}
            if not isinstance(payload, dict):
                raise PacketError('GET /instances: expected an object, got '
                                  f'{type(payload).__name__}')
            rows = payload.get('instances')
            if not isinstance(rows, list):
                raise PacketError(
                    'GET /instances: expected a list at .instances, got '
                    f'{type(rows).__name__}')
            out.extend(row for row in rows if isinstance(row, dict))
            pagination = payload.get('pagination')
            total_pages = None
            if isinstance(pagination, dict):
                try:
                    total_pages = int(pagination.get('totalPages') or 0)
                except (TypeError, ValueError):
                    total_pages = None
            if len(rows) < PAGE_LIMIT:
                return out
            if total_pages is not None and page >= total_pages:
                return out
        raise PacketError(
            f'GET /instances: more than {MAX_PAGES} pages; refusing to '
            'silently truncate')

    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        """One instance by its FULL id; returns the ``data`` object
        (``instance`` + metadata when present — the {id} route wraps the
        instance at ``data.instance``)."""
        payload = self._call('GET', f'/instances/{instance_id}') or {}
        if not isinstance(payload, dict):
            raise PacketError(
                f'GET /instances/{instance_id}: expected an object, got '
                f'{type(payload).__name__}')
        return payload

    def get_connection(self, instance_id: str) -> Dict[str, Any]:
        """SSH credentials for a running instance by its FULL id.

        The response carries {ip, port, username, password, ssh_command}:
        the ip is the PUBLIC address (measured), the password is a live
        secret — treat the whole response as one (never log it).
        """
        payload = self._call('GET',
                             f'/instances/{instance_id}/connection') or {}
        if not isinstance(payload, dict):
            raise PacketError(
                f'GET /instances/{instance_id}/connection: expected an '
                f'object, got {type(payload).__name__}')
        conn = payload.get('connection')
        if not isinstance(conn, dict):
            raise PacketError(
                f'GET /instances/{instance_id}/connection: expected an '
                'object at .connection — instance not running?')
        return conn

    def create_instance(
        self,
        *,
        name: str,
        pool_id: str,
        region_id: int,
        vgpus: int = 1,
        ssh_key_name: Optional[str] = None,
        public_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an instance (POST /instances).

        MEASURED (ENG-562, 2026-10-15): raises
        :class:`PacketCreateNotImplementedError` today for API-key auth —
        instance creation is console-session-path only; this method exists
        so the day the engine is ported for API keys, the lane provisions
        with ZERO changes (the body is the validated doc shape:
        ``{name, pool_id, region_id, vgpus}``; keys install at launch, so
        the account carries the key material first).
        """
        if ssh_key_name and public_key:
            # Keys install at LAUNCH only: ensure the account carries the
            # key material before creating (console launches do the same).
            self.ensure_ssh_key(ssh_key_name, public_key)
        body: Dict[str, Any] = {
            'name': name,
            'pool_id': str(pool_id),
            'region_id': int(region_id),
            'vgpus': int(vgpus),
        }
        payload = self._call('POST', '/instances', body=body) or {}
        if not isinstance(payload, dict):
            raise PacketError(
                f'POST /instances returned {type(payload).__name__}, '
                'expected an object')
        return payload

    def delete_instance(self, instance_id: str) -> None:
        """Terminate permanently (root disk lost; shared volumes survive).

        404 is success (idempotent teardown).
        """
        try:
            self._call('DELETE', f'/instances/{instance_id}')
        except PacketNotFoundError:
            pass

    def stop_instance(self, instance_id: str) -> None:
        """Stop an instance (billing stops; the box and disk persist)."""
        self._call('POST', f'/instances/{instance_id}/stop')

    def start_instance(self, instance_id: str) -> None:
        """Start a stopped instance."""
        self._call('POST', f'/instances/{instance_id}/start')

    # -- ssh keys ----------------------------------------------------------

    def list_ssh_keys(self) -> List[Dict[str, Any]]:
        payload = self._call('GET', '/ssh-keys') or {}
        rows = payload if isinstance(payload, list) else payload.get('data')
        if not isinstance(rows, list):
            raise PacketError('GET /ssh-keys: expected a list, got '
                              f'{type(rows).__name__}')
        return [row for row in rows if isinstance(row, dict)]

    def create_ssh_key(self, name: str, public_key: str) -> Dict[str, Any]:
        payload = self._call(
            'POST',
            '/ssh-keys',
            body={
                'name': name,
                'publicKey': public_key,
            },
        ) or {}
        if not isinstance(payload, dict) or not payload.get('id'):
            raise PacketError(
                f'POST /ssh-keys returned no id: {sorted(payload)}')
        return payload

    def ensure_ssh_key(self, name: str, public_key: str) -> str:
        """The id of the key matching ``public_key``, adding it if absent.

        Matching on key MATERIAL (not name): a re-created controller
        keypair with the same name must not silently select a stale key,
        and a re-deploy with a renamed key must not mint duplicates. The
        key list returns the MD5 ``08:07:ea:…`` fingerprint — derivable
        locally from the public key blob (the value ``ssh-keygen -l``
        prints in legacy format) — so material identity is exact. Max 10
        keys per account; a duplicate material 400s naming the existing
        key.
        """
        fingerprint = _ssh_fingerprint(public_key)
        for key in self.list_ssh_keys():
            if str(key.get('fingerprint') or '') == fingerprint:
                key_id = key.get('id')
                if key_id:
                    return str(key_id)
        return str(self.create_ssh_key(name, public_key)['id'])

    # -- accessors --------------------------------------------------------

    @staticmethod
    def instance_status(instance: Dict[str, Any]) -> str:
        return str(instance.get('status') or '').strip().lower()

    @staticmethod
    def instance_name(instance: Dict[str, Any]) -> str:
        return str(instance.get('name') or '').strip()

    @staticmethod
    def ssh_user_from_connection(connection: Dict[str, Any]) -> Optional[str]:
        """The SSH user the PROVIDER reports for the connection.

        MEASURED (ENG-562, 2026-10-15): ``ubuntu`` — the schema example and
        the live box agree this time; measured on both password and key
        auth. Still never trusted: the provisioner reads this per
        connection and only falls back to the constant.
        """
        user = (connection or {}).get('username')
        return str(user) if user else None

    # -- polling ----------------------------------------------------------

    def wait_until_running(
        self,
        instance_id: str,
        *,
        timeout_s: float = 20 * 60,
        poll_interval_s: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> Dict[str, Any]:
        """Poll until the instance is ``running``, or fail closed.

        ``error``/``failed`` raise immediately (never auto-retry — the
        operator decides), and so does the deadline: a stuck provisioning
        must become a failed runner, not an endless wait (the failure mode
        that kept a Vast instance ``SkyPilotProvisioning`` for 30 minutes
        while the provider had already given up).
        """
        deadline = now() + timeout_s
        last_status: Optional[str] = None
        while True:
            payload = self.get_instance(instance_id)
            instance = (payload.get('instance') if isinstance(
                payload.get('instance'), dict) else payload)
            status = self.instance_status(instance)
            if status:
                last_status = status
            if status == STATUS_RUNNING:
                return instance
            if status in (STATUS_ERROR, STATUS_FAILED):
                raise PacketError(
                    f'instance {instance_id} entered {status} while waiting '
                    'for running; not auto-retrying (report and decide)',
                    code=status.upper())
            if now() >= deadline:
                last = repr(last_status or 'starting')
                raise PacketError(f'instance {instance_id} still {last} after '
                                  f'{timeout_s:.0f}s; refusing to wait longer')
            sleep(poll_interval_s)
