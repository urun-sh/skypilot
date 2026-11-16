"""Verda REST client.

Verda (https://verda.com — the rebranded DataCrunch cloud, Helsinki-based)
rents on-demand GPU VMs against a project balance: instances are deployed
from ``POST /v1/instances`` with an instance-type code (e.g.
``1H100.80S.22V``), an OS image id, and SSH key ids, and are reached over
direct SSH.

Wire protocol notes, all measured from the OpenAPI 3.1 spec at
https://api.verda.com (spec reviewed 2026-11-16; path references below are
verbatim from it):

* **OAuth2 client credentials, not API keys.** ``POST /v1/oauth2/token``
  with ``{"grant_type": "client_credentials", "client_id": ...,
  "client_secret": ...}`` returns ``{access_token, token_type: "Bearer",
  expires_in: 3600, refresh_token, scope}`` (GetAccessTokenResponseDto).
  The token is cached and re-minted when it expires (a 60s safety margin
  covers clock skew); a 401 mid-flight triggers exactly ONE re-mint and
  retry, never a loop.
* **Errors are a flat envelope** ``{"code": ..., "message": ...}`` with a
  CLOSED code enum (PublicApiErrorResponseDto): ``invalid_request,
  unauthorized_request, insufficient_funds, forbidden_action, not_found,
  conflict, server_error, service_unavailable``. The envelope is
  preserved into the raised error verbatim.
* **403 ``insufficient_funds`` is a CONFIG error, not capacity** — the
  same lesson as QuantaCloud's 402: a lane that mistakes a billing
  failure for a stockout hunts forever.
* **The instance-availability list is the ONLY stock source.** The
  instance-types list carries per-location prices, but the spec says
  explicitly: "A price does not mean the type is available there."
* **Pricing fields are STRINGS** (``"2.90"``) and the authoritative rows
  are ``location_prices``; the top-level ``price_per_hour``/``spot_price``
  are deprecated and are never read here.
* **The deprecated action endpoint is used for delete** — the spec has no
  ``DELETE /v1/instances/{id}``; teardown is
  ``POST /v1/instances/action`` with ``{"action": "delete", "id": ...}``
  (PerformInstanceActionPublicDto).

Every call carries an explicit timeout, and non-JSON or non-dict payloads
raise typed errors rather than leaking AttributeError/UnicodeDecodeError
(the trap the Spheron client taught us). No live credentials exist yet
(VERDA_CLIENT_ID/VERDA_CLIENT_SECRET arrive later): everything here is
built against fixtures derived from the OpenAPI examples.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

API_BASE = 'https://api.verda.com'

# Identifies the integration to the provider's logs.
USER_AGENT = 'urun-skypilot-verda/0.1 (+https://urun.sh)'

# One timeout for every request, including the reads inside poll loops —
# the point is the bound, not the value.
DEFAULT_TIMEOUT_S = 60

# 429 bodies carry no retry hint (PublicApiErrorResponseDto has no retry
# field); back off exponentially, bounded, three strikes.
MAX_RATE_LIMIT_RETRIES = 3
MAX_RETRY_AFTER_S = 60.0

# Credentials: env first (the task secret channel), then the file store
# (JSON {"client_id": ..., "client_secret": ...}), mirroring the env-then-
# file resolution of the sibling lanes.
CLIENT_ID_ENV = 'VERDA_CLIENT_ID'
CLIENT_SECRET_ENV = 'VERDA_CLIENT_SECRET'
CREDENTIALS_FILE = '~/.verda/credentials'

# Instance status steady states (GetInstanceResponsePublicApiDto.status
# enum, measured from the OpenAPI spec 2026-11-16). provisioning/ordered/
# new/validating are transitional; running is the SSH-reachable state;
# error/installation_failed/no_capacity are terminal failures; offline/
# discontinued/notfound/deleting are terminal teardowns.
STATUS_RUNNING = 'running'
STATUS_PROVISIONING = 'provisioning'
STATUS_ERROR = 'error'
STATUS_NO_CAPACITY = 'no_capacity'
STATUS_INSTALLATION_FAILED = 'installation_failed'
STATUS_OFFLINE = 'offline'
STATUS_DISCONTINUED = 'discontinued'
STATUS_DELETING = 'deleting'
STATUS_NOTFOUND = 'notfound'
STATUS_UNKNOWN = 'unknown'
STATUS_NEW = 'new'
STATUS_ORDERED = 'ordered'
STATUS_VALIDATING = 'validating'

# Terminal failure states: wait_until_running raises immediately on them
# (never auto-retry — the operator decides).
_FAILURE_STATUSES = (STATUS_ERROR, STATUS_INSTALLATION_FAILED,
                     STATUS_NO_CAPACITY)
# Terminal teardown states: the box is gone or going; treat 404-ish.
_TERMINAL_TEARDOWN_STATUSES = (STATUS_DISCONTINUED, STATUS_DELETING,
                               STATUS_NOTFOUND)

# Re-mint the token this many seconds before expires_in says it lapses
# (covers clock skew between us and the provider).
TOKEN_EXPIRY_MARGIN_S = 60

# The deploy poll cadence: provisioning a GPU VM takes minutes; polling
# faster is budget waste (mirrors the quantacloud lane's floor).
MIN_POLL_INTERVAL_S = 10
DEFAULT_PROVISION_TIMEOUT_S = 20 * 60


class VerdaError(RuntimeError):
    """Any Verda API failure. Never raised with credentials in it."""

    def __init__(self, message: str, code: str = 'error'):
        super().__init__(message)
        self.code = code


class VerdaAuthError(VerdaError):
    """401 / unauthorized_request — credentials missing, invalid or
    revoked. Check/rotate them."""


class VerdaInsufficientFundsError(VerdaError):
    """403 insufficient_funds — the project balance cannot cover the
    spend. A CONFIG/billing failure, never a capacity signal."""

    def __init__(self, message: str, code: str = 'insufficient_funds'):
        super().__init__(message, code=code)


class VerdaRateLimited(VerdaError):
    """429 that exhausted its bounded backoff retries."""


class VerdaNotFoundError(VerdaError):
    """404 — the resource is gone. Teardown paths treat this as success."""


def _error_envelope(raw: bytes) -> Dict[str, Any]:
    """The provider's flat error object from a body, or {} if unusable."""
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get('code'), str):
        return payload
    return {}


class VerdaClient:
    """Thin, explicit client. One method per endpoint we actually use.

    ``transport`` is injected in tests. Signature:
    ``(method, url, headers, payload) -> (status, bytes)``.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        base_url: str = API_BASE,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        transport: Optional[Callable[
            [str, str, Dict[str, str], Optional[bytes]], Any]] = None,
    ):
        client_id = (client_id or '').strip()
        client_secret = (client_secret or '').strip()
        if not client_id or not client_secret:
            raise VerdaAuthError(
                'Verda client credentials are incomplete; refusing to '
                'make calls that would answer 401 and read as "no '
                'capacity". Check '
                f'{CLIENT_ID_ENV}/{CLIENT_SECRET_ENV} or '
                f'{CREDENTIALS_FILE}')
        self._client_id = client_id
        self._client_secret = client_secret
        self._base_url = base_url.rstrip('/')
        self._timeout_s = timeout_s
        self._transport = transport
        # Cached OAuth2 token: (access_token, expires_at_monotonic_s).
        self._token: Optional[tuple] = None

    # -- plumbing ---------------------------------------------------------

    def _http(self, method: str, url: str, headers: Dict[str, str],
              payload: Optional[bytes]):
        req = urllib.request.Request(url,
                                     data=payload,
                                     headers=headers,
                                     method=method)
        # No query in errors: no credential leakage (the token request
        # body carries the secret, never the URL).
        path = url.split('?')[0]
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            except (TimeoutError, OSError) as read_exc:
                raise VerdaError(f'{method} {path} failed reading the error '
                                 f'body: {read_exc}') from read_exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise VerdaError(f'{method} {path} failed: {exc.reason}') from exc
        except (TimeoutError, OSError) as exc:
            raise VerdaError(f'{method} {path} failed: {exc}') from exc

    def _request_once(self, method: str, url: str, headers: Dict[str, str],
                      payload: Optional[bytes]):
        if self._transport is not None:
            return self._transport(method, url, headers, payload)
        return self._http(method, url, headers, payload)

    def _interpret(self, status: int, raw: bytes, method: str, path: str):
        # Never include the URL query or headers in an error: no token
        # or secret leakage.
        where = f'{method} {path}'
        if status < 400:
            if not raw:
                return None
            try:
                return json.loads(raw)
            except ValueError as exc:
                raise VerdaError(
                    f'{where}: response was not JSON ({exc})') from exc

        err = _error_envelope(raw)
        code = str(err.get('code') or '')
        message = str(err.get('message') or '')[:300]
        if not code and not message:
            message = raw[:200].decode('utf-8', 'replace')

        if status == 401 or code == 'unauthorized_request':
            raise VerdaAuthError(
                f'{where}: HTTP {status} ({code}). Check '
                f'{CLIENT_ID_ENV}/{CLIENT_SECRET_ENV} (or rotate them in '
                'the Verda console).')
        if status == 403 and code == 'insufficient_funds':
            raise VerdaInsufficientFundsError(f'{where}: {code}: {message}')
        if status == 404 or code == 'not_found':
            raise VerdaNotFoundError(f'{where}: not found ({message})')
        if status == 429:
            raise VerdaRateLimited(
                f'{where}: rate limited ({code or "rate_limited"}); '
                f'detail: {message}')
        raise VerdaError(
            f'{where}: HTTP {status}: {code or "error"}: '
            f'{message}',
            code=code or 'error')

    def _raw_call(self,
                  method: str,
                  path: str,
                  *,
                  token: Optional[str],
                  body: Optional[Dict[str, Any]] = None) -> Any:
        """One HTTP round trip with NO auth retry (used by the token
        mint itself, which is public)."""
        url = f'{self._base_url}{path}'
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
        }
        payload = None
        if token is not None:
            headers['Authorization'] = f'Bearer {token}'
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            status, raw = self._request_once(method, url, headers, payload)
            if status == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                time.sleep(min(2.0 * (attempt + 1), MAX_RETRY_AFTER_S))
                continue
            return self._interpret(status, raw, method, path)
        raise AssertionError('rate-limit retry loop exhausted')  # unreachable

    # -- auth --------------------------------------------------------------

    def _mint_token(self) -> str:
        """POST /v1/oauth2/token (GetAccessTokenDto) -> cached Bearer."""
        payload = self._raw_call(
            'POST',
            '/v1/oauth2/token',
            token=None,
            body={
                'grant_type': 'client_credentials',
                'client_id': self._client_id,
                'client_secret': self._client_secret,
            },
        )
        if not isinstance(payload, dict) or not payload.get('access_token'):
            raise VerdaError(
                f'token response carried no access_token: {payload!r}')
        expires_in = float(payload.get('expires_in') or 3600)
        expires_at = time.monotonic() + max(expires_in - TOKEN_EXPIRY_MARGIN_S,
                                            0)
        token = str(payload['access_token'])
        self._token = (token, expires_at)
        return token

    def _access_token(self) -> str:
        token, expires_at = self._token or (None, 0.0)
        if token is None or time.monotonic() >= expires_at:
            return self._mint_token()
        return token

    def _call(self,
              method: str,
              path: str,
              body: Optional[Dict[str, Any]] = None) -> Any:
        """One authenticated call; a 401 re-mints the token exactly once."""
        try:
            return self._raw_call(method,
                                  path,
                                  token=self._access_token(),
                                  body=body)
        except VerdaAuthError:
            # The cached token may have lapsed between calls: force a
            # re-mint and retry exactly once (a second 401 is a real
            # credential problem and raises to the operator).
            self._token = None
            return self._raw_call(method,
                                  path,
                                  token=self._mint_token(),
                                  body=body)

    # -- account ----------------------------------------------------------

    def get_balance(self) -> Dict[str, Any]:
        """GET /v1/balance -> BalanceResponseDto {amount, currency}."""
        payload = self._call('GET', '/v1/balance') or {}
        if not isinstance(payload, dict):
            raise VerdaError('GET /v1/balance: expected an object, got '
                             f'{type(payload).__name__}')
        return payload

    # -- catalog ----------------------------------------------------------

    def list_instance_types(self) -> List[Dict[str, Any]]:
        """GET /v1/instance-types -> array of InstanceType.

        Authoritative pricing is ``location_prices`` (deprecated
        top-level price fields are never read here; see module docstring).
        """
        payload = self._call('GET', '/v1/instance-types')
        if not isinstance(payload, list):
            raise VerdaError('GET /v1/instance-types: expected an array, '
                             f'got {type(payload).__name__}')
        return [row for row in payload if isinstance(row, dict)]

    def list_locations(self) -> List[Dict[str, Any]]:
        """GET /v1/locations -> array of Location {code, name, country_code}."""
        payload = self._call('GET', '/v1/locations')
        if not isinstance(payload, list):
            raise VerdaError('GET /v1/locations: expected an array, got '
                             f'{type(payload).__name__}')
        return [row for row in payload if isinstance(row, dict)]

    def list_availability(self) -> List[Dict[str, Any]]:
        """GET /v1/instance-availability -> per-location in-stock codes.

        The ONLY stock source: a type is rentable in a location iff its
        code appears in that location's ``availabilities`` list.
        """
        payload = self._call('GET', '/v1/instance-availability')
        if not isinstance(payload, list):
            raise VerdaError('GET /v1/instance-availability: expected an '
                             f'array, got {type(payload).__name__}')
        return [row for row in payload if isinstance(row, dict)]

    # -- ssh keys ---------------------------------------------------------

    def list_ssh_keys(self) -> List[Dict[str, Any]]:
        """GET /v1/ssh-keys -> array of GetKeysResponseDto (with `key`
        MATERIAL and an MD5 `fingerprint`)."""
        payload = self._call('GET', '/v1/ssh-keys')
        if not isinstance(payload, list):
            raise VerdaError('GET /v1/ssh-keys: expected an array, got '
                             f'{type(payload).__name__}')
        return [row for row in payload if isinstance(row, dict)]

    def create_ssh_key(self, name: str, key: str) -> str:
        """POST /v1/ssh-keys (AddKeyDto {name, key}) -> the new key id."""
        payload = self._call('POST',
                             '/v1/ssh-keys',
                             body={
                                 'name': name,
                                 'key': key
                             })
        if not isinstance(payload, str) or not payload:
            # The spec's 201 example is the bare key UUID string.
            raise VerdaError(f'POST /v1/ssh-keys returned no id: {payload!r}')
        return payload

    def ensure_ssh_key(self, name: str, public_key: str) -> str:
        """The id of the key matching ``public_key`` material, or a new one.

        The key list returns the MATERIAL itself (``key`` field), so
        matching is exact string equality on the stripped key — no
        fingerprint derivation needed (unlike QuantaCloud).
        """
        material = public_key.strip()
        for key_row in self.list_ssh_keys():
            if str(key_row.get('key') or '').strip() == material:
                key_id = key_row.get('id')
                if key_id:
                    return str(key_id)
        return self.create_ssh_key(name, material)

    # -- instances ---------------------------------------------------------

    def create_instance(
        self,
        *,
        instance_type: str,
        image: str,
        hostname: str,
        ssh_key_ids: List[str],
        location_code: str,
        is_spot: bool = False,
    ) -> Dict[str, Any]:
        """POST /v1/instances (DeployInstancePublicDto) -> the instance.

        ``hostname`` is the only REQUIRED body field per the schema; the
        lane always passes instance_type/image/ssh_key_ids/location_code
        explicitly so the deploy is fully determined.
        """
        body: Dict[str, Any] = {
            'hostname': hostname,
            'instance_type': instance_type,
            'image': image,
            'ssh_key_ids': ssh_key_ids,
            'location_code': location_code,
            'is_spot': is_spot,
        }
        payload = self._call('POST', '/v1/instances', body=body) or {}
        if not isinstance(payload, dict) or not payload.get('id'):
            raise VerdaError(f'POST /v1/instances returned no id: {payload!r}')
        return payload

    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        """GET /v1/instances/{id} -> GetInstanceResponsePublicApiDto."""
        payload = self._call('GET', f'/v1/instances/{instance_id}') or {}
        if not isinstance(payload, dict):
            raise VerdaError(f'GET /v1/instances/{instance_id}: expected an '
                             f'object, got {type(payload).__name__}')
        return payload

    def list_instances(self) -> List[Dict[str, Any]]:
        """GET /v1/instances -> every instance on the project."""
        payload = self._call('GET', '/v1/instances')
        if not isinstance(payload, list):
            raise VerdaError('GET /v1/instances: expected an array, got '
                             f'{type(payload).__name__}')
        return [row for row in payload if isinstance(row, dict)]

    def delete_instance(self, instance_id: str) -> None:
        """Teardown via the deprecated action endpoint (the spec has no
        DELETE /v1/instances/{id}): POST /v1/instances/action with
        ``{"action": "delete", "id": <uuid>}``. 404 is success
        (idempotent teardown)."""
        try:
            self._call('POST',
                       '/v1/instances/action',
                       body={
                           'action': 'delete',
                           'id': instance_id
                       })
        except VerdaNotFoundError:
            pass

    # -- accessors ---------------------------------------------------------

    @staticmethod
    def instance_status(instance: Dict[str, Any]) -> str:
        return str(instance.get('status') or '').strip().lower()

    @staticmethod
    def public_ip(instance: Dict[str, Any]) -> Optional[str]:
        """The public address; ``ip`` is nullable until the box is up."""
        ip = instance.get('ip')
        return str(ip) if ip else None

    # -- polling -----------------------------------------------------------

    def wait_until_running(
        self,
        instance_id: str,
        *,
        timeout_s: float = DEFAULT_PROVISION_TIMEOUT_S,
        poll_interval_s: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> Dict[str, Any]:
        """Poll until the instance is ``running``, or fail closed.

        Terminal failure statuses raise immediately (never auto-retry —
        the operator decides), and so does the deadline: a stuck
        provisioning must become a failed runner, not an endless wait.
        """
        if poll_interval_s < MIN_POLL_INTERVAL_S:
            raise VerdaError(f'poll_interval_s={poll_interval_s} is below '
                             f'the floor of {MIN_POLL_INTERVAL_S}s')
        deadline = now() + timeout_s
        last_status: Optional[str] = None
        while True:
            instance = self.get_instance(instance_id)
            status = self.instance_status(instance)
            if status:
                last_status = status
            if status == STATUS_RUNNING:
                return instance
            if status in _FAILURE_STATUSES:
                raise VerdaError(
                    f'instance {instance_id} entered {status} while '
                    'waiting for running; not auto-retrying (report and '
                    'decide)')
            if now() >= deadline:
                raise VerdaError(f'instance {instance_id} still '
                                 f'{last_status or "provisioning"} after '
                                 f'{timeout_s:.0f}s; refusing to wait longer')
            sleep(poll_interval_s)
