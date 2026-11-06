"""io.net VMaaS REST client.

io.net (https://ai.io.net) rents on-demand NVIDIA GPU VMs from a prepaid
USDC balance. The automation surface is a plain-JSON REST API under
``https://api.io.solutions/enterprise/v1/io-cloud/vmaas`` (docs:
https://docs.io.net; every endpoint below was verified live 2026-11-03):

* **x-api-key auth on every endpoint.** ``x-api-key: <IO_NET_API_KEY>``;
  the key is provisioned in the console and there is no dedicated
  validation endpoint — a bad key surfaces as HTTP 401 on the first real
  call.
* **CRITICAL — custom User-Agent.** The provider fronts the API with
  Cloudflare, which BANS python-urllib's default UA with HTTP 403
  (Error 1010 "Access denied", measured live 2026-10-23). Every request
  MUST carry ``User-Agent: urun-skypilot-ionet/0.1 (+https://urun.sh)``
  (verified passing) — never urllib's default.
* **Hardware is the catalog.** ``GET /hardware`` returns the full
  config×location matrix (``id`` like ``8H100.80S.176V__FI``,
  ``deploy_id`` like ``8H100.80S.176V``, country-code ``location``).
  ``deploy_id`` is the stable identity the deploy API accepts back;
  it repeats across locations (one row per (config, location)).
* **Price endpoint doubles as the stock oracle.** ``GET /price`` with
  ``replica_count=1&duration_hours=1`` answers either the in-stock
  quote (``{"data": {..., "total_cost_usdc": ...}}``) or — HTTP 200,
  NOT an HTTP error — ``{"status": "failed", "error": "No available
  instances found for deployment"}``. The failed envelope means SOLD
  OUT and is surfaced as ``None`` from ``price_check``.
  ``available_replica_count`` in the success body is CAPPED AT 1: it is
  a ≥1 availability flag, never a unit count (never present it as one).
* **price is per-VM $/hr.** The hardware row's ``price`` covers the
  whole box for ``num_cards`` GPUs; ``total_cost_usdc`` covers the
  whole deployment (replicas × hours).
* **Deployments carry names.** The deploy body takes
  ``resource_private_name`` and ``GET /deployments`` lists every deployment
  with its ``name``, so the cluster identity is server-side — no local
  identity store.
* **SSH arrives as a command string.** ``GET /deployment/{id}/vms``
  workers carry ``ssh_access`` — the full ``ssh -p <port> <user>@<host>``
  command the provider's own UI shows — plus ``network_services`` with
  ``public_ip``/``public_port`` as the fallback address source.
* **Destroy is the only teardown.** ``DELETE /deployment/{id}``
  destroys the VM and its data (first hour non-refundable, then
  pro-rated); there is no stop-and-keep, so a silent
  destroy-as-stop would destroy while the caller believed it
  preserved (declared unsupported at the cloud layer instead).

Wire protocol notes (each verified live 2026-11-03):

* **Plain JSON** under a ``data`` envelope (``hardware``, ``workers``,
  ``deployments``); deploy/extend answer the created/updated object.
* **No documented rate budget.** 429s are retried defensively (bounded
  exponential, three strikes) but no budget constants are published.
* **API-key env spelling**: ``IO_NET_API_KEY`` is the only spelling this
  stack reads (the controller's task secret channel).
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

API_BASE = 'https://api.io.solutions/enterprise/v1/io-cloud/vmaas'

# CRITICAL: the provider's Cloudflare bans python-urllib's default UA
# (HTTP 403, Error 1010 "Access denied", measured live 2026-10-23).
# Every request — including the reads inside poll loops — must carry
# this identification string (verified passing).
USER_AGENT = 'urun-skypilot-ionet/0.1 (+https://urun.sh)'

# One timeout for every request, including the reads inside poll loops —
# the point is the bound, not the value.
DEFAULT_TIMEOUT_S = 60

# No rate budget is published for this API; 429 handling is defensive only.
MAX_RATE_LIMIT_RETRIES = 3
MAX_RETRY_AFTER_S = 60.0

# The VM image every lane deployment boots: 'general' (Ubuntu 24.04 +
# CUDA + NVIDIA drivers, verified in the deploy body's vm_image_type
# enumeration live 2026-11-03); 'datascience' is the alternative.
DEFAULT_IMAGE_TYPE = 'general'

# list_deployments walks this many pages maximum (default page_size: 20 x
# 10 = 200 deployments) before refusing — the bound exists so a
# pathological or hostile account cannot loop the identity walk forever;
# the account would have to be decades past exhausted for the cap to lie.
MAX_DEPLOYMENT_PAGES = 20

# Observed deployment status values (docs + live list 2026-11-03; the
# list endpoint was renamed GET /deploy -> GET /deployments and became
# 1-indexed (page >= 1 enforced, 422 below) by 2026-11-05 — live-probed):
# 'running' while the boxes are up and serving; the steady-state signal
# this client trusts for readiness is status == 'running' plus SSH-bearing
# VMs, not compute_minutes_remaining (the set is not documented as
# closed).
STATUS_RUNNING = 'running'

# The provider's documented poll cadence is minutes, not seconds (a GPU VM
# boots an OS + CUDA stack); polling faster is budget waste.
MIN_POLL_INTERVAL_S = 10

# ~20-minute worst-case boot (OS + CUDA + drivers) plus margin.
DEFAULT_PROVISION_TIMEOUT_S = 20 * 60

# Environment variable the callers read the key from (the task secret
# channel's spelling; the vendor publishes no alias convention).
API_KEY_ENV = 'IO_NET_API_KEY'
API_KEY_FILE = '~/.ionet/api_key'

# The documented SSH user for the 'general' image (Ubuntu 24.04 + CUDA):
# the fallback ONLY when ssh_access carries no user (it normally does).
# The Latitude lesson: docs can say root while the box answers ubuntu —
# the live ssh_access user is the source of truth, this is the bootstrap.
SSH_USER_FALLBACK = 'root'

class IonetError(RuntimeError):
    """Any io.net API failure. Never raised with the API key in it.

    Carries the provider's own error code when the response had one, so
    callers can branch on it without re-parsing.
    """

    def __init__(self, message: str, code: str = ''):
        super().__init__(message)
        self.code = code


class IonetAuthError(IonetError):
    """401 — key missing, invalid, or revoked. Check/rotate it."""


class IonetRateLimited(IonetError):
    """429 that exhausted its retries. No documented body hint; bounded
    exponential backoff already ran."""


class IonetNotFoundError(IonetError):
    """404 — the resource is gone. Destroy paths treat this as success."""


def _error_envelope(raw: bytes) -> Dict[str, Any]:
    """The provider's error object from a body, or {} if unusable.

    Errors answer ``{"status": "failed", "error": ...}`` or a gateway's
    non-JSON page; a body that is not a JSON object must not crash the
    error path itself.
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class IonetClient:
    """Thin, explicit client. One method per endpoint we actually use.

    ``transport`` is injected in tests. Signature:
    ``(method, url, headers, payload) -> (status, bytes)``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_BASE,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        transport: Optional[Callable[
            [str, str, Dict[str, str], Optional[bytes]], Any]] = None,
    ):
        if not api_key or not api_key.strip():
            raise IonetAuthError(
                'io.net API key is empty; refusing to make '
                'unauthenticated calls that Cloudflare would answer '
                'with 403/401 and the lane would read as "no capacity"')
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
        # No query in errors: no key leakage (the query carries the
        # oracle's parameters, the headers carry the key).
        path = url.split('?')[0]
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                # resp.read() runs OUTSIDE urlopen's URLError handling: a
                # stalled body raises TimeoutError/OSError directly and would
                # escape every caller that only catches IonetError (the
                # trap CodeRabbit flagged on the Spheron client).
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            except (TimeoutError, OSError) as read_exc:
                raise IonetError(f'{method} {path} failed reading the error '
                                 f'body: {read_exc}') from read_exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise IonetError(f'{method} {path} failed: {exc.reason}') from exc
        except (TimeoutError, OSError) as exc:
            raise IonetError(f'{method} {path} failed: {exc}') from exc

    def _request_once(self, method: str, url: str, headers: Dict[str, str],
                      payload: Optional[bytes]):
        if self._transport is not None:
            status, raw = self._transport(method, url, headers, payload)
        else:
            status, raw = self._http(method, url, headers, payload)
        return status, raw

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """One authenticated JSON call. Every endpoint takes the x-api-key
        header (there are no public endpoints on this API)."""
        url = f'{self._base_url}{path}'
        if query:
            # location_ids is a URL-encoded JSON array ("US" not US);
            # quote (not quote_plus) keeps the %22/%5B shape the provider
            # documented.
            url += '?' + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
            'x-api-key': self._api_key,
        }
        payload = None
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            status, raw = self._request_once(method, url, headers, payload)
            if status == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                # No documented retry hint on 429 bodies: bounded
                # exponential, the same defense every sibling client runs.
                time.sleep(min(2.0 * (attempt + 1), MAX_RETRY_AFTER_S))
                continue
            return self._interpret(status, raw, method, path)

        # Unreachable: the loop returns or raises on the final attempt.
        raise AssertionError('rate-limit retry loop exhausted')

    def _interpret(self, status: int, raw: bytes, method: str, path: str):
        # Never include the URL query or headers in an error: no key
        # leakage.
        where = f'{method} {path}'
        if status < 400:
            # The empty-body success path lives UNDER the status check: a
            # 4xx/5xx with an empty body is a FAILURE (raise below), never
            # a silent None — the bug class where a provider's empty-body
            # error would read as "done" while the box keeps billing.
            if not raw:
                return None
            try:
                return json.loads(raw)
            # ValueError covers JSONDecodeError AND UnicodeDecodeError (a
            # non-UTF-8 gateway page is a ValueError, not a
            # JSONDecodeError), so no raw exception escapes an
            # IonetError-only caller.
            except ValueError as exc:
                raise IonetError(
                    f'{where}: response was not JSON ({exc})') from exc

        err = _error_envelope(raw)
        code = str(err.get('code') or '')
        message = str(err.get('message') or err.get('error') or '')[:300]
        if not code and not message:
            # An error with no usable envelope still must not collapse to
            # a bare status line: show the raw prefix instead.
            message = raw[:200].decode('utf-8', 'replace')

        if status == 401:
            raise IonetAuthError(
                f'{where}: HTTP 401. Check {API_KEY_ENV} (or rotate the key '
                'in the io.net console under API keys).')
        if status == 404:
            raise IonetNotFoundError(f'{where}: not found ({message})')
        if status == 429:
            default = code or 'rate_limited'
            raise IonetRateLimited(
                f'{where}: rate limited ({default}); detail: {message}')
        default = code or 'error'
        raise IonetError(
            f'{where}: HTTP {status}: {default}: {message}',
            code=code,
        )

    # -- catalog ----------------------------------------------------------

    def list_hardware(self) -> List[Dict[str, Any]]:
        """GET /hardware — the config×location matrix (``data.hardware``).

        One row per (hardware config, location); ``deploy_id`` repeats
        across locations and is the STABLE identity the deploy API
        accepts back. ``price`` is per-VM $/hr (the whole box for
        ``num_cards`` GPUs). This is the fetcher's entire input.
        """
        payload = self._call('GET', '/hardware') or {}
        if not isinstance(payload, dict):
            raise IonetError('GET /hardware: expected an object, got '
                             f'{type(payload).__name__}')
        data = payload.get('data')
        if not isinstance(data, dict):
            raise IonetError('GET /hardware: expected an object at .data, '
                             f'got {type(data).__name__}')
        rows = data.get('hardware')
        if not isinstance(rows, list):
            raise IonetError(
                'GET /hardware: expected a list at .data.hardware, got '
                f'{type(rows).__name__}')
        return [row for row in rows if isinstance(row, dict)]

    def price_check(
        self,
        hardware_id: str,
        gpus_per_vm: int,
        location: str,
        *,
        replica_count: int = 1,
        duration_hours: int = 1,
        currency: str = 'usdc',
    ) -> Optional[Dict[str, Any]]:
        """GET /price — the stock oracle AND the quote endpoint.

        Returns the inner ``data`` dict on success
        (``replica_count``/``gpus_per_vm``/``available_replica_count``/
        ``total_cost_usdc``) and ``None`` when the provider answers the
        no-instances envelope — which is HTTP 200 with
        ``{"status": "failed", "error": "No available instances found
        for deployment"}``, NOT an HTTP error (measured live 2026-11-03:
        the sold-out RTX6000Ada 1x FI answers exactly this). Every other
        failure raises normally.

        ``available_replica_count`` is CAPPED AT 1 by the provider — it
        is a ≥1 availability flag, never a unit count; callers must not
        present it as units (the stock-honesty rule).
        """
        if not str(hardware_id or '').strip():
            raise IonetError('refusing a price probe without a hardware_id '
                             '(the deploy_id, e.g. 8H100.80S.176V)')
        # location_ids is a JSON array of one country code, URL-encoded:
        # %5B%22US%22%5D — the exact shape the provider documented.
        query: Dict[str, Any] = {
            'hardware_id': str(hardware_id).strip(),
            'gpus_per_vm': int(gpus_per_vm),
            'replica_count': int(replica_count),
            'duration_hours': int(duration_hours),
            'currency': str(currency or 'usdc'),
            'location_ids': json.dumps([str(location)]),
        }
        payload = self._call('GET', '/price', query=query)
        if not isinstance(payload, dict):
            raise IonetError('GET /price: expected an object, got '
                             f'{type(payload).__name__}')
        if str(payload.get('status') or '') == 'failed':
            # The provider's sold-out answer: HTTP 200, no instances.
            # This is "not offered right now", never a parse failure.
            return None
        data = payload.get('data')
        if not isinstance(data, dict):
            raise IonetError(
                'GET /price: expected an object at .data, got '
                f'{type(data).__name__}')
        return data

    # -- deployments --------------------------------------------------------

    def deploy(
        self,
        *,
        resource_private_name: str,
        duration_hours: int,
        hardware_id: str,
        gpus_per_vm: int,
        location_ids: List[str],
        ssh_keys: Dict[str, str],
        vm_image_type: str = DEFAULT_IMAGE_TYPE,
        network_services: Optional[Dict[str, Any]] = None,
    ) -> str:
        """POST /deploy — rent the VMs; returns the deployment id.

        The body is the provider's documented shape (snake_case keys):
        ``resource_private_name`` (the server-side cluster identity),
        ``duration_hours`` (1..8760), ``gpus_per_vm`` (1..8),
        ``hardware_id`` (the stable ``deploy_id``), ``location_ids``
        (country codes), ``ssh_keys`` (name → public key MATERIAL,
        uploaded inline — unlike MassedCompute there is no key registry),
        ``vm_image_type`` and the optional ``network_services`` port map.
        ``vms_qty`` is deprecated — never sent.
        """
        name = str(resource_private_name or '').strip()
        if not name:
            raise IonetError(
                'refusing to deploy without a resource_private_name (the '
                'server-side cluster identity)')
        if not str(hardware_id or '').strip():
            raise IonetError(
                'refusing to deploy without a hardware_id (the stable '
                'catalog deploy_id)')
        if not ssh_keys or not any(str(v or '').strip()
                                   for v in ssh_keys.values()):
            raise IonetError(
                'refusing to deploy with no ssh key MATERIAL (the keys are '
                'uploaded inline with the deploy; a box without one is '
                'unreachable and still bills)')
        body: Dict[str, Any] = {
            'resource_private_name': name,
            'duration_hours': int(duration_hours),
            'hardware_id': str(hardware_id).strip(),
            'gpus_per_vm': int(gpus_per_vm),
            'location_ids': [str(loc) for loc in (location_ids or [])],
            'ssh_keys': {
                str(k): str(v) for k, v in ssh_keys.items()
                if str(v or '').strip()
            },
            'vm_image_type': str(vm_image_type or DEFAULT_IMAGE_TYPE),
        }
        if network_services:
            body['network_services'] = network_services
        payload = self._call('POST', '/deploy', body=body) or {}
        if not isinstance(payload, dict):
            raise IonetError('POST /deploy: expected an object, got '
                             f'{type(payload).__name__}')
        # The answer carries the created deployment under .data (with
        # .id) — tolerate a top-level id too, never a guess past both.
        source = payload.get('data')
        if not isinstance(source, dict):
            source = payload
        deployment_id = str(source.get('id') or '').strip()
        if not deployment_id:
            raise IonetError(
                f'POST /deploy returned no deployment id: '
                f'{sorted(payload)[:5]}')
        return deployment_id

    def list_deployments(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """GET /deployments — every deployment on the account
        (``data.deployments``), PAGINATED.

        This is the server-side cluster identity source: deployments
        carry their ``name`` (the deploy body's ``resource_private_name``),
        so the provisioner finds the cluster's deployment by name —
        which is why a single page cannot be the answer (CodeRabbit round
        1): a deployment past page 1 would read as ABSENT, so teardown
        would re-deploy a live cluster and a stale name lookup would
        duplicate it. Pages are followed until one returns fewer than
        ``page_size`` rows, deduplicated by ``id`` (the API documents no
        cross-page ordering guarantee), capped at MAX_DEPLOYMENT_PAGES
        so a pathological account cannot loop the caller forever.
        """
        collected: Dict[str, Dict[str, Any]] = {}
        current = int(page)
        while True:
            query: Dict[str, Any] = {
                'page': current,
                'page_size': int(page_size),
            }
            if status:
                query['status'] = str(status)
            payload = self._call('GET', '/deployments', query=query) or {}
            if not isinstance(payload, dict):
                raise IonetError(
                    'GET /deployments: expected an object, got '
                    f'{type(payload).__name__}')
            data = payload.get('data')
            if not isinstance(data, dict):
                raise IonetError(
                    'GET /deployments: expected an object at .data, got '
                    f'{type(data).__name__}')
            rows = data.get('deployments')
            if not isinstance(rows, list):
                raise IonetError(
                    'GET /deployments: expected a list at .data.deployments, '
                    f'got {type(rows).__name__}')
            valid = [row for row in rows if isinstance(row, dict)]
            for row in valid:
                row_id = str(row.get('id') or '')
                if row_id and row_id not in collected:
                    collected[row_id] = row
            if len(valid) < int(page_size):
                break  # short page: the account is exhausted
            current += 1
            if current - int(page) >= MAX_DEPLOYMENT_PAGES:
                raise IonetError(
                    f'GET /deployments: exceeded {MAX_DEPLOYMENT_PAGES} '
                    f'pages '
                    f'({int(page_size)} rows each) — refusing an unbounded '
                    'walk; raise the cap if the account is really this big')
        return list(collected.values())

    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """GET /deployment/{id} — the deployment detail object."""
        if not str(deployment_id or '').strip():
            raise IonetError('refusing a deployment read without an id')
        payload = self._call('GET',
                             f'/deployment/{deployment_id}') or {}
        if not isinstance(payload, dict):
            raise IonetError(
                f'GET /deployment/{deployment_id}: expected an object, got '
                f'{type(payload).__name__}')
        # The detail endpoint wraps in .data; tolerate the bare object —
        # never a guess at a deeper key.
        data = payload.get('data')
        if isinstance(data, dict):
            return data
        return payload

    def get_deployment_vms(
        self,
        deployment_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """GET /deployment/{id}/vms — the workers (``data.workers``).

        Each worker carries ``ssh_access`` (the full SSH command string
        the provider's UI shows) and ``network_services`` with
        ``public_ip``/``public_port`` as the fallback address source.
        """
        if not str(deployment_id or '').strip():
            raise IonetError('refusing a VM read without a deployment id')
        payload = self._call(
            'GET',
            f'/deployment/{deployment_id}/vms',
            query={'page': int(page), 'page_size': int(page_size)},
        ) or {}
        if not isinstance(payload, dict):
            raise IonetError(
                f'GET /deployment/{deployment_id}/vms: expected an object, '
                f'got {type(payload).__name__}')
        data = payload.get('data')
        if not isinstance(data, dict):
            raise IonetError(
                f'GET /deployment/{deployment_id}/vms: expected an object '
                f'at .data, got {type(data).__name__}')
        rows = data.get('workers')
        if not isinstance(rows, list):
            raise IonetError(
                f'GET /deployment/{deployment_id}/vms: expected a list at '
                f'.data.workers, got {type(rows).__name__}')
        return [row for row in rows if isinstance(row, dict)]

    def destroy_deployment(self, deployment_id: str) -> None:
        """DELETE /deployment/{id} — destroy the VMs and their data.

        Idempotent teardown: a 404 for an already-gone deployment is
        success (the box is already destroyed), which is what makes
        reconcile safe to retry. Billing: the first hour is
        non-refundable, then pro-rated.
        """
        if not str(deployment_id or '').strip():
            raise IonetError(
                'refusing to destroy with an empty deployment id')
        try:
            self._call('DELETE', f'/deployment/{deployment_id}')
        except IonetNotFoundError:
            pass

    def extend_deployment(self, deployment_id: str,
                          duration_hours: int) -> None:
        """POST /deployment/{id}/extend — add time to a live deployment."""
        if not str(deployment_id or '').strip():
            raise IonetError('refusing to extend with an empty deployment id')
        self._call(
            'POST',
            f'/deployment/{deployment_id}/extend',
            body={'duration_hours': int(duration_hours)},
        )

    # -- accessors ---------------------------------------------------------

    @staticmethod
    def deployment_id(deployment: Dict[str, Any]) -> str:
        return str(deployment.get('id') or '').strip()

    @staticmethod
    def deployment_name(deployment: Dict[str, Any]) -> str:
        return str(deployment.get('name') or '').strip()

    @staticmethod
    def deployment_status(deployment: Dict[str, Any]) -> str:
        return str(deployment.get('status') or '').strip().lower()

    @staticmethod
    def parse_ssh_access(worker: Dict[str, Any]) -> Optional[Tuple[
            str, str, int]]:
        """The worker's (user, host, port) parsed from ``ssh_access``.

        ``ssh_access`` is the full command string the provider's own UI
        shows — ``ssh -p <port> <user>@<host>`` (with stray ``-o``
        options in some answers). The port and user are optional in the
        string; the fallbacks are the SSH default port 22 and the image's
        documented user. A worker with NO address at all returns None —
        never a guessed host that would SSH somewhere foreign.
        """
        command = str(worker.get('ssh_access') or '').strip()
        if not command:
            return None
        tokens = command.split()
        if not tokens or tokens[0] != 'ssh':
            # Not the documented shape: refuse rather than guess at a
            # foreign syntax (never ssh into a parsed-wrong host).
            return None
        port, user, host = 22, '', ''
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token == '-p' and i + 1 < len(tokens):
                try:
                    port = int(tokens[i + 1])
                except ValueError:
                    return None
                i += 2
                continue
            if token.startswith('-'):
                # An option (-o, -i, ...): skip its argument too when
                # one follows (``-o StrictHostKeyChecking=no``).
                i += 2 if token in ('-o', '-i', '-L', '-R') and \
                    i + 1 < len(tokens) else 1
                continue
            target = token
            if '@' in target:
                user, _, host = target.rpartition('@')
            else:
                host = target
            i += 1
        if not host:
            return None
        return (user or SSH_USER_FALLBACK, host, port)

    @staticmethod
    def worker_address(worker: Dict[str, Any]) -> Optional[Tuple[
            str, str, int]]:
        """The worker's (user, host, port): ``ssh_access`` first, then the
        ``network_services`` ``public_ip``/``public_port`` fallback.

        ``network_services`` is a name-keyed map of the ports opened at
        deploy; any entry carrying a public_ip answers as the address
        (with its public_port). Neither source yielding a host returns
        None — the box is unreachable and that must stay loud.
        """
        parsed = IonetClient.parse_ssh_access(worker)
        if parsed is not None:
            user, host, port = parsed
            if host:
                return (user, host, port)
        services = worker.get('network_services')
        if isinstance(services, dict):
            for service in services.values():
                if not isinstance(service, dict):
                    continue
                host = str(service.get('public_ip') or '').strip()
                if not host:
                    continue
                port_raw = service.get('public_port')
                try:
                    port = int(port_raw) if port_raw is not None else 22
                except (TypeError, ValueError):
                    port = 22
                # The service map carries no user; the general image's
                # documented user is the fallback.
                return (SSH_USER_FALLBACK, host, port)
        return None

    # -- polling -----------------------------------------------------------

    def wait_until_running(
        self,
        deployment_id: str,
        *,
        timeout_s: float = DEFAULT_PROVISION_TIMEOUT_S,
        poll_interval_s: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> Dict[str, Any]:
        """Poll the deployment until status 'running', or fail closed.

        A 404 inside the poll loop is tolerated until the deadline: a
        JUST-CREATED deployment can 404 on its first detail read (the
        read-after-write window server-side — the QuantaCloud ENG-575
        lesson), so the first NotFound must not read as "deleted". The
        deadline still fails closed: a stuck deploy must become a failed
        runner, not an endless wait — the box bills from the first hour
        either way.
        """
        if poll_interval_s < MIN_POLL_INTERVAL_S:
            raise IonetError(
                f'poll_interval_s={poll_interval_s} is below the documented '
                f'floor of {MIN_POLL_INTERVAL_S}s')
        deadline = now() + timeout_s
        while True:
            try:
                deployment = self.get_deployment(deployment_id)
            except IonetNotFoundError:
                deployment = {}
            if self.deployment_status(deployment) == STATUS_RUNNING:
                return deployment
            if now() >= deadline:
                status = repr(
                    self.deployment_status(deployment) or 'unknown')
                raise IonetError(
                    f'deployment {deployment_id} still {status} after '
                    f'{timeout_s:.0f}s; refusing to wait longer')
            sleep(poll_interval_s)