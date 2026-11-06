"""Ornn Compute MCP client.

Ornn (https://compute.ornn.com) rents hourly GPU VMs on a **spot market**:
per-GPU-type order books, 1-8 GPUs per bid on ONE node, prepaid per hour,
part-hour refunds to credits. A filled bid auto-starts a single isolated VM
from the org's first custom image (or Ornn's default) with **every active
account SSH key** attached — there is no per-instance key or image argument
on the bid.

There is NO REST API key. The automation surface is the hosted MCP server
(JSON-RPC 2.0 over streamable HTTP at https://mcp.ornn.com/mcp) driven by an
OAuth credential: ``client_id`` + ``client_secret`` + ``refresh_token``.
Wire receipts (2026-10-08, ENG-541):

* The RFC 7591 registration echoes ``token_endpoint_auth_method: none`` but
  the token endpoint REQUIRES the client secret (401 ``invalid_client`` on
  the public PKCE variant) — we always send ``client_secret_post``.
* **Both tokens rotate on every refresh.** The new pair is persisted
  (mode 600) before the next call; a spent refresh token is never reused.
  Refreshing with an already-spent token returns 200 (advisory rotation, not
  enforced), so an env-injected credential survives controller pod restarts;
  the client still rotates-and-persists by default.
* Money/mutation tools are confirm-gated: without ``confirm: true`` the call
  is FREE and returns a PREVIEW — the launch-refusal floor.
* Responses are JSON or SSE; a tools/call result's ``content[0].text`` is a
  JSON document on success (or a bare token such as ``not_found`` /
  ``image_required`` on refusal). Refusals carry their body verbatim into
  the raised error — a flattened refusal reads as "no capacity" and that is
  exactly the bug class that misprices a lane.
* A market bid that cannot fill is CANCELLED by the server — a capacity
  refusal to surface, never a condition to blind-retry.
* Manual ``ornn_access_activate`` on a spot reservation can refuse with
  ``image_required`` (neither bid nor activate exposes an image argument in
  the live tools/list schema — reproduced twice 2026-10-08). The lane's
  launch path is the bid's AUTO-LAUNCH ONLY; this refusal is carried as a
  typed, never-retried error.
* Tool arguments are camelCase (``gpuSlug``, ``reservationId``,
  ``sshKeyIds``); telemetry tools are the snake_case exception
  (``node_id``).

API v2 (2026-11-05/06): the tool surface was renamed mid-migration —
``ornn_spot_books`` → ``market_books``, ``ornn_spot_book`` → ``market_book``,
``ornn_spot_trades`` → ``market_trades``, ``ornn_schedules_list`` →
``listings_list``, ``ornn_reservations_*`` → ``reservations_*``,
``ornn_access_*`` → ``access_*``, ``ornn_ssh_keys_*`` → ``ssh_keys_*``.
The LAUNCH path is now MEASURED against live reservations (2026-11-06,
~$2.25 total spend, one on-demand rtx_pro_6000 fill exercised
bid→activate→teardown end to end):

* Bids buy NODES — whole machines — via ``market_bid`` (v1's 1-8 GPU
  slice is gone); ``market_quote`` is the free preview, and the fill ID
  IS the reservation ID. The v1 confirm-gate is gone server-side; this
  client rebuilds the launch-refusal floor on the quote (no
  ``confirm`` → no spend).
* A filled bid does NOT auto-launch: ``access_activate`` boots the
  machine and provisions the ``sshKeyIds`` passed THERE (later key adds
  do not reach a booted box).
* The SSH surface is ``storage_targets`` (``nodes[].ip`` +
  ``nodeUsername``) — ``access_show`` STAYS EMPTY for VM-mode
  reservations even with the box up (measured on two live fills).
* Teardown is ``market_end {fillId}`` — async (the node vanishes from
  ``storage_targets`` in ~30s) and NOT 404-idempotent: repeats while
  processing refuse with "Wait for conflicting operation", treated as
  success-in-flight by this client.

Every call carries an explicit timeout (a single hung HTTP call once wedged
a runner 30+ min); token values never appear in error strings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

# The MCP server (JSON-RPC 2.0 over streamable HTTP).
MCP_BASE = "https://mcp.ornn.com/mcp"

# The OAuth authorization server's token endpoint.
TOKEN_URL = "https://compute.ornn.com/v1/auth/mcp/token"

# Identifies the integration to the provider's logs.
USER_AGENT = "urun-skypilot-ornn/0.1 (+https://urun.sh)"

# One timeout for EVERY request, including the reads inside poll loops —
# the point is the bound, not the value.
DEFAULT_TIMEOUT_S = 60

# 429 handling: no documented rate limit on the MCP surface, so back off
# bounded-exponentially; three strikes then raise.
MAX_RATE_LIMIT_RETRIES = 3

# listings_list pages drained per call (100 rows each = 2,000 listings;
# the marketplace had ~23 available listings at v2 launch — the bound
# exists so a server cursor loop can never pin the runner).
_LISTINGS_MAX_PAGES = 20

MAX_RETRY_AFTER_S = 60.0

# The provider applies Ubuntu security updates (with reboots) BEFORE the SSH
# port opens; first boot takes minutes longer. Poll slower than the updates.
MIN_POLL_INTERVAL_S = 15

# ~4-minute deploy record plus the minutes-long security-update first boot,
# with margin — fail CLOSED on the deadline (a stuck reservation must become
# a failed runner, not an endless wait).
DEFAULT_PROVISION_TIMEOUT_S = 25 * 60

# Access tokens are issued with expires_in=3600; refresh this early.
_ACCESS_TOKEN_MARGIN_S = 60

# The credential: the OAuth client JSON blob
# {"client_id": ..., "client_secret": ..., "refresh_token": ...}.
# (A refresh-token-only env CANNOT work: the exchange requires
# client_id + client_secret — the 401 receipt.)
CREDENTIALS_ENV = "ORNN_MCP_CREDENTIALS"
CREDENTIALS_FILE = "~/.ornn/mcp-credentials"
CLIENT_FILE = "~/.ornn/mcp-client.json"

# Reservation / order statuses (receipts 2026-10-08; open sets — anything
# unmapped is transitional and keeps waiting).
STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
ORDER_RUNNING = "running"
ORDER_ENDED = "ended"

# Manual activation refusal bodies.
REFUSAL_IMAGE_REQUIRED = "image_required"
REFUSAL_NOT_FOUND = "not_found"
# ornn_access_show refuses with this body while a fresh fill is still
# pendingOperation=activate — the fill→deploy window (measured ~4 min,
# 2026-10-08 receipts). A live fill died +16s on this refusal (ENG-580,
# 2026-10-20: order 57829c4e filled at $1.80, wait_until_ready raised
# before the VM could deploy). In wait_until_ready it is a POLL signal,
# bounded by the deadline like any other not-yet-ready state.
ACCESS_PENDING_REFUSAL = "cannot launch in its current state"


class OrnnError(RuntimeError):
    """Any Ornn API failure. Never raised with a token value in it."""


class OrnnAuthError(OrnnError):
    """401 / token-exchange failure — credential missing, spent, or revoked."""


class OrnnRateLimitedError(OrnnError):
    """429 that exhausted its retries. Carries the last retry hint."""

    def __init__(self, message: str, retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class OrnnNotFoundError(OrnnError):
    """The resource is gone. Terminate paths treat this as success."""


class OrnnCapacityRefusedError(OrnnError):
    """A market bid was cancelled — capacity refusal, never blind-retried."""


class OrnnImageBlockedError(OrnnError):
    """``ornn_access_activate`` refused with ``image_required``.

    Neither the bid nor the activate call exposes an image argument in the
    live tools/list schema (receipt 2026-10-08, reproduced twice), so this
    refusal is NOT retryable by this integration: the launch path is the
    bid's AUTO-LAUNCH, and manual activation is carried as this typed error
    so an operator sees the provider's own refusal instead of a wedge.
    """


class OrnnMcpProtocolError(OrnnError):
    """JSON-RPC layer failure (malformed envelope, missing content, ...)."""


class OrnnInputError(OrnnError):
    """The server rejected our arguments (``Input validation error``).

    A programming bug on our side: loud, never retried, carries the
    server's message so the arg name is visible without a debugger.
    """


class OrnnCredentials:
    """The OAuth credential blob + the rotating token pair.

    ``client_id``/``client_secret`` come from the RFC 7591 registration;
    ``refresh_token`` + ``access_token`` rotate on every refresh (persisted
    immediately — the old refresh token is single-use by contract even
    though the server currently accepts spent tokens advisorially).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        persist_path: Optional[str] = None,
    ):
        if not client_id or not client_secret or not refresh_token:
            raise OrnnAuthError(
                "Ornn credential is incomplete: expects the JSON blob "
                f"{CREDENTIALS_ENV} with client_id, client_secret and "
                "refresh_token (a refresh-token-only credential cannot "
                "refresh: the exchange requires client_id + client_secret)")
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._access_token: Optional[str] = None
        self._access_expires_at: float = 0.0
        # Set when the credential was loaded from a file (rotation is
        # persisted back); None when it came from the env (kept in memory).
        self._persist_path = persist_path

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        credentials_env: str = CREDENTIALS_ENV,
        credentials_file: str = CREDENTIALS_FILE,
        client_file: str = CLIENT_FILE,
    ) -> "OrnnCredentials":
        """The credential from the env JSON blob, else the local files."""
        raw = os.environ.get(credentials_env, "").strip()
        if raw:
            try:
                blob = json.loads(raw)
            except ValueError as exc:
                raise OrnnAuthError(
                    f"{credentials_env} is not valid JSON ({exc})") from exc
            if not isinstance(blob, dict):
                raise OrnnAuthError(
                    f"{credentials_env} must be a JSON object with "
                    "client_id, client_secret and refresh_token")
            return cls(
                str(blob.get("client_id") or ""),
                str(blob.get("client_secret") or ""),
                str(blob.get("refresh_token") or ""),
            )
        creds_path = Path(credentials_file).expanduser()
        if not creds_path.is_file():
            raise OrnnAuthError(
                f"no Ornn credential: {credentials_env} is unset and "
                f"{credentials_file} does not exist")
        try:
            blob = json.loads(creds_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OrnnAuthError(
                f"cannot read Ornn credential file {credentials_file} ({exc})"
            ) from exc
        # The file store keeps the rotated pair; the client registration
        # (client_id/client_secret) lives beside it.
        client_path = Path(client_file).expanduser()
        client_id = str(blob.get("client_id") or "")
        client_secret = str(blob.get("client_secret") or "")
        # The token store may carry only the rotating pair; the RFC 7591
        # client registration (client_id + client_secret) lives beside it.
        if client_path.is_file():
            try:
                client_blob = json.loads(
                    client_path.read_text(encoding="utf-8"))
                if isinstance(client_blob, dict):
                    if not client_id:
                        client_id = str(client_blob.get("client_id") or "")
                    if not client_secret:
                        client_secret = str(
                            client_blob.get("client_secret") or "")
            except (OSError, ValueError):
                pass
        return cls(
            client_id,
            client_secret,
            str(blob.get("refresh_token") or ""),
            persist_path=str(creds_path),
        )

    # -- refresh -----------------------------------------------------------

    def access_token(
        self,
        *,
        transport: Optional[Callable[
            [str, str, Dict[str, str], Optional[bytes]], Any]] = None,
    ) -> str:
        """A valid access token, refreshing first when expired/missing.

        ``transport`` threads the client's injected transport into the
        refresh so tests never hit the network.
        """
        if (self._access_token is None or
                time.time() >= self._access_expires_at):
            self.refresh(transport=transport)
        return str(self._access_token)

    def refresh(
        self,
        *,
        transport: Optional[Callable[
            [str, str, Dict[str, str], Optional[bytes]], Any]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> str:
        """Exchange the refresh token; rotate BOTH tokens; persist first.

        ``grant_type=refresh_token`` with ``client_secret_post`` (the
        registration echoes ``token_endpoint_auth_method: none`` but the
        endpoint requires the secret — 401 ``invalid_client`` otherwise).
        """
        body = urllib.parse.urlencode({
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        if transport is not None:
            status, raw = transport("POST", TOKEN_URL, headers, body)
        else:
            req = urllib.request.Request(TOKEN_URL,
                                         data=body,
                                         headers=headers,
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    status, raw = resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                try:
                    status, raw = exc.code, exc.read()
                except (TimeoutError, OSError) as read_exc:
                    raise OrnnError(
                        "POST token endpoint failed reading the error "
                        f"body: {read_exc}") from read_exc
            except urllib.error.URLError as exc:  # pragma: no cover
                raise OrnnError(
                    f"POST token endpoint failed: {exc.reason}") from exc
            except (TimeoutError, OSError) as exc:
                raise OrnnError(f"POST token endpoint failed: {exc}") from exc
        if status in (400, 401):
            raise OrnnAuthError(
                "Ornn token refresh was refused (HTTP "
                f"{status}); the credential's refresh token chain is broken "
                "— re-authorize the MCP OAuth client")
        if status != 200 or not raw:
            raise OrnnError(f"Ornn token refresh returned HTTP {status} "
                            f"({raw[:120].decode('utf-8', 'replace')!r})")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise OrnnError(
                f"Ornn token refresh returned non-JSON ({exc})") from exc
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise OrnnError("Ornn token refresh returned no access_token: "
                            f"{sorted(payload)}")
        # BOTH tokens rotate; persist before the next call (the old refresh
        # token is single-use by contract).
        self._access_token = str(payload["access_token"])
        self._access_expires_at = time.time() + float(
            payload.get("expires_in") or 3600) - _ACCESS_TOKEN_MARGIN_S
        if payload.get("refresh_token"):
            self._refresh_token = str(payload["refresh_token"])
        self.persist()
        return self._access_token

    def persist(self) -> None:
        """Persist the rotated pair to the file store (never the env path)."""
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        blob = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": self._refresh_token,
            "access_token": self._access_token,
            "expires_in": int(max(0.0, self._access_expires_at - time.time())),
        }
        existing = {}
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, ValueError):
            existing = {}
        existing.update(blob)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        os.chmod(path, 0o600)


class OrnnClient:
    """Thin MCP client. One method per tool we actually use.

    The JSON-RPC handshake (initialize + notifications/initialized) runs
    lazily per transport; the server keeps no session id (probed), so plain
    request/response works. ``transport`` is injected in tests:
    ``(method, url, headers, payload) -> (status, bytes)``.
    """

    _JSONRPC_PROTOCOL = "2025-03-26"

    def __init__(
        self,
        credentials: OrnnCredentials,
        *,
        base_url: str = MCP_BASE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: Optional[Callable[
            [str, str, Dict[str, str], Optional[bytes]], Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport
        self._sleep = sleep
        self._rpc_id = 0
        self._initialized = False

    # -- plumbing ----------------------------------------------------------

    def _http(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload: Optional[bytes],
    ):
        req = urllib.request.Request(url,
                                     data=payload,
                                     headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                # resp.read() runs OUTSIDE urlopen's URLError handling: a
                # stalled body raises TimeoutError/OSError directly and
                # would escape every caller that only catches OrnnError.
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            except (TimeoutError, OSError) as read_exc:
                raise OrnnError(
                    f"{method} {url.split('?')[0]} failed reading the error "
                    f"body: {read_exc}") from read_exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise OrnnError(
                f"{method} {url.split('?')[0]} failed: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise OrnnError(
                f"{method} {url.split('?')[0]} failed: {exc}") from exc

    def _request_once(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload: Optional[bytes],
    ):
        if self._transport is not None:
            return self._transport(method, url, headers, payload)
        return self._http(method, url, headers, payload)

    def _parse_body(self, raw: bytes) -> Any:
        """JSON or SSE (``data:`` lines); a loud protocol error otherwise."""
        text = raw.decode("utf-8", "replace")
        stripped = text.lstrip()
        if stripped.startswith("data:") or "\ndata:" in text:
            for line in text.splitlines():
                if line.startswith("data:"):
                    body = line[5:].strip()
                    if body:
                        return json.loads(body)
            raise OrnnMcpProtocolError("SSE response with no data lines: "
                                       f"{text[:200]!r}")
        return json.loads(text)

    def _rpc(self, call: str, params: Optional[Dict[str, Any]]) -> Any:
        """One JSON-RPC request -> the ``result`` (or a typed failure)."""
        self._rpc_id += 1
        request: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": call,
        }
        if params is not None:
            request["params"] = params
        body = json.dumps(request).encode("utf-8")
        where = f"JSON-RPC {call}"
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            headers = {
                "Authorization": f"Bearer {self._credentials.access_token(transport=self._transport)}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "User-Agent": USER_AGENT,
            }
            status, raw = self._request_once("POST", self._base_url, headers,
                                             body)
            if status == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                # Bounded exponential backoff; no documented limit.
                self._sleep(min(2.0 * (attempt + 1), MAX_RETRY_AFTER_S))
                continue
            break
        if status == 429:
            raise OrnnRateLimitedError(
                f"{where}: rate limited after {MAX_RATE_LIMIT_RETRIES} "
                "backoffs")
        if status in (401, 403):
            raise OrnnAuthError(
                f"{where}: HTTP {status} — the MCP access token was refused; "
                "refresh/re-authorize the Ornn OAuth credential")
        if status != 200:
            raise OrnnError(f"{where}: HTTP {status}: "
                            f"{raw[:200].decode('utf-8', 'replace')!r}")
        if not raw:
            raise OrnnMcpProtocolError(f"{where}: empty response body")
        try:
            envelope = self._parse_body(raw)
        except ValueError as exc:
            raise OrnnMcpProtocolError(
                f"{where}: response was not JSON ({exc})") from exc
        if not isinstance(envelope, dict):
            raise OrnnMcpProtocolError(
                f"{where}: response was not a JSON-RPC envelope: "
                f"{str(envelope)[:200]!r}")
        if "error" in envelope:
            # JSON-RPC error level (below the tool's own isError): carry the
            # provider body verbatim.
            raise OrnnMcpProtocolError(
                f"{where}: {json.dumps(envelope['error'])[:300]}")
        return envelope.get("result")

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._rpc(
            "initialize", {
                "protocolVersion": self._JSONRPC_PROTOCOL,
                "capabilities": {},
                "clientInfo": {
                    "name": USER_AGENT.split(" ")[0],
                    "version": "0.1.0"
                },
            })
        self._rpc_id += 1
        # A notification (no id): the server keeps no session (probed), so
        # failures here must not wedge the client.
        body = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._credentials.access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }
        try:
            self._request_once("POST", self._base_url, headers, body)
        except OrnnError:
            pass
        self._initialized = True

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """tools/call -> the parsed ``content[0].text`` payload.

        The inner text is a JSON document on success; refusals are bare
        tokens (``not_found``, ``image_required``) or a message — mapped to
        typed errors that PRESERVE the provider body.
        """
        self._initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        where = f"tools/call {name}"
        if not isinstance(result, dict):
            raise OrnnMcpProtocolError(
                f"{where}: result was not an object: {str(result)[:200]!r}")
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise OrnnMcpProtocolError(
                f"{where}: result has no content: {str(result)[:200]!r}")
        first = content[0]
        if not isinstance(first, dict) or first.get("type") != "text":
            raise OrnnMcpProtocolError(
                f"{where}: content[0] is not text: {str(first)[:200]!r}")
        text = str(first.get("text") or "")
        if result.get("isError"):
            return self._refusal(name, text)
        try:
            return json.loads(text)
        except ValueError:
            # Bare-string payloads exist on refusal paths only; success
            # envelopes are JSON. Keep them loud and typed.
            return self._refusal(name, text, unparseable_success=True)

    def _refusal(self,
                 name: str,
                 text: str,
                 unparseable_success: bool = False) -> None:
        """Map a refusal body to the typed error that preserves it."""
        body = text.strip()
        where = f"tools/call {name}"
        if body == REFUSAL_IMAGE_REQUIRED:
            raise OrnnImageBlockedError(
                f"{where}: Ornn refused manual access activation with "
                f"'{REFUSAL_IMAGE_REQUIRED}' (reproduced 2026-10-08: neither "
                "the bid nor the activate call exposes an image argument in "
                "the live tools/list schema) — not retried; the lane's "
                "launch path is the bid's auto-launch")
        if body == REFUSAL_NOT_FOUND:
            raise OrnnNotFoundError(f"{where}: not_found")
        if unparseable_success:
            raise OrnnMcpProtocolError(
                f"{where}: success payload was not JSON: {body[:300]!r}")
        if body.startswith("Input validation error"):
            raise OrnnInputError(f"{where}: {body}")
        # Funding refusals, cli_route_not_allowed, and every unmapped body
        # stay distinguishable verbatim.
        raise OrnnError(f"{where}: {body[:500]}")

    # -- identity ----------------------------------------------------------

    def whoami(self) -> Dict[str, Any]:
        payload = self._call_tool("identity_whoami", {})
        return payload if isinstance(payload, dict) else {}

    # -- spot market -------------------------------------------------------

    def list_spot_books(self) -> List[Dict[str, Any]]:
        """All order books (``market_books``); ``bestAskPricePerGpuHourMicroUsd``
        null = no supply RIGHT NOW (single polls are never stock events)."""
        payload = self._call_tool("market_books", {})
        books = payload.get("orderBooks") if isinstance(payload, dict) else None
        if not isinstance(books, list):
            raise OrnnMcpProtocolError(
                "tools/call market_books: expected orderBooks list, got "
                f"{str(payload)[:200]!r}")
        return books

    def get_spot_book(self, gpu_slug: str, *, region: Optional[str] = None,
                     sale_type: Optional[str] = None) -> Dict[str, Any]:
        """One book's depth (``market_book``): asks/bids ``{gpus, orders,
        pricePerGpuHourMicroUsd}``.

        v2 keys a book by (gpuSlug, region, saleType) — the optional filters
        select between a slug's multiple books. UNMEASURED against the live
        endpoint (the catalog path reads market_books, not this).
        """
        arguments: Dict[str, Any] = {"gpuSlug": gpu_slug}
        if region is not None:
            arguments["region"] = region
        if sale_type is not None:
            arguments["saleType"] = sale_type
        payload = self._call_tool("market_book", arguments)
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                f"tools/call market_book({gpu_slug}): expected an object, "
                f"got {str(payload)[:200]!r}")
        return payload

    def list_spot_trades(self, gpu_slug: str, *, region: Optional[str] = None,
                         sale_type: Optional[str] = None) -> List[Dict[str, Any]]:
        arguments: Dict[str, Any] = {"gpuSlug": gpu_slug}
        if region is not None:
            arguments["region"] = region
        if sale_type is not None:
            arguments["saleType"] = sale_type
        payload = self._call_tool("market_trades", arguments)
        trades = payload.get("trades") if isinstance(payload, dict) else None
        return trades if isinstance(trades, list) else []

    def quote_market(
        self,
        gpu_slug: str,
        *,
        nodes: int = 1,
        sale_type: str = "on-demand",
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The free fill quote (``market_quote``): per-node capacity,
        price and region for a would-be bid — v2's preview surface.

        Live-measured 2026-11-06: the quote carries ``nodes[]`` with
        ``gpus``/``gpusPerNode``/``pricePerGpuHourMicroUsd``/``listingId``
        per node. v2 bids buy WHOLE MACHINES (``nodes``), never v1's
        1-8 GPU slice — the caller resolves the node count from this
        quote's own ``gpusPerNode``.
        """
        arguments: Dict[str, Any] = {
            "gpuSlug": gpu_slug,
            "nodes": int(nodes),
            "saleType": sale_type,
        }
        if region is not None:
            arguments["region"] = region
        payload = self._call_tool("market_quote", arguments)
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call market_quote: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    def place_spot_bid(
        self,
        gpu_slug: str,
        *,
        nodes: int = 1,
        sale_type: str = "on-demand",
        region: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Place a WHOLE-NODE market bid (``market_bid``); v2 of the API.

        v2 (live-measured 2026-11-06): bids buy ``nodes`` — whole
        machines — and on-demand market bids FILL INSTANTLY, the fill ID
        being the reservation ID (``order.fillIds[0]``). The v1
        confirm-gate is GONE server-side; ``market_quote`` is the free
        preview, so without ``confirm`` this returns the quote — never a
        spend (the lane's launch-refusal floor, rebuilt on v2's own
        probe).
        """
        if not confirm:
            return self.quote_market(gpu_slug, nodes=nodes,
                                     sale_type=sale_type, region=region)
        arguments: Dict[str, Any] = {
            "gpuSlug": gpu_slug,
            "nodes": int(nodes),
            "saleType": sale_type,
        }
        if region is not None:
            arguments["region"] = region
        if idempotency_key is not None:
            arguments["idempotencyKey"] = idempotency_key
        payload = self._call_tool("market_bid", arguments)
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call market_bid: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    def list_spot_orders(self) -> List[Dict[str, Any]]:
        payload = self._call_tool("market_orders", {})
        orders = payload.get("orders") if isinstance(payload, dict) else None
        if not isinstance(orders, list):
            raise OrnnMcpProtocolError(
                "tools/call market_orders: expected orders list, got "
                f"{str(payload)[:200]!r}")
        return orders

    def cancel_spot_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a RESTING bid (``market_cancel``; v2 rename — the
        confirm-gate is gone server-side, so every call is live)."""
        payload = self._call_tool("market_cancel", {"orderId": order_id})
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call market_cancel: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    def end_fill(self, fill_id: str) -> Dict[str, Any]:
        """End a FILLED bid (``market_end``): the v2 teardown — a fill ID
        is its reservation ID.

        Live-measured 2026-11-06: the call is ASYNC — it echoes the
        reservation record while the node disappears from
        ``storage_targets`` within ~30s — and it is NOT 404-idempotent
        like v1's cancel: a repeat while the end is still processing
        refuses with "Wait for conflicting operation". That refusal
        means the teardown is ALREADY IN FLIGHT — the caller's goal —
        so it is returned as data here (the typed refusal only fires
        for other bodies); an ended-fill repeat after processing also
        lands here rather than wedging teardown.
        """
        try:
            payload = self._call_tool("market_end", {"fillId": fill_id})
        except OrnnError as exc:
            if ("conflicting operation" in str(exc)):
                return {"status": "end_already_in_flight", "fillId": fill_id}
            raise
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call market_end: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    # -- reservations ------------------------------------------------------

    def list_reservations(self,
                          *,
                          kind: Optional[str] = None,
                          status: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """The account's reservations (``reservations_list``; the v1
        ``ornn_reservations_list`` rename). v2 rows carry the ID under
        ``reservationId`` — there is no ``id`` field (live-measured
        2026-11-06) — and the envelope paginates by ``cursor``.
        """
        arguments: Dict[str, Any] = {"limit": int(limit)}
        if kind is not None:
            arguments["kind"] = kind
        if status is not None:
            arguments["status"] = status
        payload = self._call_tool("reservations_list", arguments)
        rows = payload.get("reservations") if isinstance(payload,
                                                         dict) else None
        if not isinstance(rows, list):
            raise OrnnMcpProtocolError(
                "tools/call reservations_list: expected reservations "
                f"list, got {str(payload)[:200]!r}")
        return rows

    def show_reservation(self, reservation_id: str) -> Dict[str, Any]:
        payload = self._call_tool("reservations_show",
                                  {"reservationId": reservation_id})
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call reservations_show: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    def list_schedules(self,
                       *,
                       listing_kind: Optional[str] = None,
                       gpu_slug: Optional[str] = None,
                       state: str = "available",
                       page_limit: int = 100) -> List[Dict[str, Any]]:
        """Available listings (``listings_list``) — heavyweight; used ONLY as
        the zero-stock cross-check and never to buy capacity.

        v2 paginates by ``cursor``; pages are drained (bounded) so the
        cross-check never reasons from a truncated page. ``gpu_slug`` has no
        server-side filter in v2 — it post-filters rows locally.
        """
        rows: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        seen_cursors: set = set()
        pages = 0
        while True:
            arguments: Dict[str, Any] = {"state": state, "limit": page_limit}
            if listing_kind is not None:
                arguments["kind"] = listing_kind
            if cursor:
                arguments["cursor"] = cursor
            payload = self._call_tool("listings_list", arguments)
            page = None
            if isinstance(payload, dict):
                page = payload.get("listings")
            elif isinstance(payload, list):
                page = payload
            if not isinstance(page, list):
                raise OrnnMcpProtocolError(
                    "tools/call listings_list: expected a listings list, "
                    f"got {str(payload)[:200]!r}")
            rows.extend(r for r in page if isinstance(r, dict))
            pages += 1
            # An EMPTY page with a cursor is mid-scan, not the end: keep
            # draining (CodeRabbit on skypilot #88 round 1).
            cursor = None
            if isinstance(payload, dict):
                nxt = payload.get("nextCursor") or payload.get("cursor")
                cursor = str(nxt) if nxt else None
            if not cursor:
                break
            if cursor in seen_cursors:
                raise OrnnMcpProtocolError(
                    "tools/call listings_list: cursor "
                    f"{cursor[:32]!r} repeats — a server pagination loop, "
                    "never a listing set")
            if pages >= _LISTINGS_MAX_PAGES:
                raise OrnnMcpProtocolError(
                    f"tools/call listings_list: drained {_LISTINGS_MAX_PAGES} "
                    "pages with a cursor remaining — refusing to reason from "
                    "a truncated listing set")
            seen_cursors.add(cursor)
        if gpu_slug is not None:
            rows = [r for r in rows
                    if str(r.get("gpuSlug") or "") == gpu_slug]
        return rows

    # -- access ------------------------------------------------------------

    def show_access(self, reservation_id: str) -> List[Dict[str, Any]]:
        """The reservation's access nodes (``access_show``; the v1
        ``ornn_access_show`` rename).

        Live-measured 2026-11-06: this surface STAYS EMPTY for VM-mode
        reservations — two live reservations, hours of polling, zero
        nodes — while the boxes were up with sshd answering. The VM
        SSH surface is ``storage_targets`` (see
        ``list_storage_targets``); this call is kept for shape parity
        and a future server-side fix.
        """
        payload = self._call_tool("access_show",
                                  {"reservationId": reservation_id})
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(nodes, list):
            raise OrnnMcpProtocolError(
                "tools/call access_show: expected nodes list, got "
                f"{str(payload)[:200]!r}")
        return nodes

    def list_storage_targets(self) -> List[Dict[str, Any]]:
        """The account's node targets (``storage_targets``) — the MEASURED
        v2 SSH surface: each target carries ``nodeUsername`` and
        ``nodes[]`` with the live ``ip`` per node, keyed by
        ``reservationIds`` (live-measured 2026-11-06: a booted VM's
        address appears here — and nowhere in access_show — within
        minutes of ``access_activate``).
        """
        payload = self._call_tool("storage_targets", {})
        targets = payload.get("targets") if isinstance(payload, dict) else None
        if not isinstance(targets, list):
            raise OrnnMcpProtocolError(
                "tools/call storage_targets: expected targets list, got "
                f"{str(payload)[:200]!r}")
        return targets

    @staticmethod
    def target_for_reservation(targets: List[Dict[str, Any]],
                               reservation_id: str) -> Optional[Dict[str, Any]]:
        """The target row whose ``reservationIds`` name this reservation."""
        wanted = str(reservation_id or "").strip()
        for target in targets:
            if not isinstance(target, dict):
                continue
            ids = target.get("reservationIds")
            if isinstance(ids, list) and wanted in [
                    str(i) for i in ids if i is not None]:
                return target
        return None

    @staticmethod
    def target_node_endpoint(target: Dict[str, Any]) -> Tuple[str, str]:
        """The target's FIRST node (ip, username) — loudly refused when
        absent (never a guessed address for a box that is billing)."""
        nodes = target.get("nodes")
        node = next((n for n in (nodes or []) if isinstance(n, dict) and
                     str(n.get("ip") or "").strip()), None)
        if node is None:
            raise OrnnMcpProtocolError(
                f"storage target carries no node ip: {str(target)[:200]!r}")
        user = str(target.get("nodeUsername") or "").strip()
        if not user:
            raise OrnnMcpProtocolError(
                f"storage target carries no nodeUsername: {str(target)[:200]!r}")
        return str(node["ip"]).strip(), user

    def activate_access(
        self,
        reservation_id: str,
        ssh_key_ids: List[str],
        *,
        access_mode: str = "vm",
        machine_count: Optional[int] = None,
        network_mode: Optional[str] = "public",
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Boot the reservation's VMs (``access_activate``; the v1
        ``ornn_access_activate`` rename).

        v2 MEASURED (2026-11-06): activation is the step that BOOTS the
        machine — a filled bid alone yields a reservation with zero
        nodes — and the ``sshKeyIds`` passed HERE are the ones
        provisioned onto the box (keys added later do not reach a
        booted VM's authorized_keys). Async: returns ``{"nodes": []}``
        while the machine boots; the address surfaces in
        ``storage_targets``. The v1 confirm-gate and the v1
        ``image_required`` refusal are gone server-side; the v1
        auto-launch-on-fill is GONE too — activation is now REQUIRED
        in the launch path.
        """
        arguments: Dict[str, Any] = {
            "reservationId": reservation_id,
            "sshKeyIds": list(ssh_key_ids),
            "accessMode": access_mode,
        }
        if machine_count is not None:
            arguments["machineCount"] = machine_count
        if network_mode is not None:
            arguments["networkMode"] = network_mode
        if request_id is not None:
            arguments["requestId"] = request_id
        payload = self._call_tool("access_activate", arguments)
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call access_activate: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    # -- ssh keys ----------------------------------------------------------
    def list_ssh_keys(self) -> List[Dict[str, Any]]:
        payload = self._call_tool("ssh_keys_list", {"limit": 100})
        keys = payload.get("ssh_keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            raise OrnnMcpProtocolError(
                "tools/call ssh_keys_list: expected ssh_keys list, got "
                f"{str(payload)[:200]!r}")
        return keys

    def add_ssh_key(self,
                    public_key: str,
                    *,
                    label: str) -> Dict[str, Any]:
        """Register an ACCOUNT key (``ssh_keys_add``; the v1
        ``ornn_ssh_keys_add`` rename — v2 has no confirm gate, every
        call is live). The key reaches a VM only via the
        ``access_activate`` call's ``sshKeyIds`` (live-measured
        2026-11-06: later adds do not propagate to booted boxes)."""
        arguments: Dict[str, Any] = {"publicKey": public_key, "label": label}
        payload = self._call_tool("ssh_keys_add", arguments)
        if not isinstance(payload, dict):
            raise OrnnMcpProtocolError(
                "tools/call ssh_keys_add: expected an object, got "
                f"{str(payload)[:200]!r}")
        return payload

    # -- polling -----------------------------------------------------------

    def wait_until_ready(
        self,
        reservation_id: str,
        *,
        timeout_s: float = DEFAULT_PROVISION_TIMEOUT_S,
        poll_interval_s: float = MIN_POLL_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> Dict[str, Any]:
        """Poll until the reservation's node address surfaces, then
        return the storage-target row (``target_node_endpoint`` parses
        the ip/username).

        v2 MEASURED (2026-11-06): readiness is NOT ``access_show`` —
        that surface stays empty for VM-mode reservations even with the
        box up; the node address surfaces in ``storage_targets`` within
        minutes of ``access_activate``. Fails CLOSED on: a cancelled
        reservation (the market bid died beneath us — a capacity
        refusal to surface, never auto-retried), a completed
        reservation without a node (the v1 2026-10-08 wedge class,
        still fail-closed), and the deadline (the security-update first
        boot takes minutes; the default tolerates it, past that it
        fails loud).
        """
        if poll_interval_s < 5:
            raise OrnnError(
                f"poll_interval_s={poll_interval_s} is absurdly fast for "
                "Ornn's security-update first boot")
        deadline = now() + timeout_s
        last_status: Optional[str] = None
        while True:
            reservation = self.show_reservation(reservation_id)
            status = str(reservation.get("status") or "").strip().lower()
            if status != last_status:
                last_status = status
            targets = self.list_storage_targets()
            target = self.target_for_reservation(targets, reservation_id)
            if target is not None:
                try:
                    self.target_node_endpoint(target)
                    return target
                except OrnnMcpProtocolError:
                    # A target row without an address is mid-provision:
                    # a poll signal, not a verdict — keep polling under
                    # the deadline; anything else stays loud.
                    pass
            if status == STATUS_CANCELLED:
                raise OrnnCapacityRefusedError(
                    f"reservation {reservation_id} was cancelled while "
                    "waiting for the node to surface — a capacity "
                    "refusal at the current ask, never auto-retried")
            if status == STATUS_COMPLETED:
                # A completed reservation with no node can never become
                # ready on its own — the wedge class, fail-closed.
                raise OrnnError(
                    f"reservation {reservation_id} completed "
                    f"(endReason={reservation.get('endReason')!r}) without "
                    "ever exposing a node — the wedge class, fail-closed: "
                    "terminate and report, not retried")
            if now() >= deadline:
                raise OrnnError(
                    f"reservation {reservation_id} still "
                    f"{last_status or 'provisioning'!r} with no node "
                    f"after {timeout_s:.0f}s; refusing to wait longer "
                    "(security-update first boot tolerated, deadline "
                    "is the bound)")
            sleep(poll_interval_s)
