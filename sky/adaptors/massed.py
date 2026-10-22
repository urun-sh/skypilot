"""MassedCompute REST client.

MassedCompute (https://massedcompute.com) rents on-demand NVIDIA GPU VMs
from a prepaid balance. The automation surface is a plain-JSON REST API
under ``https://vm.massedcompute.com/api/v1`` (docs:
https://vm-docs.massedcompute.com/api/v1 — customer API traffic goes to
``vm.massedcompute.com``, NEVER ``api.massedcompute.com``):

* **Bearer token auth on every endpoint.** ``Authorization: Bearer
  <token>``; the token is provisioned in the console's API settings and
  validated live via ``POST /account/token/validation`` (probed
  2026-10-21: ``{"message": "Valid Token"}``).
* **Products, not offers.** ``GET /gpu-inventory`` keys the inventory by
  STABLE product name (``gpu_1x_pro_6000_blackwell`` = 1x RTX PRO 6000
  Blackwell 96GB at 219 cents/hr); the name is the launch API's
  ``productName`` and is the catalog identity. Unlike QuantaCloud there is
  no ephemeral offer UUID to re-resolve at launch.
* **Instances carry names.** The launch body takes ``instanceName`` and
  ``GET /instance`` lists every running instance with its name, so the
  cluster's identity is server-side (``instanceName``) — no local
  cluster-identity store (the QuantaCloud wrinkle does not exist here).
* **SSH keys are registered by NAME.** The launch body's ``sshKeys`` takes
  key NAMES registered on the account (``POST /ssh-keys`` with
  ``{name, publicKey}``), and ``GET /ssh-keys`` returns the ``public_key``
  MATERIAL, so key identity is matchable locally without a fingerprint
  derivation. The instance also reports ``username``/``password`` (password
  SSH); the lane uses the registered key, as every sibling lane does.
* **Boot readiness is a flag.** ``GET /instance/{uuid}`` reports
  ``os_booted`` (0/1) plus ``ip``; readiness is ``os_booted == 1`` and a
  non-empty ``ip``. A just-launched instance can 404 briefly (the
  read-after-write window, the QuantaCloud ENG-575 lesson), so the boot
  poll tolerates 404 until its deadline.
* **Terminate is the only teardown.** ``POST /instance/terminate`` destroys
  the VM and its data; there is a restart but no stop-and-keep, so a
  ``stop`` that silently terminated would destroy while the caller believed
  it preserved (declared unsupported at the cloud layer instead).

Wire protocol notes that differ from the sibling brokers (each verified
against https://vm-docs.massedcompute.com/api/v1 on 2026-10-21):

* **Plain JSON**, not JSON:API. Successes are a single envelope object per
  endpoint (``gpu_inventory``, ``images``, ``sshKeys``, ``runningInstances``,
  ``runningInstance``, ``sshKey``, ``response``); launch/terminate answer
  ``202`` with the uuid under ``response``.
* **No documented rate budget.** 429s are retried defensively (bounded
  exponential, three strikes) but no budget constants are published.
* **API-key env spelling**: ``MASSED_API_KEY`` is the only spelling this
  stack reads (the controller's task secret channel); the vendor publishes
  no alternate env convention, so there is no alias to normalize (unlike
  Vast's VAST_AI_API_KEY).
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.error
import urllib.request

API_BASE = 'https://vm.massedcompute.com/api/v1'

# Identifies the integration to the provider's logs.
USER_AGENT = 'urun-skypilot-massed/0.1 (+https://urun.sh)'

# One timeout for every request, including the reads inside poll loops —
# the point is the bound, not the value.
DEFAULT_TIMEOUT_S = 60

# No rate budget is published for this API; 429 handling is defensive only.
MAX_RATE_LIMIT_RETRIES = 3
MAX_RETRY_AFTER_S = 60.0

# The plain Ubuntu Server image with NVIDIA drivers that every lane
# deployment boots (measured against GET /images 2026-10-21: vm_image_id
# 84, "Ubuntu Server 22.04 w/ drivers"). Resolved LIVE by name at launch —
# the numeric id can drift; the name is the stable key.
DEFAULT_IMAGE_NAME = 'Ubuntu Server 22.04 w/ drivers'

# The launch API's region selector: 'any' lets the provider place the VM
# wherever the product has capacity; a catalog Region token (us-east-1,
# us-central-9, ...) pins it.
REGION_ANY = 'any'

# Observed instance status values (docs + live GET /instance 2026-10-21):
# 'rented' while alive, 'booting' transitional, 'terminated' in the
# terminate response. The steady-state signal this client trusts is
# `os_booted == 1` plus a non-empty ip, not the status string (the set is
# not documented as closed).
STATUS_RENTED = 'rented'
STATUS_BOOTING = 'booting'
STATUS_TERMINATED = 'terminated'

# The provider's documented poll cadence is minutes, not seconds (a GPU VM
# boots an OS + drivers); polling faster is budget waste.
MIN_POLL_INTERVAL_S = 10

# ~20-minute worst-case boot (OS + NVIDIA drivers) plus margin.
DEFAULT_PROVISION_TIMEOUT_S = 20 * 60

# Environment variable the callers read the key from (the task secret
# channel's spelling; the vendor publishes no alias convention).
API_KEY_ENV = 'MASSED_API_KEY'
API_KEY_FILE = '~/.massed/api_key'


class MassedError(RuntimeError):
    """Any MassedCompute API failure. Never raised with the API key in it.

    Carries the provider's own error code when the response had one, so
    callers can branch on it without re-parsing.
    """

    def __init__(self, message: str, code: str = ''):
        super().__init__(message)
        self.code = code


class MassedAuthError(MassedError):
    """401 — key missing, invalid, or revoked. Check/rotate it."""


class MassedRateLimited(MassedError):
    """429 that exhausted its retries. No documented body hint; bounded
    exponential backoff already ran."""


class MassedNotFoundError(MassedError):
    """404 — the resource is gone. Terminate paths treat this as success."""


def _error_envelope(raw: bytes) -> Dict[str, Any]:
    """The provider's error object from a body, or {} if unusable.

    Errors are documented as ``{"message": ..., "code": ...}``; a body that
    is not a JSON object must not crash the error path itself.
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MassedClient:
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
            raise MassedAuthError(
                'MassedCompute API key is empty; refusing to make '
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
                # stalled body raises TimeoutError/OSError directly and would
                # escape every caller that only catches MassedError (the
                # trap CodeRabbit flagged on the Spheron client).
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read()
            except (TimeoutError, OSError) as read_exc:
                raise MassedError(f'{method} {path} failed reading the error '
                                  f'body: {read_exc}') from read_exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise MassedError(f'{method} {path} failed: {exc.reason}') from exc
        except (TimeoutError, OSError) as exc:
            raise MassedError(f'{method} {path} failed: {exc}') from exc

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
    ) -> Any:
        """One authenticated JSON call. Every endpoint takes the Bearer
        token (there are no public endpoints on this API)."""
        url = f'{self._base_url}{path}'
        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
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
            # JSONDecodeError), so no raw exception escapes a
            # MassedError-only caller.
            except ValueError as exc:
                raise MassedError(
                    f'{where}: response was not JSON ({exc})') from exc

        err = _error_envelope(raw)
        code = str(err.get('code') or '')
        message = str(err.get('message') or '')[:300]
        if not code and not message:
            # An error with no usable envelope still must not collapse to a
            # bare status line: show the raw prefix instead.
            message = raw[:200].decode('utf-8', 'replace')

        if status == 401:
            raise MassedAuthError(
                f'{where}: HTTP 401. Check {API_KEY_ENV} (or rotate the key '
                'in the MassedCompute console under Settings → API).')
        if status == 404:
            raise MassedNotFoundError(f'{where}: not found ({message})')
        if status == 429:
            default = code or 'rate_limited'
            raise MassedRateLimited(
                f'{where}: rate limited ({default}); detail: {message}')
        default = code or 'error'
        raise MassedError(
            f'{where}: HTTP {status}: {default}: {message}',
            code=code,
        )

    # -- account ----------------------------------------------------------

    def validate_token(self) -> Dict[str, Any]:
        """POST /account/token/validation — the credential probe.

        The provider answers ``{"message": "Valid Token"}`` for a live key;
        a bad key raises 401 (MassedAuthError) here.
        """
        payload = self._call('POST', '/account/token/validation', body={})
        if not isinstance(payload, dict):
            raise MassedError('POST /account/token/validation: expected an '
                              f'object, got {type(payload).__name__}')
        return payload

    # -- catalog ----------------------------------------------------------

    def list_gpu_inventory(self) -> Dict[str, Dict[str, Any]]:
        """GET /gpu-inventory — the products map keyed by product name.

        One entry per rentable product (name, description, hourly price in
        CENTS, vCPU/RAM/storage specs, the regions with capacity, and the
        total ``capacity_available``). This is the fetcher's entire input.
        """
        payload = self._call('GET', '/gpu-inventory') or {}
        if not isinstance(payload, dict):
            raise MassedError('GET /gpu-inventory: expected an object, got '
                              f'{type(payload).__name__}')
        inventory = payload.get('gpu_inventory')
        if not isinstance(inventory, dict):
            raise MassedError(
                'GET /gpu-inventory: expected a map at .gpu_inventory, got '
                f'{type(inventory).__name__}')
        return {
            str(name): row
            for name, row in inventory.items()
            if isinstance(row, dict)
        }

    def list_images(self) -> List[Dict[str, Any]]:
        """GET /images — the bootable images list."""
        payload = self._call('GET', '/images') or {}
        if not isinstance(payload, dict):
            raise MassedError('GET /images: expected an object, got '
                              f'{type(payload).__name__}')
        rows = payload.get('images')
        if not isinstance(rows, list):
            raise MassedError('GET /images: expected a list at .images, '
                              f'got {type(rows).__name__}')
        return [row for row in rows if isinstance(row, dict)]

    def default_image_id(self) -> int:
        """The numeric id of the lane's default boot image, resolved LIVE.

        The launch API wants the numeric ``imageId``; the ids drift (the
        list is a live surface), so the NAME is the stable key and the id
        is re-resolved on every launch. A missing name is a loud error —
        never a silent fallback to some other image (a wrong image boots a
        box without drivers and the launch fails far from the cause).
        """
        for image in self.list_images():
            if str(image.get('vm_image_name') or
                   '').strip() == DEFAULT_IMAGE_NAME:
                image_id = image.get('vm_image_id')
                if isinstance(image_id, int) and not isinstance(image_id, bool):
                    return image_id
        raise MassedError(
            f'GET /images carries no {DEFAULT_IMAGE_NAME!r} image with a '
            'numeric vm_image_id; the lane refuses to launch on a guessed '
            'image')

    # -- ssh keys ---------------------------------------------------------

    def list_ssh_keys(self) -> List[Dict[str, Any]]:
        """GET /ssh-keys — the account's registered keys.

        Unlike QuantaCloud (whose key list returns only fingerprints),
        this list returns the ``public_key`` MATERIAL, so key identity is
        an exact local string comparison.
        """
        payload = self._call('GET', '/ssh-keys') or {}
        if not isinstance(payload, dict):
            raise MassedError('GET /ssh-keys: expected an object, got '
                              f'{type(payload).__name__}')
        rows = payload.get('sshKeys')
        if not isinstance(rows, list):
            raise MassedError('GET /ssh-keys: expected a list at .sshKeys, '
                              f'got {type(rows).__name__}')
        return [row for row in rows if isinstance(row, dict)]

    def create_ssh_key(self, name: str, public_key: str) -> Dict[str, Any]:
        """POST /ssh-keys — register a key under ``name``.

        The body is ``{"name", "publicKey"}`` (camelCase, per the docs);
        the answer is ``{"sshKey": {"id", "name"}}``.
        """
        payload = self._call(
            'POST',
            '/ssh-keys',
            body={
                'name': name,
                'publicKey': public_key,
            },
        ) or {}
        if not isinstance(payload, dict):
            raise MassedError('POST /ssh-keys: expected an object, got '
                              f'{type(payload).__name__}')
        ssh_key = payload.get('sshKey')
        if not isinstance(ssh_key, dict) or not ssh_key.get('id'):
            raise MassedError(
                f'POST /ssh-keys returned no sshKey.id: {sorted(payload)}')
        return ssh_key

    def ensure_ssh_key(self, name: str, public_key: str) -> str:
        """The NAME under which ``public_key`` is registered, adding it if
        absent.

        Matching on key MATERIAL (not name): a re-created controller
        keypair must not mint a duplicate, and a renamed key must not
        select a stale one. The launch API takes key NAMES, so the return
        is the name the material is registered under — the existing key's
        name when the material is already on the account (a re-deploy with
        a renamed key reuses the registered name), or the requested name
        after a create.
        """
        material = str(public_key or '').strip()
        if not material:
            raise MassedError('refusing to register an empty public key')
        for key in self.list_ssh_keys():
            if str(key.get('public_key') or '').strip() == material:
                existing = str(key.get('name') or '').strip()
                return existing or name
        self.create_ssh_key(name, material)
        return name

    # -- instances ----------------------------------------------------------

    def list_instances(self) -> List[Dict[str, Any]]:
        """GET /instance — every running instance on the account.

        This is the server-side cluster identity source: instances carry
        their ``name`` (the launch body's ``instanceName``), so the
        provisioner finds the cluster's box by name — no local identity
        store (the QuantaCloud wrinkle).
        """
        payload = self._call('GET', '/instance') or {}
        if not isinstance(payload, dict):
            raise MassedError('GET /instance: expected an object, got '
                              f'{type(payload).__name__}')
        rows = payload.get('runningInstances')
        if not isinstance(rows, list):
            raise MassedError(
                'GET /instance: expected a list at .runningInstances, got '
                f'{type(rows).__name__}')
        return [row for row in rows if isinstance(row, dict)]

    def get_instance(self, instance_uuid: str) -> Optional[Dict[str, Any]]:
        """GET /instance/{uuid} — the instance, None while it materializes.

        Measured live 2026-10-22 (the massed claim-05): the vendor answers
        the FIRST status read after the launch's 202 with HTTP 200 and
        ``runningInstance: null`` — NOT a 404 — so the read-after-write
        window has TWO shapes (200-with-null and 404) and the null must
        read as "not visible yet", never a parse failure: the first live
        launch's wait died on exactly this and the cleanly-launched box
        had to be torn down. The wait loop tolerates the None until the
        deadline, exactly like the 404.
        """
        payload = self._call('GET', f'/instance/{instance_uuid}') or {}
        if not isinstance(payload, dict):
            raise MassedError(
                f'GET /instance/{instance_uuid}: expected an object, got '
                f'{type(payload).__name__}')
        if 'runningInstance' not in payload:
            # The field ABSENT is a malformed response (an empty body or a
            # schema drift), not the materialization window: fail loud
            # rather than poll the deadline on garbage (CodeRabbit round).
            raise MassedError(
                f'GET /instance/{instance_uuid}: the payload has no '
                f'runningInstance field; refusing to guess what the vendor '
                f'meant by {sorted(payload)[:5]}')
        instance = payload.get('runningInstance')
        if instance is None:
            # Present but null: the materialization window (measured live
            # 2026-10-22, claim-05) — not visible YET, never a failure.
            return None
        if not isinstance(instance, dict):
            raise MassedError(
                f'GET /instance/{instance_uuid}: expected an object at '
                f'.runningInstance, got {type(instance).__name__}')
        return instance

    def launch_instance(
        self,
        *,
        image_id: int,
        product_name: str,
        region_name: str = REGION_ANY,
        instance_name: str = '',
        ssh_key_names: Optional[List[str]] = None,
        coupon: str = '',
        command: str = '',
        volume_ids: Optional[List[str]] = None,
    ) -> str:
        """POST /instance/launch — rent one VM; returns its uuid.

        The body is the provider's documented shape (camelCase keys):
        ``imageId`` (numeric), ``productName`` (the stable product name,
        e.g. ``gpu_1x_pro_6000_blackwell``), ``regionName`` (``'any'`` or a
        region token), ``instanceName``, ``sshKeys`` (the NAMES registered
        on the account), and the optional ``coupon``/``command``/
        ``volumeIds``. The answer is ``202 {"response": "<uuid>"}``.
        """
        if not product_name or not str(product_name).strip():
            raise MassedError(
                'refusing to launch without a productName (the stable '
                'catalog product name, e.g. gpu_1x_pro_6000_blackwell)')
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise MassedError(f'refusing to launch with a non-numeric imageId '
                              f'{image_id!r}')
        body: Dict[str, Any] = {
            'imageId': image_id,
            'productName': str(product_name).strip(),
            'regionName': str(region_name or REGION_ANY).strip() or REGION_ANY,
            'instanceName': str(instance_name or '').strip(),
            'coupon': str(coupon or ''),
            'command': str(command or ''),
            'sshKeys': [str(k) for k in (ssh_key_names or [])],
        }
        if volume_ids:
            body['volumeIds'] = [str(v) for v in volume_ids]
        payload = self._call('POST', '/instance/launch', body=body) or {}
        if not isinstance(payload, dict):
            raise MassedError('POST /instance/launch: expected an object, '
                              f'got {type(payload).__name__}')
        uuid = str(payload.get('response') or '').strip()
        if not uuid:
            raise MassedError(
                f'POST /instance/launch returned no uuid at .response: '
                f'{sorted(payload)}')
        return uuid

    def terminate_instances(self, instance_uuids: List[str]) -> None:
        """POST /instance/terminate — destroy the VMs and their data.

        Idempotent teardown: a 404 for an already-gone instance is
        success (the box is already destroyed), which is what makes
        reconcile safe to retry.
        """
        if not instance_uuids:
            return
        body = {'instanceUuids': [str(u) for u in instance_uuids]}
        try:
            self._call('POST', '/instance/terminate', body=body)
        except MassedNotFoundError:
            pass

    # -- accessors ---------------------------------------------------------

    @staticmethod
    def instance_uuid(instance: Dict[str, Any]) -> str:
        return str(instance.get('uuid') or '').strip()

    @staticmethod
    def instance_name(instance: Dict[str, Any]) -> str:
        return str(instance.get('name') or '').strip()

    @staticmethod
    def instance_ip(instance: Dict[str, Any]) -> str:
        return str(instance.get('ip') or '').strip()

    @staticmethod
    def instance_status(instance: Dict[str, Any]) -> str:
        return str(instance.get('status') or '').strip().lower()

    @staticmethod
    def instance_ssh_user(instance: Dict[str, Any]) -> Optional[str]:
        """The SSH user the PROVIDER reports for this instance.

        The docs' example shows ``"Ubuntu"``; the lane treats the live
        value as the source of truth (the Latitude lesson: docs can say
        root while the box answers ubuntu) and falls back to the
        provisioner's SSH_USER constant only when the field is absent.
        """
        user = str(instance.get('username') or '').strip()
        return user or None

    @staticmethod
    def instance_booted(instance: Dict[str, Any]) -> bool:
        """Boot readiness: ``os_booted == 1`` AND a non-empty ip.

        The flag alone is not enough — the instance must also carry the
        address the provisioner will SSH to.
        """
        booted = instance.get('os_booted')
        if isinstance(booted, bool):
            flag = booted
        elif isinstance(booted, int):
            flag = booted == 1
        else:
            flag = False
        return flag and bool(MassedClient.instance_ip(instance))

    # -- polling -----------------------------------------------------------

    def wait_until_booted(
        self,
        instance_uuid: str,
        *,
        timeout_s: float = DEFAULT_PROVISION_TIMEOUT_S,
        poll_interval_s: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> Dict[str, Any]:
        """Poll until the instance is booted with an ip, or fail closed.

        A 404 inside the poll loop is tolerated until the deadline: a
        JUST-LAUNCHED instance can 404 on its first status read (the
        read-after-write window server-side — the QuantaCloud ENG-575
        lesson), so the first NotFound must not read as "deleted". The
        deadline still fails closed: a stuck boot must become a failed
        runner, not an endless wait.
        """
        if poll_interval_s < MIN_POLL_INTERVAL_S:
            raise MassedError(
                f'poll_interval_s={poll_interval_s} is below the documented '
                f'floor of {MIN_POLL_INTERVAL_S}s')
        deadline = now() + timeout_s
        while True:
            try:
                instance = self.get_instance(instance_uuid)
            except MassedNotFoundError:
                instance = None
            if instance is None:
                # The read-after-write window has TWO shapes (the 200 with
                # runningInstance: null AND the 404) — both mean "not
                # visible yet"; tolerate either until the deadline.
                if now() >= deadline:
                    raise MassedError(
                        f'instance {instance_uuid} not found after '
                        f'{timeout_s:.0f}s; refusing to wait longer') from None
                sleep(poll_interval_s)
                continue
            if self.instance_booted(instance):
                return instance
            if now() >= deadline:
                status = repr(self.instance_status(instance) or 'booting')
                raise MassedError(
                    f'instance {instance_uuid} still {status} without an ip '
                    f'after {timeout_s:.0f}s; refusing to wait longer')
            sleep(poll_interval_s)
