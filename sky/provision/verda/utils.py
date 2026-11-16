"""Verda provisioner glue: credentials + re-exports for the instance
module.

The HTTP client itself lives in ``sky/adaptors/verda.py`` and every call
it makes carries an explicit timeout. This module owns only the
provisioner-facing conventions:

* the OAuth2 client credentials are read from ``VERDA_CLIENT_ID`` +
  ``VERDA_CLIENT_SECRET`` (the task secret channel's spelling) with
  ``~/.verda/credentials`` (JSON ``{"client_id": ..., "client_secret":
  ...}``) as the file fallback — env first, then file, mirroring the
  sibling lanes;
* the lane's fixed deploy knobs (ssh key name, default image) are named
  once, here.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from sky.adaptors import verda as verda_api

# The name the lane's SSH key is registered under. ensure_ssh_key matches
# on key MATERIAL (the key list returns the material itself), so a rename
# never mints a duplicate and a re-created keypair never selects a stale
# key.
SSH_KEY_NAME = 'skypilot'

# The OS image slug the provisioner deploys (GET /v1/images examples:
# ubuntu-24.04-cuda-12.8-open-docker style slugs).
DEFAULT_IMAGE = 'ubuntu-24.04-cuda-12.8-open-docker'

# Re-exports: the instance module speaks these through ``utils.`` so the
# adapter boundary stays one line.
VerdaClient = verda_api.VerdaClient
VerdaError = verda_api.VerdaError
VerdaNotFoundError = verda_api.VerdaNotFoundError
STATUS_RUNNING = verda_api.STATUS_RUNNING
STATUS_OFFLINE = verda_api.STATUS_OFFLINE
STATUS_DISCONTINUED = verda_api.STATUS_DISCONTINUED
STATUS_DELETING = verda_api.STATUS_DELETING
STATUS_NOTFOUND = verda_api.STATUS_NOTFOUND
instance_status = verda_api.VerdaClient.instance_status
public_ip = verda_api.VerdaClient.public_ip


def resolve_credentials() -> Optional[Tuple[str, str]]:
    """(client_id, client_secret) from env or the credential file.

    Env FIRST (the controller's task secret channel), then
    ``~/.verda/credentials`` (JSON {"client_id", "client_secret"}).
    Returns None when neither source is configured — the caller decides
    how to fail loud.
    """
    client_id = os.environ.get(verda_api.CLIENT_ID_ENV, '').strip()
    client_secret = os.environ.get(verda_api.CLIENT_SECRET_ENV, '').strip()
    if client_id and client_secret:
        return client_id, client_secret
    path = Path(verda_api.CREDENTIALS_FILE).expanduser()
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            raise verda_api.VerdaAuthError(
                f'{path} is not valid JSON; expected '
                '{"client_id": ..., "client_secret": ...}')
        if isinstance(payload, dict):
            client_id = str(payload.get('client_id') or '').strip()
            client_secret = str(payload.get('client_secret') or '').strip()
            if client_id and client_secret:
                return client_id, client_secret
    return None


def client_from_env() -> VerdaClient:
    """A client with the credentials from env or the credential file.

    Raises VerdaAuthError when credentials are missing — never a client
    that would answer every call with 401 and read as "no capacity".
    """
    creds = resolve_credentials()
    if creds is None:
        raise verda_api.VerdaAuthError(
            'Verda credentials are not configured: neither '
            f'{verda_api.CLIENT_ID_ENV}/{verda_api.CLIENT_SECRET_ENV} '
            f'nor {verda_api.CREDENTIALS_FILE} is set. Refusing to make '
            'unauthenticated calls that would read as "no capacity".')
    client_id, client_secret = creds
    return VerdaClient(client_id, client_secret)
