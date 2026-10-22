"""MassedCompute provisioner glue: credentials + re-exports for the
instance module.

The HTTP client itself lives in ``sky/adaptors/massed.py`` and every call
it makes carries an explicit timeout. This module owns only the
provisioner-facing conventions:

* the key is read from ``MASSED_API_KEY`` (the task secret channel's
  spelling — the ONLY spelling; the vendor publishes no alias convention)
  with ``~/.massed/api_key`` as the file fallback — mirrors
  QuantaCloud/Spheron/Vast/Latitude;
* the lane's fixed deploy knobs (the SSH key name) are named once, here.
"""

import os
from pathlib import Path
from typing import Optional

from sky.adaptors import massed as massed_api

# The name the lane's SSH key is registered under. The launch API takes
# key NAMES, and ensure_ssh_key matches on key MATERIAL (the key list
# returns the public_key), so a rename never mints a duplicate and a
# re-created keypair never selects a stale key.
SSH_KEY_NAME = 'skypilot'

# Re-exports: the instance module speaks these through ``utils.`` so the
# adapter boundary stays one line.
MassedClient = massed_api.MassedClient
MassedError = massed_api.MassedError
MassedAuthError = massed_api.MassedAuthError
MassedNotFoundError = massed_api.MassedNotFoundError
REGION_ANY = massed_api.REGION_ANY


def client_from_env() -> MassedClient:
    """A client with the key from the env or the credential file.

    Reads the env FIRST (the controller's task secret channel), then
    ``~/.massed/api_key`` (the local store). An empty key raises
    MassedAuthError inside the client — never a client that would answer
    every authenticated call with 401 and read as "no capacity".
    """
    key = os.environ.get(massed_api.API_KEY_ENV, '').strip()
    if not key:
        path = Path(massed_api.API_KEY_FILE).expanduser()
        if path.is_file():
            key = path.read_text(encoding='utf-8').strip()
    return MassedClient(key)


def resolve_api_key() -> Optional[str]:
    """The key from env or credential file, or None (for credential bridges)."""
    key = os.environ.get(massed_api.API_KEY_ENV, '').strip()
    if key:
        return key
    path = Path(massed_api.API_KEY_FILE).expanduser()
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    return None