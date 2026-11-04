"""io.net provisioner glue: credentials + re-exports for the instance
module.

The HTTP client itself lives in ``sky/adaptors/ionet.py`` and every call
it makes carries an explicit timeout plus the custom User-Agent the
provider's Cloudflare demands (python-urllib's default UA is banned,
HTTP 403 Error 1010). This module owns only the provisioner-facing
conventions:

* the key is read from ``IO_NET_API_KEY`` (the task secret channel's
  spelling — the ONLY spelling; the vendor publishes no alias
  convention) with ``~/.ionet/api_key`` as the file fallback — mirrors
  QuantaCloud/Spheron/Vast/Latitude/Massed;
* the lane's fixed deploy knobs (the SSH key name, the VM image type)
  are named once, here.
"""

import os
from pathlib import Path
from typing import Optional

from sky.adaptors import ionet as ionet_api

# The name the lane's SSH key is uploaded under in the deploy body's
# ssh_keys map (name → MATERIAL; io.net has no key registry — the
# material rides inline with every deploy, so the name is only a label
# and no matching/ensure step exists, unlike MassedCompute).
SSH_KEY_NAME = 'skypilot'

# The VM image type every lane deployment boots: 'general' (Ubuntu 24.04
# + CUDA + NVIDIA drivers; the deploy body's vm_image_type enumeration,
# verified live 2026-11-03). Resolved from the adaptor's constant so the
# two can never drift.
DEFAULT_IMAGE_TYPE = ionet_api.DEFAULT_IMAGE_TYPE

# Re-exports: the instance module speaks these through ``utils.`` so the
# adapter boundary stays one line.
IonetClient = ionet_api.IonetClient
IonetError = ionet_api.IonetError
IonetAuthError = ionet_api.IonetAuthError
IonetNotFoundError = ionet_api.IonetNotFoundError
STATUS_RUNNING = ionet_api.STATUS_RUNNING


def client_from_env() -> IonetClient:
    """A client with the key from the env or the credential file.

    Reads the env FIRST (the controller's task secret channel), then
    ``~/.ionet/api_key`` (the local store). An empty key raises
    IonetAuthError inside the client — never a client that would answer
    every authenticated call with 401/403 and read as "no capacity".
    """
    key = os.environ.get(ionet_api.API_KEY_ENV, '').strip()
    if not key:
        path = Path(ionet_api.API_KEY_FILE).expanduser()
        if path.is_file():
            key = path.read_text(encoding='utf-8').strip()
    return IonetClient(key)


def resolve_api_key() -> Optional[str]:
    """The key from env or credential file, or None (for credential bridges)."""
    key = os.environ.get(ionet_api.API_KEY_ENV, '').strip()
    if key:
        return key
    path = Path(ionet_api.API_KEY_FILE).expanduser()
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    return None