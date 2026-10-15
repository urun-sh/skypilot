"""Packet.ai provisioner glue: credentials + re-exports for the
instance module.

The HTTP client itself lives in ``sky/adaptors/packet.py`` and every call
it makes carries an explicit timeout. This module owns only the
provisioner-facing conventions:

* the key is read from ``PACKET_API_KEY`` (the task secret channel's
  spelling — the ONLY spelling; the vendor CLI documents this env var but
  does NOT implement it, so nothing routes through the CLI) with
  ``~/.packet/credentials`` (the skill manual's store, mode 600) as the
  file fallback — mirrors Spheron/Vast/Latitude/Quantacloud;
* the lane's fixed deploy knobs (ssh key name, the create-path status) are
  named once, here.
"""

import os
from pathlib import Path
from typing import Optional

from sky.adaptors import packet as packet_api

# The name the lane's SSH key is registered under. ensure_ssh_key matches
# on key MATERIAL (via the MD5 hex-colon fingerprint derived from the
# public key blob — the format the provider's key list MEASURES, not the
# SHA256 form), so a rename never mints a duplicate and a re-created
# keypair with the same name never selects a stale key.
SSH_KEY_NAME = 'skypilot'

# Re-exports: the instance module speaks these through ``utils.`` so the
# adapter boundary stays one line.
PacketClient = packet_api.PacketClient
PacketError = packet_api.PacketError
PacketAuthError = packet_api.PacketAuthError
PacketNotFoundError = packet_api.PacketNotFoundError
PacketCreateNotImplementedError = packet_api.PacketCreateNotImplementedError
PacketUnavailableError = packet_api.PacketUnavailableError
STATUS_RUNNING = packet_api.STATUS_RUNNING
STATUS_STOPPED = packet_api.STATUS_STOPPED
STATUS_ERROR = packet_api.STATUS_ERROR
STATUS_FAILED = packet_api.STATUS_FAILED
instance_status = packet_api.PacketClient.instance_status
instance_name = packet_api.PacketClient.instance_name
ssh_user_from_connection = packet_api.PacketClient.ssh_user_from_connection


def client_from_env() -> PacketClient:
    """A client with the key from the env or the credential file.

    Reads the env FIRST (the controller's task secret channel), then
    ``~/.packet/credentials`` (the manual's local store). An empty key
    raises PacketAuthError — never a client that would answer every
    authenticated call with 401 and read as "no capacity".
    """
    key = os.environ.get(packet_api.API_KEY_ENV, '').strip()
    if not key:
        path = Path(packet_api.API_KEY_FILE).expanduser()
        if path.is_file():
            key = path.read_text(encoding='utf-8').strip()
    return PacketClient(key)


def resolve_api_key() -> Optional[str]:
    """The key from env or credential file, or None (for credential bridges)."""
    key = os.environ.get(packet_api.API_KEY_ENV, '').strip()
    if key:
        return key
    path = Path(packet_api.API_KEY_FILE).expanduser()
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    return None
