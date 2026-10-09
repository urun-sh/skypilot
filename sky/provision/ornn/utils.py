"""Ornn provisioner glue: credentials, identity store, re-exports.

The MCP client lives in ``sky/adaptors/ornn.py`` and every call it makes
carries an explicit timeout. This module owns the provisioner-facing
conventions:

* the credential is the OAuth JSON blob ``ORNN_MCP_CREDENTIALS`` with
  ``~/.ornn/mcp-credentials`` (mode 600) as the file fallback — no REST
  API key exists on Ornn, and a refresh-token-only env CANNOT refresh (the
  exchange requires client_id + client_secret);
* **attribution lives in a local identity store**: the bid API has NO
  name/hostname field, so ``run_instances`` records cluster_name_on_cloud ->
  reservation id in ``~/.sky/ornn-clusters.json``. An UNATTRIBUTED
  reservation is invisible to reconcile/terminate by design (the ownership
  gate) — never guess-and-delete a reservation we did not record.

Also: the SSH login is ``tenant`` per the Ornn docs (MEASURE on first
healthy fill — both 2026-10-08 wedges died before SSH; the Latitude
lesson is that the manual can lie).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from sky.adaptors import ornn as ornn_api

# The SSH login the Ornn docs name for spot VMs. UNMEASURED 2026-10-08 (both
# measurement bids wedged before SSH): docs say `tenant`; measure on the
# first healthy fill and correct here (the Latitude manual said root and
# the image answered as ubuntu).
SSH_USER = "tenant"

# The label the lane's account SSH key is registered under (matched by key
# MATERIAL, not name — spot VMs attach every active account key at launch).
SSH_KEY_LABEL = "skypilot"

# The single region token (a namespace column: the bid API takes no
# region — ``gpuSlug`` + ``gpus`` only).
REGION = "ornn"

# The local attribution store: cluster_name_on_cloud -> reservation id.
# The bid API has no name field; this file is the ONLY mapping from a
# SkyPilot cluster to its Ornn reservation.
CLUSTERS_FILE = "~/.sky/ornn-clusters.json"

# Re-exports: the instance module speaks these through `utils.` so the
# adapter boundary stays one line.
OrnnClient = ornn_api.OrnnClient
OrnnCredentials = ornn_api.OrnnCredentials
OrnnError = ornn_api.OrnnError
OrnnAuthError = ornn_api.OrnnAuthError
OrnnNotFoundError = ornn_api.OrnnNotFoundError
OrnnCapacityRefusedError = ornn_api.OrnnCapacityRefusedError
OrnnImageBlockedError = ornn_api.OrnnImageBlockedError
STATUS_ACTIVE = ornn_api.STATUS_ACTIVE
STATUS_COMPLETED = ornn_api.STATUS_COMPLETED
STATUS_CANCELLED = ornn_api.STATUS_CANCELLED


def client_from_env() -> OrnnClient:
    """A client with the credential from the env JSON blob or the files."""
    return OrnnClient(OrnnCredentials.from_env())


def resolve_credentials() -> Optional[Dict[str, Any]]:
    """The credential blob from the env or files, or None (bridges)."""
    try:
        credentials = OrnnCredentials.from_env()
    except OrnnAuthError:
        return None
    return {
        "client_id": credentials._client_id,  # pylint: disable=protected-access
        "refresh_token": credentials._refresh_token,  # pylint: disable=protected-access
    }


# -- identity store --------------------------------------------------------


def _clusters_path() -> Path:
    return Path(CLUSTERS_FILE).expanduser()


def load_state() -> Dict[str, Dict[str, Any]]:
    """The attribution store ({} when absent or corrupt — never raises)."""
    path = _clusters_path()
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(blob, dict):
        return {}
    return {
        str(name): row
        for name, row in blob.items()
        if isinstance(row, dict) and row.get("reservation_id")
    }


def save_state(state: Dict[str, Dict[str, Any]]) -> None:
    path = _clusters_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.chmod(path, 0o600)


def remember_cluster(cluster_name_on_cloud: str, reservation_id: str,
                     gpu_slug: str, gpus: int) -> None:
    state = load_state()
    state[str(cluster_name_on_cloud)] = {
        "reservation_id": str(reservation_id),
        "gpu_slug": str(gpu_slug),
        "gpus": int(gpus),
    }
    save_state(state)


def forget_cluster(cluster_name_on_cloud: str) -> None:
    state = load_state()
    state.pop(str(cluster_name_on_cloud), None)
    save_state(state)


def reservation_id_for(cluster_name_on_cloud: str) -> Optional[str]:
    """The recorded reservation id for this cluster, or None (unattributed
    reservations are invisible to us by design)."""
    row = load_state().get(str(cluster_name_on_cloud))
    if not row:
        return None
    return str(row.get("reservation_id"))
