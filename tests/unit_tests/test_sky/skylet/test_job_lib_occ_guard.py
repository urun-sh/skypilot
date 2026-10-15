"""The urun occupant mirror rows must survive skylet's status reconciler.

The node-side `urun_occ_to_jobs` sync writes one `jobs` row per live urun
occupant (run_timestamp `occ:<occupant_id>`, metadata `{"urun_occ": true}`)
into the node's jobs.db. Those rows have no ray driver pid BY DESIGN — their
status is owned by the occupancy heartbeat — but `job_lib.update_status()`
(the skylet JobSchedulerEvent reconciler, every 300s) demotes every
non-terminal row without a live driver pid to FAILED_DRIVER. Without the
guard below, every occupant row in the dashboard JOBS panel flips to
FAILED_DRIVER within one scheduler tick (observed live on prod-usw2 node
fb9e993316: RUNNING -> FAILED_DRIVER after one update_status()).

Red-first: these tests fail on the unguarded update_status().
"""

import json
import os

import pytest

from sky.skylet import job_lib


@pytest.fixture
def isolated_jobs_db(tmp_path, monkeypatch):
    """Point $SKY_RUNTIME_DIR/.sky/jobs.db at a fresh tmp dir. Must be set
    before job_lib initializes its cached _DB connection."""
    runtime_dir = tmp_path / "sky-runtime"
    sky_dir = runtime_dir / ".sky"
    sky_dir.mkdir(parents=True)
    monkeypatch.setenv("SKY_RUNTIME_DIR", str(runtime_dir))
    # _DB is cached per-process; reset it so the fixture's dir is the one
    # actually opened (each test gets a fresh module state).
    monkeypatch.setattr(job_lib, "_DB", None)
    yield sky_dir


def _add_occ_row(run_timestamp: str, status_setter) -> int:
    job_id, _ = job_lib.add_job(
        "test-occupant",
        "urunocc",
        run_timestamp,
        "{}",
        metadata=json.dumps({
            "urun_occ": True,
            "occupant_id": run_timestamp
        }),
    )
    if status_setter is not None:
        status_setter(job_id)
    return job_id


def _set_pending(job_id: int) -> None:
    job_lib.set_status(job_id, job_lib.JobStatus.PENDING)


def test_update_status_keeps_running_occ_row(isolated_jobs_db):
    job_id = _add_occ_row("occ:test-running-1", job_lib.set_job_started)
    assert job_lib.get_status(job_id) == job_lib.JobStatus.RUNNING

    job_lib.update_status()

    assert job_lib.get_status(job_id) == job_lib.JobStatus.RUNNING


def test_update_status_keeps_pending_occ_row(isolated_jobs_db):
    job_id = _add_occ_row("occ:test-pending-1", _set_pending)
    assert job_lib.get_status(job_id) == job_lib.JobStatus.PENDING

    job_lib.update_status()

    assert job_lib.get_status(job_id) == job_lib.JobStatus.PENDING


def test_update_status_still_demotes_stale_real_jobs(isolated_jobs_db,
                                                     monkeypatch):
    """The guard must be narrow: a plain INIT row (no urun_occ) that is stale
    still gets demoted. Staleness is forced through job_lib's own INIT grace
    window (a negative grace makes a fresh INIT row already past it) — no
    patching of the lazily-imported psutil proxy, whose attribute writes are
    import-order sensitive under the full suite."""
    job_id, _ = job_lib.add_job("real-job", "realhash", "rt-not-occ", "{}")
    assert job_lib.get_status(job_id) == job_lib.JobStatus.INIT
    monkeypatch.setattr(job_lib, "_INIT_SUBMIT_GRACE_PERIOD", -1)

    job_lib.update_status()

    assert job_lib.get_status(job_id) == job_lib.JobStatus.FAILED_DRIVER


def test_update_status_leaves_terminal_occ_rows_alone(isolated_jobs_db):
    job_id = _add_occ_row("occ:test-cancelled-1", _set_pending)
    job_lib.set_status(job_id, job_lib.JobStatus.CANCELLED)

    job_lib.update_status()

    assert job_lib.get_status(job_id) == job_lib.JobStatus.CANCELLED
