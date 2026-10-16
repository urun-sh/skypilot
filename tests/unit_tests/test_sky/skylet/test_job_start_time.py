"""An occupant mirror row's real start time must survive to RUNNING.

A node's urun occupant is ADMITTED (its true start) potentially minutes
before the sync daemon writes its jobs.db row, and the row sits PENDING for
the whole boot. `set_job_started` stamped wall-clock NOW at the RUNNING
transition, so a booting occupant showed duration 0s in the dashboard the
entire boot (live receipt, prod-usw2 2026-10-16: qwen_voice_sense rows all
`0s` while the occupant booted 14+ min), and the admission time was lost
again at the ready transition.

Red-first: these tests fail without set_job_start_time / the COALESCE in
set_job_started.
"""

import json

import pytest

from sky.skylet import job_lib


@pytest.fixture
def isolated_jobs_db(tmp_path, monkeypatch):
    """Fresh jobs.db per test (same shape as test_job_lib_occ_guard)."""
    runtime_dir = tmp_path / "sky-runtime"
    sky_dir = runtime_dir / ".sky"
    sky_dir.mkdir(parents=True)
    monkeypatch.setenv("SKY_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setattr(job_lib, "_DB", None)
    monkeypatch.setattr(job_lib.constants,
                        "PERSISTENT_RUN_RESTARTING_SIGNAL_FILE",
                        str(tmp_path / "no-restarting-signal"))
    yield sky_dir


def _add_occ_row(run_timestamp: str) -> int:
    job_id, _ = job_lib.add_job(
        "test-occupant",
        "urunocc",
        run_timestamp,
        "{}",
        metadata=json.dumps({"urun_occ": True}),
    )
    return job_id


def test_start_time_survives_running_transition(isolated_jobs_db):
    """set_job_started must not overwrite a real, earlier start_at."""
    admission_at = 1_000_000.0
    job_id = _add_occ_row("occ:test-start-1")
    job_lib.set_job_start_time(job_id, admission_at)
    job_lib.set_status(job_id, job_lib.JobStatus.PENDING)
    assert job_lib.get_status(job_id) == job_lib.JobStatus.PENDING

    job_lib.set_job_started(job_id)

    info = job_lib.get_jobs_info("urunocc", all_jobs=True)[0]
    assert info.start_at == admission_at
    assert job_lib.get_status(job_id) == job_lib.JobStatus.RUNNING


def test_pending_row_carries_real_elapsed(isolated_jobs_db):
    """A booting (PENDING) row's start_at is the admission time, not NULL —
    the dashboard computes duration = now - start_at for live rows."""
    admission_at = 1_000_000.0
    job_id = _add_occ_row("occ:test-start-2")
    job_lib.set_job_start_time(job_id, admission_at)
    job_lib.set_status(job_id, job_lib.JobStatus.PENDING)

    info = job_lib.get_jobs_info("urunocc", all_jobs=True)[0]

    assert job_lib.get_status(job_id) == job_lib.JobStatus.PENDING
    assert info.start_at == admission_at


def test_set_job_started_without_prestamp_uses_now(isolated_jobs_db):
    """A plain job (start_at still the -1 default) keeps today's behavior:
    RUNNING stamps wall-clock now."""
    job_id, _ = job_lib.add_job("plain-job", "u1", "rt-plain", "{}")

    job_lib.set_job_started(job_id)

    info = job_lib.get_jobs_info("u1", all_jobs=True)[0]
    assert info.start_at is not None
    assert info.start_at > 0
    assert info.start_at != -1
