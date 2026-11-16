"""Regression tests for TaskCodeGen.add_epilogue all-137 handling.

Live incident (2026-11-16): a single-node probe job's bootstrap process was
SIGKILLed (exit 137) on the only worker, so returncodes == [137]. The
generated epilogue did `next(r for r in returncodes if r != 137)` which
raised StopIteration, crashing the job wrapper and turning an infra kill
into an opaque FAILED job.
"""
import textwrap
import types
from unittest import mock

from sky.backends import task_codegen


def _generate_epilogue_block() -> str:
    """Generates the epilogue code block from RayCodeGen."""
    codegen = task_codegen.RayCodeGen()
    codegen.add_prologue(job_id=1)
    codegen.add_epilogue()
    # add_epilogue appends exactly one chunk to self._code.
    return codegen._code[-1]


def _run_epilogue(returncodes):
    """Executes the generated epilogue block with mocked runtime deps.

    Returns captured print args and the exit code passed to sys.exit.
    """
    code = textwrap.dedent(_generate_epilogue_block())
    compile(code, '<generated-epilogue>', 'exec')
    captured = {'exit_code': None, 'printed': []}

    class FakeScheduler:
        def schedule_step(self):
            pass

    class FakeJobLib:
        scheduler = FakeScheduler()

        class JobStatus:
            FAILED = 'FAILED'
            SUCCEEDED = 'SUCCEEDED'

        def set_exit_codes(self, job_id, codes):
            pass

        def set_status(self, job_id, status):
            pass

    class FakeSys:
        def exit(self, code):
            captured['exit_code'] = code
            raise SystemExit(code)

    def fake_print(*args, **kwargs):
        captured['printed'].append(args)

    ns = {
        'returncodes': returncodes,
        'job_lib': FakeJobLib(),
        'time': mock.Mock(),
        'sys': FakeSys(),
        'constants': types.SimpleNamespace(SKYLET_VERSION=28),
        'print': fake_print,
    }
    try:
        exec(code, ns)  # pylint: disable=exec-used
    except SystemExit as e:
        # sys.exit(1) is the generated epilogue's expected failure path.
        assert e.code == 1
    return captured


ALL_137_REASON = ('(all workers exited with code 137 — SIGKILL, '
                  'likely OOM or external cleanup)')


def test_epilogue_all_137_no_crash():
    """returncodes == [137, 137] must not crash the epilogue."""
    captured = _run_epilogue([137, 137])
    assert captured['exit_code'] == 1
    reason = captured['printed'][0][2]
    assert reason == ALL_137_REASON


def test_epilogue_single_137_no_crash():
    """returncodes == [137] (single node) must not crash the epilogue."""
    captured = _run_epilogue([137])
    assert captured['exit_code'] == 1
    reason = captured['printed'][0][2]
    assert reason == ALL_137_REASON


def test_epilogue_mixed_137_keeps_existing_message():
    """Mixed returncodes keeps the existing non-137 message."""
    captured = _run_epilogue([137, 1])
    assert captured['exit_code'] == 1
    reason = captured['printed'][0][2]
    assert reason == ('(A Worker failed with return code 1, SkyPilot '
                      'cleaned up the processes on other nodes with '
                      'return code 137)')
