import os
import shutil
import subprocess
import sys
import time
from unittest import mock

import pytest

from sky.backends import wheel_utils


def test_build_wheels():
    shutil.rmtree(wheel_utils.WHEEL_DIR, ignore_errors=True)
    start = time.time()
    wheel_path, _ = wheel_utils.build_sky_wheel()
    assert wheel_path.exists()
    duration = time.time() - start

    start = time.time()
    wheel_path, _ = wheel_utils.build_sky_wheel()
    assert wheel_path.exists()
    duration_cached = time.time() - start

    assert duration_cached < duration

    # simulate uncleaned symlinks due to interruption
    shutil.rmtree(wheel_utils.WHEEL_DIR, ignore_errors=True)
    wheel_utils.WHEEL_DIR.mkdir()
    (wheel_utils.WHEEL_DIR / 'sky').symlink_to(wheel_utils.SKY_PACKAGE_PATH,
                                               target_is_directory=True)
    for root, _, _ in os.walk(str(wheel_utils.WHEEL_DIR)):
        # set file date to 1970-01-01 00:00 UTC
        os.utime(root, (0, 0))
    wheel_path, _ = wheel_utils.build_sky_wheel()
    assert wheel_path.exists()

    shutil.rmtree(wheel_utils.WHEEL_DIR, ignore_errors=True)


def test_pip_missing_uv_environment():
    """Test error handling when pip module is not found in UV environment."""
    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[sys.executable, '-m', 'pip'],
            stderr='/path/to/python: No module named pip')

        with mock.patch('shutil.which') as mock_which:
            # Simulate UV is available
            mock_which.return_value = '/usr/bin/uv'

            with pytest.raises(RuntimeError) as exc_info:
                wheel_utils._build_sky_wheel()

            error_msg = str(exc_info.value)
            assert 'pip module not found' in error_msg
            assert 'Since you have UV installed' in error_msg
            assert 'uv pip install pip' in error_msg


def test_pip_missing_conda_environment():
    """Test error handling when pip module is not found in conda environment."""
    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[sys.executable, '-m', 'pip'],
            stderr='/path/to/python: No module named pip')

        with mock.patch('shutil.which') as mock_which:

            def which_side_effect(cmd):
                if cmd == 'uv':
                    return None
                elif cmd == 'conda':
                    return '/usr/bin/conda'
                return None

            mock_which.side_effect = which_side_effect

            with pytest.raises(RuntimeError) as exc_info:
                wheel_utils._build_sky_wheel()

            error_msg = str(exc_info.value)
            assert 'pip module not found' in error_msg
            assert 'Since you have conda installed' in error_msg
            assert 'conda install pip' in error_msg


def test_pip_missing_no_package_manager():
    """Test error handling when pip is missing with no known package manager."""
    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[sys.executable, '-m', 'pip'],
            stderr='/path/to/python: No module named pip')

        with mock.patch('shutil.which') as mock_which:
            mock_which.return_value = None

            with pytest.raises(RuntimeError) as exc_info:
                wheel_utils._build_sky_wheel()

            error_msg = str(exc_info.value)
            assert 'pip module not found' in error_msg
            assert 'Please install pip for your Python environment' in error_msg
            assert sys.executable in error_msg


def test_pip_command_other_failure():
    """Test error handling when pip command fails for non-module reasons."""
    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[sys.executable, '-m', 'pip', 'wheel'],
            stderr=
            'ERROR: Directory /tmp/something is not installable. Neither setup.py nor pyproject.toml found.'
        )

        with pytest.raises(RuntimeError) as exc_info:
            wheel_utils._build_sky_wheel()

        error_msg = str(exc_info.value)
        assert 'pip wheel command failed' in error_msg
        assert 'Directory /tmp/something is not installable' in error_msg


def test_python_executable_not_found():
    """Test error handling when Python executable is not found (rare case)."""
    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = FileNotFoundError(
            "[Errno 2] No such file or directory: '/nonexistent/python'")

        with pytest.raises(RuntimeError) as exc_info:
            wheel_utils._build_sky_wheel()

        error_msg = str(exc_info.value)
        assert 'Python executable not found' in error_msg
        assert sys.executable in error_msg


def test_python_m_pip_usage():
    """Test that we use 'python -m pip' instead of 'pip3'."""
    # This is a simpler test that just verifies the command format
    # The actual wheel building is tested in test_wheels.py

    # Create a minimal mock that captures the subprocess.run call
    with mock.patch('subprocess.run') as mock_run:
        # Make subprocess.run fail with a specific error so we can catch it
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[sys.executable, '-m', 'pip'],
            stderr='Intentional test failure')

        try:
            wheel_utils._build_sky_wheel()
        except RuntimeError:
            # We expect this to fail, we just want to check the command
            pass

        # Verify that subprocess.run was called with python -m pip
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]

        # Check the command format
        assert call_args[0] == sys.executable
        assert call_args[1:3] == ['-m', 'pip']
        assert 'wheel' in call_args
        assert '--no-deps' in call_args

        # Verify we're NOT using pip3
        assert 'pip3' not in call_args


def test_wheel_build_reproducible():
    """Test that wheel builds are reproducible across separate processes.

    PYTHONHASHSEED is randomized per-process, so without the fix (setting
    SOURCE_DATE_EPOCH and PYTHONHASHSEED in the pip subprocess env), two
    separate processes building the same source can produce different wheel
    hashes due to non-deterministic metadata ordering and zip timestamps.

    We simulate this by running each build in a separate subprocess,
    just like different API server replicas would. Each subprocess gets
    a naturally randomized PYTHONHASHSEED.
    """
    build_script = (
        'import shutil, os; '
        'from sky.backends import wheel_utils; '
        'shutil.rmtree(str(wheel_utils.WHEEL_DIR), ignore_errors=True); '
        '_, h = wheel_utils.build_sky_wheel(); '
        'print(h)')

    try:
        result1 = subprocess.run([sys.executable, '-c', build_script],
                                 capture_output=True,
                                 text=True,
                                 check=True)
        hash1 = result1.stdout.strip().splitlines()[-1]

        result2 = subprocess.run([sys.executable, '-c', build_script],
                                 capture_output=True,
                                 text=True,
                                 check=True)
        hash2 = result2.stdout.strip().splitlines()[-1]

        assert hash1 == hash2, (
            f'Wheel build is not reproducible across processes: '
            f'{hash1} != {hash2}. '
            'Check that SOURCE_DATE_EPOCH and PYTHONHASHSEED are set in the '
            'pip wheel subprocess environment in _build_sky_wheel().')
    finally:
        shutil.rmtree(wheel_utils.WHEEL_DIR, ignore_errors=True)


def _counting_filelock_factory(counter):
    import threading
    """A filelock.FileLock stand-in that counts acquisitions of the wheel
    lock (path-keyed, so unrelated locks in the same module stay free)."""

    class _CountingLock:
        # Real mutual exclusion, not just counting: the fake must preserve
        # the lock's blocking semantics or the double-checked slow path
        # runs concurrently and rebuilds N times for a reason the REAL
        # lock would have prevented.
        _gate = threading.Lock()

        def __init__(self, lock_path):
            self._path = str(lock_path)

        def __enter__(self):
            _CountingLock._gate.acquire()
            if self._path == str(wheel_utils._WHEEL_LOCK_PATH):
                counter['enter'] += 1
            return self

        def __exit__(self, *args):
            _CountingLock._gate.release()
            return False

        def acquire(self, *args, **kwargs):
            return self.__enter__() or True

        def release(self):
            self.__exit__()

    class _FileLockModule:
        FileLock = _CountingLock
        Timeout = getattr(wheel_utils.filelock, 'Timeout', Exception)

    return _FileLockModule


def _hermetic_wheel_state(tmp_path, stale=False):
    """Fake sky package + WHEEL_DIR with one wheel; returns the wheel path.

    WHEEL_DIR gets a current wheel (mtime AFTER the sky package) unless
    `stale`, in which case the package file's mtime is pushed past the
    wheel's, making the freshness rule demand a rebuild.
    """
    import uuid

    sky_pkg = tmp_path / 'sky_pkg'
    (sky_pkg / 'sub').mkdir(parents=True)
    pkg_file = sky_pkg / 'sub' / 'mod.py'
    pkg_file.write_text('x = 1\n')
    from sky import sky_logging  # noqa: F401 - sky import cost is shared
    wheel_body = f'fake-wheel-{uuid.uuid4().hex}'.encode()
    wheel_dir = tmp_path / 'wheels' / uuid.uuid4().hex
    wheel_dir.mkdir(parents=True)
    # The wheel name MUST match the module's own _WHEEL_PATTERN
    # (skypilot-<sky.__version__>-*) or the freshness glob finds nothing and
    # the wheel is judged stale.
    wheel = wheel_dir / f'skypilot-{wheel_utils.sky.__version__}-py2.py3-none-any.whl'
    wheel.write_bytes(wheel_body)
    if stale:
        # STALE = the source is NEWER than the wheel, but BOTH in the past:
        # a freshly rebuilt wheel (mtime=now) must then read as current to
        # the double-check, or the rebuild races itself.
        base = time.time() - 50
        os.utime(wheel, (base, base))
        os.utime(wheel_dir, (base, base))
        newer = base + 10
        os.utime(pkg_file, (newer, newer))
        os.utime(sky_pkg / 'sub', (newer, newer))
        os.utime(sky_pkg, (newer, newer))
    return wheel, wheel_body


def test_concurrent_current_wheel_locks_at_most_once(tmp_path, monkeypatch):
    """N threads racing build_sky_wheel on a CURRENT wheel must not
    serialize on the global wheel lock: the freshness walk and the temp-copy
    check run lock-free, and the copy itself is atomic. Live receipt
    (r1000m on lax, 2026-11-16): ~920 executor processes all blocked at
    FileLock.__enter__ inside build_sky_wheel, launching at ~10/min."""
    import tempfile
    import threading

    wheel, _ = _hermetic_wheel_state(tmp_path)
    monkeypatch.setattr(wheel_utils, 'SKY_PACKAGE_PATH', tmp_path / 'sky_pkg')
    monkeypatch.setattr(wheel_utils, 'WHEEL_DIR', tmp_path / 'wheels')
    counter = {'enter': 0}
    monkeypatch.setattr(wheel_utils, 'filelock',
                        _counting_filelock_factory(counter))
    monkeypatch.setattr(wheel_utils, '_SKY_WHEEL_MEMO', None, raising=False)

    temp_dir = wheel.parent.name  # /tmp/<hash> resolution below
    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results = []
    errors = []

    def runner():
        try:
            barrier.wait(timeout=30)
            results.append(wheel_utils.build_sky_wheel())
        except Exception as e:  # noqa: BLE001 - the assertion is the check
            errors.append(e)

    threads = [threading.Thread(target=runner) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, errors
    import pathlib as _pl
    temp_root = _pl.Path(tempfile.gettempdir()) / wheel.parent.name
    # every caller got the same served copy, and it is whole
    for path, wheel_hash in results:
        assert wheel_hash == wheel.parent.name
        served = path / wheel.name
        assert served.exists()
    # cleanup the shared /tmp dir this test seeded
    shutil.rmtree(temp_root, ignore_errors=True)
    assert counter['enter'] <= 1, (
        f'the global wheel lock was acquired {counter["enter"]} times for a '
        'CURRENT wheel across 16 concurrent callers — build_sky_wheel is '
        'serializing launches again')


def test_concurrent_stale_wheel_rebuilds_exactly_once(tmp_path, monkeypatch):
    """A stale wheel still rebuilds EXACTLY once under concurrent callers
    (double-checked locking), and every caller ends up on the new wheel."""
    import tempfile
    import threading

    wheel, _ = _hermetic_wheel_state(tmp_path, stale=True)
    monkeypatch.setattr(wheel_utils, 'SKY_PACKAGE_PATH', tmp_path / 'sky_pkg')
    monkeypatch.setattr(wheel_utils, 'WHEEL_DIR', tmp_path / 'wheels')
    counter = {'enter': 0, 'build': 0}
    monkeypatch.setattr(wheel_utils, 'filelock',
                        _counting_filelock_factory(counter))

    new_wheel_dir = tmp_path / 'wheels' / 'rebuilthash00'

    def fake_build():
        counter['build'] += 1
        new_wheel_dir.mkdir(parents=True, exist_ok=True)
        new_wheel = new_wheel_dir / f'skypilot-{wheel_utils.sky.__version__}-py2.py3-none-any.whl'
        # mtime = now: newer than the stale package (base+10), so the
        # double-check inside the lock reads the wheel as current.
        new_wheel.write_bytes(b'rebuilt-wheel')
        return new_wheel

    monkeypatch.setattr(wheel_utils, '_build_sky_wheel', fake_build)
    monkeypatch.setattr(wheel_utils, '_SKY_WHEEL_MEMO', None, raising=False)

    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results = []
    errors = []

    def runner():
        try:
            barrier.wait(timeout=30)
            results.append(wheel_utils.build_sky_wheel())
        except Exception as e:  # noqa: BLE001 - the assertion is the check
            errors.append(e)

    threads = [threading.Thread(target=runner) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, errors
    import pathlib as _pl
    temp_root = _pl.Path(tempfile.gettempdir()) / 'rebuilthash00'
    assert counter['build'] == 1, (
        f'the wheel was rebuilt {counter["build"]} times under 16 racing '
        'callers — double-checked locking must rebuild exactly once')
    hashes = {h for _, h in results}
    assert hashes == {'rebuilthash00'}, hashes
    for path, _ in results:
        assert (path /
                f'skypilot-{wheel_utils.sky.__version__}-py2.py3-none-any.whl'
               ).exists()
    shutil.rmtree(temp_root, ignore_errors=True)
