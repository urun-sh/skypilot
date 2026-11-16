import json
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
    """A filelock.FileLock stand-in that counts acquisitions of the wheel
    lock (path-keyed, so unrelated locks in the same module stay free)."""
    import threading

    class _CountingLock:
        # Real mutual exclusion, not just counting: the fake must preserve
        # the lock's blocking semantics or a double-checked slow path would
        # run concurrently for a reason the REAL lock would have prevented.
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


def test_prebuilt_pointer_serves_with_no_lock_and_no_walk(
        tmp_path, monkeypatch):
    """With the image-build pointer present, build_sky_wheel serves it
    immediately: N concurrent launches acquire the wheel lock 0 times and
    NEVER walk the sky package (r1000m ruling: the wheel is stable for the
    server's whole life and already lives in the R2 runtime mount)."""
    import threading

    wheel_dir = tmp_path / 'sky-wheel'
    wheel_dir.mkdir(parents=True)
    wheel = wheel_dir / 'skypilot-0.13.0-py3-none-any.whl'
    wheel.write_bytes(b'prebuilt-wheel-bytes')
    pointer = tmp_path / '_prebuilt_wheel.json'
    pointer.write_text(
        json.dumps({
            'wheel_path': str(wheel),
            'wheel_hash': 'a' * 32,
        }))
    monkeypatch.setattr(wheel_utils, '_PREBUILT_WHEEL_POINTER_PATH', pointer)
    monkeypatch.setattr(wheel_utils, '_PREBUILT_WHEEL_VERIFIED', False)
    counter = {'enter': 0}
    monkeypatch.setattr(wheel_utils, 'filelock',
                        _counting_filelock_factory(counter))

    # The nested freshness walk is uncachable on this branch; detect ANY
    # walk by counting os.walk calls (os.walk is the only filesystem scan
    # the non-pointer path performs — any count means the pointer fast path
    # was skipped). A counter, not a raiser: os.walk is process-global and
    # a raiser breaks unrelated plugin threads mid-collection.
    walks = {'count': 0}
    _real_walk = os.walk

    def _counting_walk(*args, **kwargs):
        walks['count'] += 1
        return _real_walk(*args, **kwargs)

    monkeypatch.setattr(wheel_utils.os, 'walk', _counting_walk)

    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results = []
    errors = []

    def runner():
        try:
            barrier.wait(timeout=30)
            results.append(wheel_utils.build_sky_wheel())
        except Exception as e:  # noqa: BLE001 - the assertions are the check
            errors.append(e)

    threads = [threading.Thread(target=runner) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert results == [(wheel, 'a' * 32)] * n_threads
    assert counter['enter'] == 0, (
        'the pointer fast path must not acquire the wheel lock at all')
    assert walks['count'] == 0, (
        'the pointer fast path must never walk the sky package')


def test_prebuilt_pointer_missing_wheel_fails_loud(tmp_path, monkeypatch):
    """A built image whose pointed-at wheel was stripped must fail LOUDLY on
    the first call, never silently fall back to a per-launch build."""
    wheel_dir = tmp_path / 'sky-wheel'
    wheel_dir.mkdir(parents=True)
    pointer = tmp_path / '_prebuilt_wheel.json'
    pointer.write_text(
        json.dumps({
            'wheel_path': str(wheel_dir / 'gone.whl'),
            'wheel_hash': 'b' * 32,
        }))
    monkeypatch.setattr(wheel_utils, '_PREBUILT_WHEEL_POINTER_PATH', pointer)
    monkeypatch.setattr(wheel_utils, '_PREBUILT_WHEEL_VERIFIED', False)

    with pytest.raises(RuntimeError, match='prebuilt-wheel pointer'):
        wheel_utils.build_sky_wheel()
