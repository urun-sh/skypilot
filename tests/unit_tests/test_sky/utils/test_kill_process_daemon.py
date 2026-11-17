"""Tests for sky.utils.subprocess_utils.kill_process_daemon."""

import os
import subprocess
import sys
import time

import psutil
import pytest

from sky.utils import subprocess_utils


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='zombie reaping is a POSIX /proc behavior')
def test_kill_process_daemon_reaps_launcher() -> None:
    """The daemon launcher must be reaped, not left a zombie.

    kill_process_daemon spawns `python subprocess_daemon.py`, which
    double-forks (daemonize) — the direct child exits almost immediately
    while the real daemon grandchild detaches. A dropped Popen handle never
    reaps that exit, leaving one zombie per call under the caller (the sky
    API server's executor workers accumulated hundreds under load).
    """
    child = subprocess.Popen(['sleep', '30'])
    try:
        # The launcher is the only NEW direct child besides the sleep probe;
        # a zombie's /proc cmdline is empty, so identify it by pid set diff.
        me = psutil.Process()
        before = {c.pid for c in me.children()}
        subprocess_utils.kill_process_daemon(child.pid, use_kill_pg=True)
        launcher = None
        for _ in range(100):
            new = {c.pid for c in me.children()} - before - {child.pid}
            if new:
                launcher = new.pop()
                break
            time.sleep(0.05)
        assert launcher is not None, 'daemon launcher was never spawned'

        # The launcher exits immediately after double-forking. Reaped: its
        # /proc entry disappears. Unreaped: it lingers in state Z forever.
        deadline = time.time() + 10
        reaped = False
        while time.time() < deadline:
            try:
                with open(f'/proc/{launcher}/stat') as f:
                    _ = f.read().rsplit(')', 1)[1].split()[0]
                time.sleep(0.05)
            except (FileNotFoundError, ProcessLookupError, IndexError):
                reaped = True
                break
        assert reaped, f'launcher pid {launcher} left as a zombie'
    finally:
        child.kill()
        child.wait()
        for proc in psutil.process_iter(['cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
                if ('subprocess_daemon.py' in cmd
                        and f'--parent-pid {os.getpid()}' in cmd):
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue
