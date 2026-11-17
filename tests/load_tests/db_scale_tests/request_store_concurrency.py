#!/usr/bin/env python3
"""Concurrency proof for the API-server request store backends.

Reproduces the prod twins shape (2026-11-17: 1000 concurrent launches against
a SQLite request store -> 212 "database is locked" errors in hours 15-16,
launches stalling >60 s). The server hits the store two ways, so the harness
runs both:

  * ASYNC STORM — `--workers` coroutines insert a request and read it back
    through the server's async path (the API server's event loop).
  * SYNC STORM — `--sync-writers` threads do the same through the sync path
    (the request executor workers). Sync SQLiteConn is per-thread, so N
    threads = N sqlite connections contending on one file lock: this is where
    "database is locked" is raised (after the 60 s busy timeout).

Backend selection is the server's own: IS_SKYPILOT_SERVER + the fork's
PostgresConn when SKYPILOT_DB_CONNECTION_URI is set, SQLite otherwise.

Usage:
  # SQLite (the prod baseline):
  python3 request_store_concurrency.py --backend sqlite --workers 1000

  # Postgres (the fork's backend):
  SKYPILOT_DB_CONNECTION_URI=postgresql://user:pass@host:5432/sky \
      python3 request_store_concurrency.py --backend postgres --workers 1000
"""

import argparse
import asyncio
import os
import statistics
import sys
import threading
import time
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=('sqlite', 'postgres'),
                    default='sqlite')
parser.add_argument('--workers', type=int, default=1000,
                    help='async-storm coroutines')
parser.add_argument('--sync-writers', type=int, default=1000,
                    help='sync-storm threads')
parser.add_argument('--ops-per-worker', type=int, default=1,
                    help='insert+read cycles per worker')
parser.add_argument('--reads-per-cycle', type=int, default=2,
                    help='status reads after each insert')
parser.add_argument('--op-timeout', type=float, default=30.0,
                    help='per-op timeout in seconds (a timed-out async op '
                         'counts as an error instead of hanging the run)')
parser.add_argument('--phase-timeout', type=float, default=240.0,
                    help='wall-clock cap per phase; unfinished ops are '
                         'reported as stalled')
args = parser.parse_args()

# Backend selection happens inside sky at import time from env; set both
# BEFORE the first sky import. HOME isolation keeps this run off the
# workstation's real ~/.sky (config + state files); server mode rejects a
# config file alongside a db URI, so the isolated HOME must have none.
os.environ['IS_SKYPILOT_SERVER'] = '1'
if args.backend == 'postgres':
    uri = os.environ.get('SKYPILOT_DB_CONNECTION_URI')
    assert uri and uri.startswith('postgres'), \
        'set SKYPILOT_DB_CONNECTION_URI for --backend postgres'
else:
    os.environ['SKYPILOT_DB_CONNECTION_URI'] = ''
home = os.environ.get('LOAD_TEST_HOME')
if home:
    os.environ['HOME'] = home

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# pylint: disable=wrong-import-position
import uuid  # noqa: E402

from sky.server.requests import payloads as request_payloads  # noqa: E402
from sky.server.requests import requests as requests_mod  # noqa: E402
from sky.server.requests import storage as request_storage  # noqa: E402


def _noop():
    """Picklable stand-in entrypoint (to_row() cloudpickles it)."""


def _make_request(rid: str):
    return requests_mod.Request(
        request_id=rid,
        name='load-test',
        entrypoint=_noop,
        request_body=request_payloads.RequestBody(),
        status=requests_mod.RequestStatus.PENDING,
        created_at=time.time(),
        user_id='loadtest',
    )


def _pct(samples, p):
    if not samples:
        return float('nan')
    return sorted(samples)[int(len(samples) * p / 100)]


def _summarize(name, ins, rds, errors, total_ops, total_reads, wall):
    print(f'  {name}: wall={wall:.2f}s  inserts={len(ins)}/{total_ops} '
          f'reads={len(rds)}/{total_reads}')
    for op, samples in (('insert', ins), ('read', rds)):
        if samples:
            print(f'    {op}: p50={_pct(samples, 50):.0f}us '
                  f'p99={_pct(samples, 99):.0f}us '
                  f'max={max(samples):.0f}us '
                  f'throughput={len(samples)/wall:.0f}/s')
    if errors:
        print('    ERRORS:')
        for err, n in errors.most_common():
            print(f'      {n:5d}  {err}')
    else:
        print('    ERRORS: none')


async def _async_phase():
    """N coroutines: insert a request, read it back."""
    results = {'insert_us': [], 'read_us': [], 'errors': Counter(),
               'stalled': 0}
    requests_mod._ensure_db_initialized()

    print(f'ASYNC STORM: workers={args.workers} '
          f'ops_per_worker={args.ops_per_worker} '
          f'reads_per_cycle={args.reads_per_cycle} '
          f'phase_timeout={args.phase_timeout}s')
    barrier = asyncio.Event()
    tasks = [asyncio.create_task(_wait_barrier(barrier, wid, results))
             for wid in range(args.workers)]
    t0 = time.perf_counter()
    barrier.set()
    done, pending = await asyncio.wait(tasks, timeout=args.phase_timeout)
    wall = time.perf_counter() - t0
    results['stalled'] = len(pending)
    for task in pending:
        task.cancel()
    if pending:
        results['errors']['asyncio.stalled(op did not finish within the '
                          'phase timeout)'] += 1
    _summarize('async', results['insert_us'], results['read_us'],
               results['errors'], args.workers * args.ops_per_worker,
               args.workers * args.ops_per_worker * args.reads_per_cycle,
               wall)
    if results['stalled']:
        print(f'    stalled coroutines: {results["stalled"]}')


async def _wait_barrier(barrier, wid, results):
    await barrier.wait()
    # Inline the worker body (a separate coroutine would just proxy).
    for i in range(args.ops_per_worker):
        rid = f'lt-{wid:04d}-{i:03d}-{uuid.uuid4().hex[:8]}'
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(
                requests_mod._add_or_update_request_no_lock_async(
                    _make_request(rid)),
                timeout=args.op_timeout)
            results['insert_us'].append((time.perf_counter() - t0) * 1e6)
        except Exception as e:  # pylint: disable=broad-except
            results['errors'][type(e).__name__ + ': ' + str(e)[:120]] += 1
            continue
        for _ in range(args.reads_per_cycle):
            t0 = time.perf_counter()
            try:
                st = await asyncio.wait_for(
                    requests_mod.get_request_status_async(rid),
                    timeout=args.op_timeout)
                assert st is not None, f'{rid} not found after insert'
                results['read_us'].append((time.perf_counter() - t0) * 1e6)
            except Exception as e:  # pylint: disable=broad-except
                results['errors'][type(e).__name__ + ': ' +
                                 str(e)[:120]] += 1


def _sync_phase():
    """N threads: insert a request, read it back (executor-worker shape)."""
    results = {'insert_us': [], 'read_us': [], 'errors': Counter()}
    requests_mod._ensure_db_initialized()
    lock = threading.Lock()

    def writer(wid: int):
        for i in range(args.ops_per_worker):
            rid = f'lts-{wid:04d}-{i:03d}-{uuid.uuid4().hex[:8]}'
            t0 = time.perf_counter()
            try:
                requests_mod._add_or_update_request_no_lock(
                    _make_request(rid))
                dt = (time.perf_counter() - t0) * 1e6
                with lock:
                    results['insert_us'].append(dt)
            except Exception as e:  # pylint: disable=broad-except
                with lock:
                    results['errors'][type(e).__name__ + ': ' +
                                     str(e)[:120]] += 1
                continue
            for _ in range(args.reads_per_cycle):
                t0 = time.perf_counter()
                try:
                    st = request_storage.get_request_backend().get_request(
                        rid)
                    assert st is not None, f'{rid} not found after insert'
                    dt = (time.perf_counter() - t0) * 1e6
                    with lock:
                        results['read_us'].append(dt)
                except Exception as e:  # pylint: disable=broad-except
                    with lock:
                        results['errors'][type(e).__name__ + ': ' +
                                         str(e)[:120]] += 1

    print(f'SYNC STORM: writers={args.sync_writers} '
          f'ops_per_worker={args.ops_per_worker} '
          f'reads_per_cycle={args.reads_per_cycle}')
    threads = [threading.Thread(target=writer, args=(w,))
               for w in range(args.sync_writers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=args.phase_timeout)
    wall = time.perf_counter() - t0
    alive = sum(1 for t in threads if t.is_alive())
    if alive:
        results['errors'][f'threads still running after phase timeout '
                          f'({alive})'] += 1
    _summarize('sync', results['insert_us'], results['read_us'],
               results['errors'],
               args.sync_writers * args.ops_per_worker,
               args.sync_writers * args.ops_per_worker * args.reads_per_cycle,
               wall)


async def main():
    # pylint: disable=protected-access
    requests_mod._ensure_db_initialized()
    print(f'backend={requests_mod._DB.__class__.__name__}')
    await _async_phase()
    _sync_phase()


if __name__ == '__main__':
    asyncio.run(main())
