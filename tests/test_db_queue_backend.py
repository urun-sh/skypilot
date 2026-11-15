"""Red-first tests for the DB-backed request queue (skc#911).

The multiprocessing queue backend routes every enqueue and dequeue through a
single BaseManager process over no-timeout synchronous RPCs. Reproduced at a
1000-poller burst: 1005 manager threads / 1013 fds, 31 requests executed vs
11,230 PENDING with idle executor workers — the dispatcher's get-RPC is
starved by the put storm, and any manager stall freezes the whole request
plane. The DB backend must serve the same contract with no IPC process and
stay FIFO and atomic under concurrent consumers.
"""

import threading

import pytest

from sky.server.requests.queues import base


@pytest.fixture()
def db_backend(monkeypatch, tmp_path):
    db_path = tmp_path / '.sky' / 'api_server' / 'requests.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # The constant is HOME-expanded at import time; redirect it per-test.
    monkeypatch.setattr(base.server_constants,
                        'API_SERVER_REQUEST_DB_PATH', str(db_path))
    yield base.DbQueueBackend('short')


def test_fifo_and_roundtrip(db_backend):
    db_backend.put(('a', False, False))
    db_backend.put(('b', True, True))
    assert db_backend.qsize() == 2
    assert db_backend.get() == ('a', False, False)
    assert db_backend.get() == ('b', True, True)
    assert db_backend.get() is None
    assert db_backend.qsize() == 0


def test_empty_get_returns_none_fast(db_backend):
    assert db_backend.get() is None


def test_retryable_requeue_goes_to_the_back(db_backend):
    db_backend.put(('a', False, False))
    db_backend.put(('b', False, False))
    assert db_backend.get()[0] == 'a'
    # requeue (retry) must NOT jump the queue
    db_backend.put(('a', False, False))
    assert db_backend.get()[0] == 'b'
    assert db_backend.get()[0] == 'a'


def test_concurrent_claim_is_atomic(db_backend):
    n = 200
    for i in range(n):
        db_backend.put((f'r{i}', False, False))
    claimed = []
    lock = threading.Lock()

    def consumer():
        while True:
            item = db_backend.get()
            if item is None:
                return
            with lock:
                claimed.append(item[0])

    threads = [threading.Thread(target=consumer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # every request claimed exactly once, none duplicated, none lost
    assert sorted(claimed) == sorted(f'r{i}' for i in range(n))
    assert db_backend.qsize() == 0


def test_isolated_by_schedule_type(db_backend):
    db_backend.put(('short-a', False, False))
    long_q = base.DbQueueBackend('long')
    long_q.put(('long-a', False, False))
    assert db_backend.qsize() == 1
    assert long_q.qsize() == 1
    assert db_backend.get()[0] == 'short-a'
    assert long_q.get()[0] == 'long-a'
    assert db_backend.get() is None


def test_factory_needs_no_background_process(db_backend):
    factory = base.DbQueueFactory()
    assert factory.start() is None
    q = factory.create_queue('short')
    q.put(('x', False, False))
    assert q.get() == ('x', False, False)
