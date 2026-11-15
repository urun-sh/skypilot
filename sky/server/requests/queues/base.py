"""Abstract interfaces for request queue backends."""

import abc
import multiprocessing
import os
import queue as queue_lib
import time
from typing import List, Optional, Tuple

from sky import sky_logging
from sky.metrics import utils as metrics_utils
from sky.server import constants as server_constants
from sky.server.requests import requests as api_requests
from sky.server.requests.queues import local_queue
from sky.server.requests.queues import mp_queue
from sky.utils import common_utils

logger = sky_logging.init_logger(__name__)

# The element stored inside the in-memory queues: the public request tuple
# plus the wall-clock time it was enqueued, so get() can report how long the
# entry actually waited for a worker. The enqueue timestamp is set
# afresh on every put(), including re-enqueues (requeue, retry/backoff,
# broken-pool), so the recorded wait excludes time spent paused or running.
_InternalItem = Tuple[Tuple[str, bool, bool], float]


def _observe_queue_wait(schedule_type: str, enqueued_at: float) -> None:
    """Record time the just-dequeued entry spent waiting in the queue.

    Metric emission must never disrupt the dequeue path, so any failure
    (label mismatch, client error) is logged and swallowed.
    """
    if not metrics_utils.METRICS_ENABLED:
        return
    try:
        metrics_utils.SKY_APISERVER_QUEUE_WAIT_SECONDS.labels(
            schedule_type=schedule_type).observe(
                max(0,
                    time.time() - enqueued_at))
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f'Failed to observe queue wait metric: {e}')


class QueueBackend(abc.ABC):
    """Abstract queue backend.

    Each backend is responsible for emitting
    ``SKY_APISERVER_QUEUE_WAIT_SECONDS`` from ``get()`` (measuring the time
    between enqueue and dequeue of the returned entry), so the metric reflects
    true queue residency regardless of the backend's storage.
    """

    @abc.abstractmethod
    def put(self, item: Tuple[str, bool, bool]) -> None:
        """Put a (request_id, ignore_return_value, retryable) tuple."""
        raise NotImplementedError

    async def put_async(self, item: Tuple[str, bool, bool]) -> None:
        """Async version of put."""
        # By default we assume put is not blocking and can be
        # called directly in event loop
        self.put(item)

    @abc.abstractmethod
    def get(self) -> Optional[Tuple[str, bool, bool]]:
        """Non-blocking get. Returns None if queue is empty.

        Implementations must emit ``SKY_APISERVER_QUEUE_WAIT_SECONDS`` for the
        dequeued entry (see ``_observe_queue_wait``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def qsize(self) -> int:
        """Return approximate queue size."""
        raise NotImplementedError


class QueueBackendFactory(abc.ABC):
    """Creates queue instances and manages queue infrastructure."""

    @abc.abstractmethod
    def create_queue(self, schedule_type: str) -> QueueBackend:
        """Create a queue for the given schedule type.

        Args:
            schedule_type: The schedule type string (e.g., 'long', 'short').
        """
        raise NotImplementedError

    def start(self) -> Optional[multiprocessing.Process]:
        """Start any required background infrastructure.

        Returns:
            A process to join on shutdown, or None if no background process
            is needed.
        """
        return None

    def stop(self, process: Optional[multiprocessing.Process]) -> None:
        """Cleanup infrastructure."""
        if process is not None:
            process.kill()


class LocalQueueBackend(QueueBackend):
    """Process-local queue (thread-safe, no IPC)."""

    def __init__(self, queue_name: str):
        super().__init__()
        # queue_name is the schedule type ('long' / 'short'); used as the
        # metric label when reporting queue wait.
        self._schedule_type = queue_name
        self._queue = local_queue.get_queue(queue_name)

    def put(self, item: Tuple[str, bool, bool]) -> None:
        internal: _InternalItem = (item, time.time())
        self._queue.put(internal)

    def get(self) -> Optional[Tuple[str, bool, bool]]:
        try:
            item, enqueued_at = self._queue.get(block=False)
        except queue_lib.Empty:
            return None
        _observe_queue_wait(self._schedule_type, enqueued_at)
        return item

    def qsize(self) -> int:
        return self._queue.qsize()


class MultiprocessingQueueBackend(QueueBackend):
    """Queue backed by a multiprocessing.Queue via a manager."""

    def __init__(self, queue_name: str, port: Optional[int] = None):
        super().__init__()
        if port is None:
            port = mp_queue.get_queue_manager_port()
        # queue_name is the schedule type ('long' / 'short'); used as the
        # metric label when reporting queue wait.
        self._schedule_type = queue_name
        self._queue = mp_queue.get_queue(queue_name, port)

    def put(self, item: Tuple[str, bool, bool]) -> None:
        internal: _InternalItem = (item, time.time())
        self._queue.put(internal)

    def get(self) -> Optional[Tuple[str, bool, bool]]:
        try:
            item, enqueued_at = self._queue.get(block=False)
        except queue_lib.Empty:
            return None
        _observe_queue_wait(self._schedule_type, enqueued_at)
        return item

    def qsize(self) -> int:
        return self._queue.qsize()


class DbQueueBackend(QueueBackend):
    """Queue backed by a table in the requests DB (no IPC process).

    Why this exists (skc#911, 2026-11-15 04:24Z twins wedge): the
    multiprocessing backend routes every enqueue AND every dequeue through a
    single BaseManager process over no-timeout synchronous RPCs. Under a poll
    burst the manager GIL serves a put storm (one thread + one connection per
    client thread; measured 1005 threads / 1013 fds at a 1000-poller burst),
    the single dispatcher's get-RPC is starved (31 executed vs 11,230 PENDING
    in 4 min with idle pool workers), and ANY manager stall freezes the whole
    request plane with no recovery. The requests DB is already the durable
    store and every writer process already has it open, so routing the queue
    through it removes the manager (and the IPC hop) entirely: put = one
    INSERT, get = one atomic DELETE ... RETURNING (SQLite >= 3.35, which the
    server already asserts). FIFO by rowid. Same-instance get/put are
    serialized by SQLite's write lock; cross-process is safe under WAL.
    """

    _TABLE = 'request_queue'

    def __init__(self, queue_name: str):
        super().__init__()
        self._schedule_type = queue_name
        self._db_path = os.path.expanduser(
            server_constants.API_SERVER_REQUEST_DB_PATH)
        self._ensure_table()

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self._db_path, timeout=60)
        conn.execute('PRAGMA busy_timeout=60000')
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    rowid_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    schedule_type TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    ignore_return_value INTEGER NOT NULL,
                    retryable INTEGER NOT NULL,
                    enqueued_at REAL NOT NULL)''')
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_request_queue_type_seq '
                f'ON {self._TABLE} (schedule_type, rowid_seq)')

    def put(self, item: Tuple[str, bool, bool]) -> None:
        internal: _InternalItem = (item, time.time())
        with self._connect() as conn:
            conn.execute(
                f'INSERT INTO {self._TABLE} (schedule_type, request_id, '
                'ignore_return_value, retryable, enqueued_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (self._schedule_type, internal[0][0],
                 int(internal[0][1]), int(internal[0][2]), internal[1]))

    async def put_async(self, item: Tuple[str, bool, bool]) -> None:
        # Same rationale as the multiprocessing backend's assumption: a single
        # WAL insert is sub-millisecond; do not force a thread hop per put.
        self.put(item)

    def get(self) -> Optional[Tuple[str, bool, bool]]:
        # Atomic claim: DELETE ... RETURNING ensures a single consumer wins a
        # row even with concurrent dispatchers (SQLite >= 3.35, asserted at
        # server startup).
        with self._connect() as conn:
            row = conn.execute(
                f'DELETE FROM {self._TABLE} WHERE rowid_seq = ('
                f'  SELECT rowid_seq FROM {self._TABLE} '
                f'  WHERE schedule_type = ? ORDER BY rowid_seq LIMIT 1) '
                f'RETURNING request_id, ignore_return_value, retryable, '
                f'enqueued_at', (self._schedule_type,)).fetchone()
        if row is None:
            return None
        _observe_queue_wait(self._schedule_type, row[3])
        return (row[0], bool(row[1]), bool(row[2]))

    def qsize(self) -> int:
        with self._connect() as conn:
            (size,) = conn.execute(
                f'SELECT COUNT(*) FROM {self._TABLE} '
                f'WHERE schedule_type = ?', (self._schedule_type,)).fetchone()
        return size


class DbQueueFactory(QueueBackendFactory):
    """Factory for DB-backed queues (no background process needed)."""

    def create_queue(self, schedule_type: str) -> QueueBackend:
        return DbQueueBackend(schedule_type)


class LocalQueueFactory(QueueBackendFactory):
    """Factory for process-local queues."""

    def create_queue(self, schedule_type: str) -> QueueBackend:
        return LocalQueueBackend(schedule_type)


class MultiprocessingQueueFactory(QueueBackendFactory):
    """Factory for multiprocessing queues with a shared manager."""

    def __init__(self, port: Optional[int] = None):
        super().__init__()
        self._port = (port if port is not None else
                      mp_queue.get_queue_manager_port())

    def create_queue(self, schedule_type: str) -> QueueBackend:
        return MultiprocessingQueueBackend(schedule_type, self._port)

    def start(self) -> Optional[multiprocessing.Process]:

        if not common_utils.is_port_available(self._port):
            raise RuntimeError(
                f'SkyPilot API server fails to start as port {self._port!r} '
                'is already in use by another process.')

        queue_names = self._get_queue_names()
        process = multiprocessing.Process(target=mp_queue.start_queue_manager,
                                          args=(queue_names, self._port))
        process.start()
        mp_queue.wait_for_queues_to_be_ready(queue_names,
                                             process,
                                             port=self._port)
        return process

    @staticmethod
    def _get_queue_names() -> List[str]:
        return [st.value for st in api_requests.ScheduleType]


_queue_backend_factory: Optional[QueueBackendFactory] = None


def get_queue_backend_factory() -> Optional[QueueBackendFactory]:
    """Get the registered queue backend factory, or None if not registered.

    Callers must fall back to the ServerConfig-selected backend; returning a
    default here would silently dead-code `ServerConfig.queue_backend`.
    """
    return _queue_backend_factory


def set_queue_backend_factory(factory: QueueBackendFactory) -> None:
    """Set the queue backend factory."""
    global _queue_backend_factory
    _queue_backend_factory = factory
