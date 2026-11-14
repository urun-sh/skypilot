"""Tests for the API-server request store backends (SQLite / Postgres)."""

import os

import pytest

from sky.server.requests import requests as requests_module
from sky.utils.db import db_utils


@pytest.fixture
def fresh_request_store(monkeypatch, tmp_path):
    """Reset the request-store singleton, env, and DB path per test."""
    from sky.server import constants as server_constants

    monkeypatch.delenv('SKYPILOT_DB_CONNECTION_URI', raising=False)
    monkeypatch.delenv('IS_SKYPILOT_SERVER', raising=False)
    monkeypatch.setattr(server_constants, 'API_SERVER_REQUEST_DB_PATH',
                        str(tmp_path / 'api.db'))
    requests_module._DB = None
    yield
    requests_module._DB = None


def _roundtrip_sync(db):
    with db.conn:
        cursor = db.conn.cursor()
        cursor.execute(
            'INSERT INTO requests (request_id, status, created_at) '
            'VALUES (?, ?, ?)', ('test-sync-id', 'PENDING', 1.0))
    with db.conn:
        cursor = db.conn.cursor()
        cursor.execute(
            'SELECT request_id, status FROM requests WHERE request_id = ?',
            ('test-sync-id',))
        row = cursor.fetchone()
    assert row is not None and row[0] == 'test-sync-id' and row[1] == 'PENDING'


async def _roundtrip_async(db):
    await db.execute_and_commit_async(
        'INSERT INTO requests (request_id, status, created_at) '
        'VALUES (?, ?, ?)', ('test-async-id', 'RUNNING', 2.0))
    async with db.execute_fetchall_async(
            'SELECT request_id, status FROM requests '
            'WHERE request_id = ?', ('test-async-id',)) as rows:
        assert len(rows) == 1 and rows[0][0] == 'test-async-id'
    # ON CONFLICT ... RETURNING: first insert wins, second is a no-op.
    row = await db.execute_get_returning_value_async(
        'INSERT INTO requests (request_id, status, created_at) '
        'VALUES (?, ?, ?) ON CONFLICT(request_id) DO NOTHING '
        'RETURNING ROWID', ('test-returning-id', 'PENDING', 3.0))
    assert row is not None
    row = await db.execute_get_returning_value_async(
        'INSERT INTO requests (request_id, status, created_at) '
        'VALUES (?, ?, ?) ON CONFLICT(request_id) DO NOTHING '
        'RETURNING ROWID', ('test-returning-id', 'PENDING', 3.0))
    assert row is None


def test_sqlite_fallback_default(fresh_request_store):
    """Without Postgres env, the request store must stay on SQLite."""
    assert not db_utils.is_postgres_state_configured()
    requests_module._init_db_within_lock()
    assert isinstance(requests_module._DB, db_utils.SQLiteConn)
    _roundtrip_sync(requests_module._DB)


def test_postgres_selected_when_configured(
        fresh_request_store, monkeypatch):
    """Backend selection flips to Postgres with server+URI env set."""
    monkeypatch.setenv('IS_SKYPILOT_SERVER', '1')
    monkeypatch.setenv('SKYPILOT_DB_CONNECTION_URI',
                       'postgresql://user:pass@localhost:5432/sky')
    assert db_utils.is_postgres_state_configured()


def test_not_configured_without_server_mode(fresh_request_store,
                                            monkeypatch):
    """The URI alone (no IS_SKYPILOT_SERVER) must not flip the backend."""
    monkeypatch.setenv('SKYPILOT_DB_CONNECTION_URI',
                       'postgresql://user:pass@localhost:5432/sky')
    assert not db_utils.is_postgres_state_configured()


def test_sqlite_dialect_translation():
    """INSERT OR REPLACE becomes a Postgres upsert; bools become ints."""
    sql = ('INSERT OR REPLACE INTO requests '
           '(request_id, status, should_retry) VALUES (?, ?, ?)')
    out, params = db_utils._translate_sql_for_pg_driver(sql, ('r1', True, True))
    assert 'INSERT OR REPLACE' not in out
    assert ('ON CONFLICT (request_id) DO UPDATE SET '
            'request_id = EXCLUDED.request_id, '
            'status = EXCLUDED.status, '
            'should_retry = EXCLUDED.should_retry') in out
    assert params == ('r1', 1, 1)
    # No parameters: literal % passes through untouched (psycopg only
    # interpolates when variables are passed).
    out, params = db_utils._translate_sql_for_pg_driver(
        "SELECT request_id FROM requests WHERE request_id LIKE '%-daemon'",
        None)
    assert out == "SELECT request_id FROM requests WHERE request_id LIKE '%-daemon'"
    assert params == ()


@pytest.mark.skipif(
    not os.environ.get('SKYPILOT_TEST_PG_DB_URI'),
    reason='SKYPILOT_TEST_PG_DB_URI not set; the live proof covers this path')
async def test_postgres_backend_roundtrip(fresh_request_store, monkeypatch):
    """Full sync+async roundtrip against a live Postgres."""
    monkeypatch.setenv('IS_SKYPILOT_SERVER', '1')
    monkeypatch.setenv('SKYPILOT_DB_CONNECTION_URI',
                       os.environ['SKYPILOT_TEST_PG_DB_URI'])
    requests_module._init_db_within_lock()
    assert isinstance(requests_module._DB, db_utils.PostgresConn)
    db = requests_module._DB
    _roundtrip_sync(db)
    await _roundtrip_async(db)
    await db.close()
