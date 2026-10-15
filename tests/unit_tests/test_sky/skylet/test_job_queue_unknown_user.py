"""A job owned by a client hash with no users row must not break the queue.

Live 2026-10-15: three `sky exec` jobs from an unseen client on a urun worker
made the dashboard's cluster jobs view fail with a ClusterJobRecord
ValidationError (username=None) for the whole cluster.
"""
from sky import global_user_state
from sky.skylet import job_lib
from sky.utils import message_utils


def test_load_job_queue_unknown_user_falls_back_to_hash(monkeypatch):
    monkeypatch.setattr(global_user_state, 'get_user', lambda _h: None)
    payload = message_utils.encode_payload([{
        'job_id': 1,
        'job_name': 'sky-cmd',
        'username': 'a34654d1',
        'status': job_lib.JobStatus.SUCCEEDED.value,
    }])

    jobs = job_lib.load_job_queue(payload)

    assert jobs[0]['user_hash'] == 'a34654d1'
    assert jobs[0]['username'] == 'a34654d1'
