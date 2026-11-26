"""Round versioning: late corrections re-queue, and a stale late run cannot overwrite."""
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.database import engine
from src.models.tables import datasets, metadata, runs
from src.services import workflow

OPTS = {'holdout_days': 2, 'regularization': 0.006, 'min_support': 5, 'recommend_at': 0.68, 'quota': 3}
ROWS = [
    {'record_id': 'A1', 'date': '2026-11-16', 'vehicle': '短轴货车', 'head': '短柄扣',
     'offset_mm': 0.0, 'stage': '已试挂', 'passed': 1},
    {'record_id': 'A2', 'date': '2026-11-17', 'vehicle': '长轴客车', 'head': '长柄扣',
     'offset_mm': 0.0, 'stage': '已试挂', 'passed': 0},
    {'record_id': 'P1', 'date': '2026-11-24', 'vehicle': '短轴货车', 'head': '短柄扣',
     'offset_mm': 0.0, 'stage': '待试挂', 'passed': -1},
]


class Body:
    def __init__(self, dataset_id, options):
        self.dataset_id = dataset_id
        self.options = options


@pytest.fixture(autouse=True)
def fresh_schema():
    metadata.drop_all(engine)
    metadata.create_all(engine)


def add_revision(name, records):
    with engine.begin() as connection:
        return connection.execute(datasets.insert().values(
            name=name, checksum=name * 16, records=records, owner='seed', created=time.time()
        )).inserted_primary_key[0]


def snapshot():
    with engine.connect() as connection:
        return sorted((row['id'], row['version'], row['active'], row['state'])
                      for row in connection.execute(select(runs)).mappings())


def finish_all():
    jobs = []
    while (job := workflow.claim_run()) is not None:
        jobs.append(job)
    return jobs


def test_correction_requeues_same_round_and_switches_atomically():
    original = add_revision('d1', ROWS)
    first = workflow.add_run(Body(original, OPTS), 'analyst')
    job = workflow.claim_run()
    workflow.finish_run(job[0], job[1], {'can_publish': True}, None)
    assert snapshot() == [(first['id'], 0, 1, 'completed')]

    corrected = [dict(row) for row in ROWS]
    corrected[0]['passed'] = 0
    revision = add_revision('d2', corrected)
    queued = workflow.requeue_corrections(revision, corrected, 'analyst')
    assert queued == [{'id': first['id'] + 1, 'round_key': first['round_key'], 'version': 1}]
    # The correction is queued but the old reviewed version stays in service.
    assert snapshot() == [(first['id'], 0, 1, 'completed'), (first['id'] + 1, 1, 0, 'queued')]

    job = workflow.claim_run()
    workflow.finish_run(job[0], job[1], {'can_publish': True}, None)
    assert snapshot() == [(first['id'], 0, 0, 'completed'), (first['id'] + 1, 1, 1, 'completed')]


def test_stale_late_run_does_not_overwrite_newer_version():
    original = add_revision('d1', ROWS)
    first = workflow.add_run(Body(original, OPTS), 'analyst')
    stale = workflow.claim_run()
    corrected = [dict(row) for row in ROWS]
    corrected[0]['passed'] = 0
    revision = add_revision('d2', corrected)
    workflow.requeue_corrections(revision, corrected, 'analyst')
    newer = workflow.claim_run()

    # The stale older version finishes first while the newer one is still running.
    workflow.finish_run(stale[0], stale[1], {'can_publish': True}, None)
    # The older version cannot switch the round in because the newer edition is unfinished.
    assert snapshot() == [(first['id'], 0, 1, 'completed'), (newer[0], 1, 0, 'running')]
    workflow.finish_run(newer[0], newer[1], {'can_publish': True}, None)
    assert snapshot() == [(first['id'], 0, 0, 'completed'), (newer[0], 1, 1, 'completed')]

    # A late re-delivery of the old result must leave the round on the newer version.
    with engine.begin() as connection:
        connection.execute(runs.update().where(runs.c.id == stale[0]).values(state='running', started=stale[1]))
    workflow.finish_run(stale[0], stale[1], {'can_publish': True}, None)
    assert snapshot() == [(first['id'], 0, 0, 'completed'), (newer[0], 1, 1, 'completed')]


def test_duplicate_revision_does_not_requeue():
    add_revision('d1', ROWS)
    workflow.add_run(Body(1, OPTS), 'analyst')
    assert workflow.requeue_corrections(1, ROWS, 'analyst') == []
