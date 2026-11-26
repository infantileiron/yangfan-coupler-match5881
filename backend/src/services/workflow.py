import hashlib
import json
import time

from fastapi import HTTPException
from sqlalchemy import and_, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from src.constants.enums import State
from src.models.database import add_event, engine, fetch_one
from src.models.tables import datasets, deliveries, runs, schedules
from src.services.data import parse_options
from src.services.domain_data import PENDING


def round_key(options: dict, records: list[dict]) -> str | None:
    """Stable identity of a bench scheduling round.

    A round is defined by the pending candidates and the analysis parameters, not by
    the trial outcomes, so a corrected data revision that fixes fixturing results maps
    to the same round and simply becomes its next version.
    """
    pending = sorted((row.get('record_id'), row.get('date')) for row in records if row.get('stage') == PENDING)
    if not pending:
        return None
    payload = json.dumps({'options': options, 'pending': pending}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def _round_key_for(dataset_id, records, options):
    key = round_key(options, records)
    if key is not None:
        return key
    fallback = json.dumps({'dataset_id': dataset_id, 'options': options}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(fallback.encode()).hexdigest()


def _next_version(connection, key):
    current = connection.scalar(select(func.max(runs.c.version)).where(runs.c.round_key == key))
    return 0 if current is None else current + 1


def add_run(body, actor, key=None):
    options = parse_options(body.options)
    now = time.time()
    with engine.begin() as connection:
        data = fetch_one(connection, datasets, body.dataset_id)
        if data is None:
            raise HTTPException(404, '数据版本不存在。')
        identity = key or _round_key_for(body.dataset_id, data['records'], options)
        version = _next_version(connection, identity)
        try:
            run_id = connection.execute(insert(runs).values(dataset_id=body.dataset_id, options=options,
                                       state=State.queued.value, owner=actor, created=now, round_key=identity,
                                       version=version, active=0)).inserted_primary_key[0]
        except IntegrityError:
            raise HTTPException(409, '同一数据版本与参数的当轮计算已在排队，请等待其完成。') from None
        add_event(connection, actor, 'queue', run_id, now)
    return {'id': run_id, 'state': State.queued.value, 'round_key': identity, 'version': version}


def requeue_corrections(dataset_id, records, actor):
    """Late corrections re-run the same round instead of starting a new one."""
    created = []
    now = time.time()
    with engine.begin() as connection:
        known = list(connection.execute(select(runs.c.round_key).distinct().where(runs.c.round_key.is_not(None))).scalars())
        if not known:
            return created
        used = set(connection.execute(select(runs.c.dataset_id, runs.c.round_key)).all())
        # MySQL cannot group JSON columns, so dedupe parameter sets in Python instead.
        option_rows = connection.execute(select(runs.c.options)).scalars().all()
        seen_options = set()
        for options in option_rows:
            marker = json.dumps(options, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            if marker in seen_options:
                continue
            seen_options.add(marker)
            identity = round_key(options, records)
            if identity is None or identity not in known:
                continue
            if (dataset_id, identity) in used:
                continue
            version = _next_version(connection, identity)
            try:
                run_id = connection.execute(insert(runs).values(dataset_id=dataset_id, options=options,
                                           state=State.queued.value, owner=actor, created=now, round_key=identity,
                                           version=version, active=0)).inserted_primary_key[0]
            except IntegrityError:
                continue
            add_event(connection, actor, 'queue', run_id, now)
            created.append({'id': run_id, 'round_key': identity, 'version': version})
    return created


def claim_run():
    now = time.time()
    claimable = or_(runs.c.state == State.queued.value, and_(runs.c.state == State.running.value, runs.c.started < now - 120))
    with engine.begin() as connection:
        run = connection.execute(select(runs).where(claimable).order_by(runs.c.id).limit(1)).mappings().one_or_none()
        if run is None:
            return None
        key = run['id']
        claim = connection.execute(update(runs).where(runs.c.id == key, claimable)
                                   .values(state=State.running.value, started=now))
        if claim.rowcount != 1:
            return None
        data = fetch_one(connection, datasets, run['dataset_id'])
        return key, now, data['records'], run['options']


def _activate_round(connection, identity, now):
    """Atomically switch a round to its newest fully computed version.

    A version only takes effect once every run of that edition has completed, so a
    late, stale computation of an older version can never overwrite a newer one; a
    newer edition that is still computing leaves the current version in service.
    Older versions keep their result, review decision, and schedules for later review.
    """
    members = connection.execute(select(runs).where(runs.c.round_key == identity)).mappings().all()
    if not members:
        return None
    editions = {}
    for run in members:
        editions.setdefault(run['version'] or 0, []).append(run)
    complete = [edition for edition, edition_runs in editions.items()
                if all(run['state'] == State.completed.value for run in edition_runs)]
    if not complete:
        return None
    # Always target the newest edition that is entirely computed: a late, stale run of an
    # older edition therefore re-triggers the switch to the newer one rather than winning.
    newest = editions[max(complete)]
    target = max(newest, key=lambda run: run['id'])
    if target['active']:
        return target['id']
    for run in members:
        if run['active'] and run['id'] != target['id']:
            connection.execute(update(runs).where(runs.c.id == run['id']).values(active=0))
    connection.execute(update(runs).where(runs.c.id == target['id']).values(active=1))
    add_event(connection, 'worker', 'activate', target['id'], now)
    return target['id']


def finish_run(key, claim, result, error):
    now = time.time()
    state = State.failed.value if error else State.completed.value
    with engine.begin() as connection:
        saved = connection.execute(update(runs).where(runs.c.id == key, runs.c.started == claim,
                                   runs.c.state == State.running.value).values(state=state, result=result,
                                                  error=error, finished=now))
        if saved.rowcount:
            add_event(connection, 'worker', state, key, now)
            identity = connection.scalar(select(runs.c.round_key).where(runs.c.id == key))
            if identity:
                _activate_round(connection, identity, now)


def review_run(key, body, actor):
    now = time.time()
    note = body.note.strip()
    if len(note) < 2:
        raise ValueError('请输入至少两个字符的复核意见。')
    with engine.begin() as connection:
        run = fetch_one(connection, runs, key)
        if run is None:
            raise HTTPException(404, '分析任务不存在。')
        if run['state'] != State.completed.value:
            raise HTTPException(409, '分析完成后才能复核。')
        if body.decision == 'approved' and run['result'].get('can_publish') is False:
            raise HTTPException(409, '结果未达到本项目的发布门槛。')
        updated = connection.execute(update(runs).where(runs.c.id == key, runs.c.decision.is_(None))
                                     .values(decision=body.decision, reviewer=actor, review_note=note, reviewed=now))
        if updated.rowcount != 1:
            raise HTTPException(409, '已完成复核。请重新运行分析以提交新的复核。')
        add_event(connection, actor, body.decision, key, now)
    return {'decision': body.decision}


def schedule_report(key, body, actor):
    now = time.time()
    recipient = body.recipient.strip()
    if len(recipient) < 2:
        raise ValueError('请填写站内收件组。')
    if body.scheduled_at and body.scheduled_at.utcoffset() is None:
        raise ValueError('投递时间须含时区。')
    timestamp = body.scheduled_at.timestamp() if body.scheduled_at else now
    if body.scheduled_at and timestamp < now - 60:
        raise ValueError('投递时间已过，请调整时间。')
    with engine.begin() as connection:
        run = fetch_one(connection, runs, key)
        if run is None or run['decision'] != 'approved':
            raise HTTPException(409, '仅已批准的结果可分发。')
        required_recipient = run['result'].get('dispatch_recipient')
        if required_recipient and recipient != required_recipient:
            raise HTTPException(422, f'该报告应投递给{required_recipient}。')
        schedule_id = connection.execute(insert(schedules).values(run_id=key, recipient=recipient, cadence=body.cadence,
                                      next_at=timestamp, active=1, owner=actor)).inserted_primary_key[0]
        add_event(connection, actor, 'schedule', schedule_id, now)
    return {'id': schedule_id, 'next_at': timestamp}


def cancel_schedule(key, actor):
    with engine.begin() as connection:
        updated = connection.execute(update(schedules).where(schedules.c.id == key, schedules.c.active == 1)
                                     .values(active=0))
        if not updated.rowcount:
            raise HTTPException(409, '该计划已停止或已完成。')
        add_event(connection, actor, 'cancel', key, time.time())
    return {'id': key, 'active': False}


def publish_due():
    now = time.time()
    with engine.begin() as connection:
        pending = connection.execute(select(schedules).where(schedules.c.active == 1, schedules.c.next_at <= now)
                                     .order_by(schedules.c.next_at, schedules.c.id).limit(20)).mappings().all()
        for item in pending:
            previous = item['next_at']
            daily = item['cadence'] == 'daily'
            future = previous + (int((now - previous) / 86400) + 1) * 86400 if daily else previous
            changed = connection.execute(update(schedules).where(schedules.c.id == item['id'], schedules.c.active == 1,
                              schedules.c.next_at == previous).values(active=int(daily), next_at=future))
            if changed.rowcount:
                delivery_id = connection.execute(insert(deliveries).values(schedule_id=item['id'], run_id=item['run_id'],
                                               recipient=item['recipient'], scheduled=previous, delivered=now)).inserted_primary_key[0]
                add_event(connection, 'worker', 'delivered', delivery_id, now)
