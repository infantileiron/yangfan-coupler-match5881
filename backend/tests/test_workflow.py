"""Round versioning: corrections re-queue, and a stale late run cannot overwrite a newer one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.workflow import round_key


OPTIONS = {'holdout_days': 2, 'regularization': 0.006, 'min_support': 5, 'recommend_at': 0.68, 'quota': 3}


def rows(outcome=1):
    return [
        {'record_id': 'A1', 'date': '2026-11-16', 'vehicle': '短轴货车', 'head': '短柄扣',
         'offset_mm': 0.0, 'stage': '已试挂', 'passed': outcome},
        {'record_id': 'P1', 'date': '2026-11-24', 'vehicle': '短轴货车', 'head': '短柄扣',
         'offset_mm': 0.0, 'stage': '待试挂', 'passed': -1},
    ]


def test_correction_keeps_same_round():
    assert round_key(OPTIONS, rows(1)) == round_key(OPTIONS, rows(0))


def test_parameter_or_candidate_change_starts_new_round():
    changed = rows()
    changed[1]['record_id'] = 'P2'
    assert round_key(OPTIONS, rows()) != round_key(OPTIONS, changed)
    assert round_key({**OPTIONS, 'quota': 4}, rows()) != round_key(OPTIONS, rows())


def test_round_key_is_order_independent():
    first = rows()
    shuffled = [first[1], first[0]]
    assert round_key(OPTIONS, first) == round_key(OPTIONS, shuffled)
