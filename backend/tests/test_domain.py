"""Batch-quota selection must be exact, deterministic, and match a brute-force oracle."""
import itertools
import sys
from math import fsum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.domain import _select_batch, compute, sample_rows


def brute_force(vectors, quota):
    ids = sorted(vectors)
    size = min(quota, len(ids))
    folds = len(vectors[ids[0]])
    best = None
    for choice in itertools.combinations(ids, size):
        value = min(fsum(vectors[i][fold] for i in choice) for fold in range(folds))
        candidate_ids = tuple(sorted(choice))
        if best is None or value > best[1] + 1e-12 or (abs(value - best[1]) <= 1e-12 and candidate_ids < best[0]):
            if best is None or value >= best[1] - 1e-12:
                best = (candidate_ids, max(value, best[1] if best else value))
    return best


def test_selection_matches_brute_force_with_ties():
    vectors = {
        'A': (0.5, 0.5, 0.5),
        'B': (0.5, 0.5, 0.5),
        'C': (0.8, 0.2, 0.2),
        'D': (0.2, 0.8, 0.2),
        'E': (0.2, 0.2, 0.8),
        'F': (0.7, 0.7, 0.1),
    }
    chosen, value = _select_batch(vectors, 3)
    expected_ids, expected_value = brute_force(vectors, 3)
    assert set(chosen) == set(expected_ids)
    assert abs(value - expected_value) < 1e-12


def test_quota_caps_at_eligible_count():
    vectors = {'A': (0.9,), 'B': (0.8,)}
    chosen, value = _select_batch(vectors, 5)
    assert chosen == {'A', 'B'}
    assert abs(value - 1.7) < 1e-12


def test_empty_candidates_select_nothing():
    assert _select_batch({}, 3) == (set(), 0.0)


def test_demo_batch_is_stable_and_strictly_better_than_single():
    result = compute(sample_rows(), {'quota': 3})
    ranked = [row for row in result['rows'] if row['当轮名额入选'] == '入选']
    assert [row['当轮试挂位次'] for row in ranked] == [1, 2, 3]
    assert ranked[0]['待试挂编号'] == 'PLAN-1124-07'
    single = compute(sample_rows(), {'quota': 1})
    assert result['metrics']['整批最低折预期扣合数'] > single['metrics']['整批最低折预期扣合数']


def test_insufficient_support_is_deferred():
    result = compute(sample_rows(), {'quota': 3})
    deferred = {row['待试挂编号'] for row in result['rows'] if row['试挂安排'] == '暂缓推荐'}
    assert deferred == {'PLAN-1124-05', 'PLAN-1124-06'}
