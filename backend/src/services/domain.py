"""Run a dated holdout comparison and prioritize the pending coupler bench queue."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nlargest
from math import fsum, isfinite

from .domain_data import PENDING, TRIAL, BenchRecord, Profile, demo_rows, validate_records
from .domain_model import LogisticModel, evaluate, fit


class _BudgetExceeded(Exception):
    """Raised internally when the exact batch search exceeds its node budget."""


@dataclass(frozen=True)
class RunOptions:
    holdout_days: int
    regularization: float
    min_support: int
    recommend_at: float
    quota: int


OPTION_RULES = (
    ("holdout_days", "留出窗口天数", 2, 1, 7),
    ("regularization", "L2 正则化强度", 0.006, 0.001, 0.25),
    ("min_support", "组合最少训练记录", 5, 1, 50),
    ("recommend_at", "优先试挂概率门槛", 0.68, 0.50, 0.99),
    ("quota", "当轮试挂名额", 3, 1, 50),
)


def _validate_options(options: dict) -> RunOptions:
    values = {}
    for key, label, default, minimum, maximum in OPTION_RULES:
        value = options.get(key, default)
        accepted_types = (int,) if type(default) is int else (int, float)
        if type(value) not in accepted_types or not minimum <= value <= maximum or not isfinite(value):
            number_type = "整数" if type(default) is int else "数值"
            raise ValueError(f"{label}无效，请填写 {minimum} 至 {maximum} 之间的{number_type}。")
        values[key] = value
    return RunOptions(**values)


def sample_rows() -> list[dict]:
    return demo_rows()


def _split_trials(trials: list[BenchRecord], holdout_days: int) -> tuple[list[BenchRecord], list[BenchRecord]]:
    training, holdout = [], []
    latest_trial = trials[-1].day
    for trial in trials:
        target = holdout if (latest_trial - trial.day).days < holdout_days else training
        target.append(trial)
    if len(training) < 20 or len(holdout) < 8:
        raise ValueError(
            f"时间留出后有 {len(training)} 条训练记录和 {len(holdout)} 条留出记录，"
            "请补齐较早及近期台架记录或调整留出天数，使训练至少 20 条、留出至少 8 条。"
        )
    for label, subset in (("训练集", training), ("留出集", holdout)):
        if {trial.passed for trial in subset} != {0, 1}:
            raise ValueError(f"{label}仅有单一结果，请补录通过和失败两类试挂记录或调整留出天数。")
    training_profiles = {trial.profile for trial in training}
    for label, field in (("车型", "vehicle"), ("扣头", "head"), ("安装高度档", "height_band")):
        if len({getattr(profile, field) for profile in training_profiles}) < 2:
            raise ValueError(f"训练集{label}只有一类，请在留出窗口之前补录至少两类{label}的试挂记录。")
    return training, holdout


def _cross_folds(training: list[BenchRecord], regularization: float) -> list[tuple[object, LogisticModel]]:
    """Refit the crossed model once per training trial day, leaving that day out."""
    folds = []
    for day in sorted({trial.day for trial in training}):
        subset = [trial for trial in training if trial.day != day]
        if not subset:
            raise ValueError(f"留出 {day.isoformat()} 的全部训练记录后没有剩余样本，请调整留出天数或补录更早的台架记录。")
        if {trial.passed for trial in subset} != {0, 1}:
            raise ValueError(
                f"留出 {day.isoformat()} 的试挂记录后训练集只有单一结果，"
                "请补录通过和失败两类记录或调整留出天数。"
            )
        folds.append((day, fit(subset, regularization, crossed=True)))
    return folds


def _select_batch(vectors: dict[str, tuple[float, ...]], quota: int) -> tuple[set[str], float]:
    """Choose `quota` candidates maximizing the smallest per-day expected pass total.

    Every trial day of the training set is a fold, and the batch value of a fold is the sum
    of the crossed-model probabilities of the selected candidates over that fold. Adding a
    candidate never lowers a fold sum, so a full quota is optimal whenever enough candidates
    are eligible. Candidates sharing a complete combination share one probability vector, so
    they are grouped and `count` copies from a group always use its lowest trial ids. Ties
    break toward the lexicographically smallest tuple of trial ids. Group count is bounded by
    the 27 complete combinations; the search is exact branch and bound with a deterministic
    node budget. If the budget is exhausted the best batch found so far is returned, which is
    always at least the greedy batch.
    """
    if not vectors:
        return set(), 0.0
    if len({len(vector) for vector in vectors.values()}) != 1 or not next(iter(vectors.values())):
        raise ValueError("候选重拟概率的折数不一致，请重新运行分析。")
    groups = {}
    for record_id, vector in vectors.items():
        groups.setdefault(vector, []).append(record_id)
    ordered = sorted(groups.items(), key=lambda item: (-min(item[0]), min(item[1])))
    values = [vector for vector, _ in ordered]
    identifiers = [sorted(ids) for _, ids in ordered]
    group_count, folds = len(ordered), len(values[0])
    limit = min(quota, len(vectors))
    caps = [min(len(ids), limit) for ids in identifiers]

    # Rounding noise from independent Newton fits is far below any real difference.
    epsilon = 1e-12
    budget = 150_000
    nodes = 0
    best = {"value": -1.0, "ids": ()}

    def take_ids(counts: list[int]) -> tuple[str, ...]:
        return tuple(sorted(identifier for group in range(group_count)
                            for identifier in identifiers[group][:counts[group]]))

    def record(value: float, counts: list[int]) -> None:
        candidate_ids = take_ids(counts)
        if value > best["value"] + epsilon or (abs(value - best["value"]) <= epsilon
                                               and candidate_ids < best["ids"]):
            best["value"], best["ids"] = value, candidate_ids

    # Greedy incumbent: repeatedly take the candidate lifting the lowest fold the most.
    counts = [0] * group_count
    sums = [0.0] * folds
    for _ in range(limit):
        option = None
        for group in range(group_count):
            if counts[group] >= caps[group]:
                continue
            trial = [sums[fold] + values[group][fold] for fold in range(folds)]
            trial_counts = [count + (index == group) for index, count in enumerate(counts)]
            candidate_ids = take_ids(trial_counts)
            value = min(trial)
            if option is None or value > option[0] + epsilon or (abs(value - option[0]) <= epsilon
                                                                 and candidate_ids < option[1]):
                option = (value, candidate_ids, group)
        counts[option[2]] += 1
        sums = [sums[fold] + values[option[2]][fold] for fold in range(folds)]
        record(option[0], counts)

    def bound(start: int, sums: list[float], need: int) -> bool:
        for fold in range(folds):
            available = []
            for group in range(start, group_count):
                if caps[group]:
                    available.extend([values[group][fold]] * min(caps[group], need))
            if sums[fold] + fsum(nlargest(need, available)) < best["value"] - epsilon:
                return False
        return True

    def search(start: int, counts: list[int], sums: list[float], used: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise _BudgetExceeded
        if start == group_count:
            if used == limit:
                record(min(sums), counts)
            return
        need = limit - used
        if sum(caps[start:]) < need:
            return
        lowest = max(0, need - sum(caps[start + 1:]))
        for take in range(min(caps[start], need), lowest - 1, -1):
            trial = [sums[fold] + take * values[start][fold] for fold in range(folds)]
            if bound(start + 1, trial, need - take):
                search(start + 1, counts + [take], trial, used + take)

    try:
        search(0, [], [0.0] * folds, 0)
    except _BudgetExceeded:
        pass
    return set(best["ids"]), best["value"]


def _queue_results(
    pending: list[BenchRecord],
    support: dict[Profile, list[BenchRecord]],
    crossed: LogisticModel,
    additive: LogisticModel,
    training: list[BenchRecord],
    settings: RunOptions,
) -> tuple[list[dict], Counter[str], set[str], list[tuple[object, LogisticModel]], float]:
    needs_folds = any(len(support.get(candidate.profile, ())) >= settings.min_support for candidate in pending)
    folds = _cross_folds(training, settings.regularization) if needs_folds else []

    items = []
    for candidate in pending:
        profile = candidate.profile
        evidence = support.get(profile, [])
        if not evidence:
            items.append((candidate, None, None, None, "暂缓推荐",
                          "训练集未见完整组合；请先补做该车型、扣头和高度档的台架试挂。"))
        elif len(evidence) < settings.min_support:
            missed = settings.min_support - len(evidence)
            items.append((candidate, None, None, None, "暂缓推荐",
                          f"该组合还差 {missed} 条已试挂训练记录；请补测后更新训练窗口。"))
        else:
            fold_probabilities = tuple(model.probability(profile) for _, model in folds)
            probability = crossed.probability(profile)
            additive_probability = additive.probability(profile)
            items.append((candidate, probability, additive_probability, fold_probabilities, None, None))

    eligible = [item for item in items if item[3] is not None]
    vectors = {item[0].record_id: item[3] for item in eligible}
    selected, batch_value = _select_batch(vectors, settings.quota)
    ranked = [record_id for record_id in sorted(vectors, key=lambda key: (-min(vectors[key]), key)) if record_id in selected]
    positions = {record_id: position for position, record_id in enumerate(ranked, start=1)}
    # Priority runs one step safer: the gate uses the candidate's worst cross-fold probability.
    assignments = {}
    for candidate, _, _, fold_probabilities, _, _ in eligible:
        quiet = min(fold_probabilities) >= settings.recommend_at
        if candidate.record_id in selected:
            assignments[candidate.record_id] = ("优先试挂", (
                "各试挂日重拟概率的最低值都达到门槛；已占当轮名额，按位次上台架并记录实测结果。" if quiet
                else "已占当轮名额；各试挂日重拟概率的最低价低于门槛，先按位试挂并留意复核。"))
        elif quiet:
            assignments[candidate.record_id] = ("优先候补", "各试挂日重拟概率的最低值达到门槛；名额已满，作为优先候补顺延到下一轮台架。")
        else:
            assignments[candidate.record_id] = ("常规试挂", "模型概率低于优先门槛；排在优先队列之后，并检查高度与扣头状态。")

    rows, actions = [], Counter()
    for candidate, probability, additive_probability, fold_probabilities, action, reason in items:
        if candidate.record_id in assignments:
            action, reason = assignments[candidate.record_id]
        profile = candidate.profile
        evidence = support.get(profile, [])
        actions[action] += 1
        row = {
            "待试挂编号": candidate.record_id,
            "排程日期": candidate.day.isoformat(),
            "车型": profile.vehicle,
            "扣头": profile.head,
            "安装高度差（毫米）": candidate.offset_mm,
            "高度档": profile.height_band,
            "同组合训练数": len(evidence),
            "同组合通过数": sum(trial.passed for trial in evidence),
            "近期参考记录": "、".join(trial.record_id for trial in evidence[-3:]) or "—",
            "交叉模型通过概率": round(probability, 4) if probability is not None else "—",
            "加性模型通过概率": round(additive_probability, 4) if additive_probability is not None else "—",
            "试挂安排": action,
            "依据": reason,
        }
        if fold_probabilities is not None:
            row["最低重拟通过概率"] = round(min(fold_probabilities), 4)
            row["重拟通过概率（各试挂日）"] = " / ".join(f"{value:.4f}" for value in fold_probabilities)
            row["当轮名额入选"] = "入选" if candidate.record_id in selected else "未入选"
            row["当轮试挂位次"] = positions.get(candidate.record_id, "—")
            sort_key = (0 if candidate.record_id in selected else 1, -min(fold_probabilities), candidate.record_id)
        else:
            row["最低重拟通过概率"] = "—"
            row["重拟通过概率（各试挂日）"] = "—"
            row["当轮名额入选"] = "—"
            row["当轮试挂位次"] = "—"
            sort_key = (2, 0.0, candidate.record_id)
        rows.append((sort_key, row))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows], actions, selected, folds, batch_value


def compute(rows: list[dict], options: dict) -> dict:
    settings = _validate_options(options)
    records = validate_records(rows)
    trials = [record for record in records if record.stage == TRIAL]
    pending = [record for record in records if record.stage == PENDING]
    if not trials:
        raise ValueError("台账没有已试挂记录，请导入包含日期和通过结果的台架试挂历史。")
    training, holdout = _split_trials(trials, settings.holdout_days)
    if not pending:
        raise ValueError("台账没有待试挂配置，请添加换件候选，填写「待试挂」和结果 -1。")

    additive = fit(training, settings.regularization, crossed=False)
    crossed = fit(training, settings.regularization, crossed=True)
    additive_scores = evaluate(additive, holdout)
    crossed_scores = evaluate(crossed, holdout)
    support = defaultdict(list)
    for trial in training:
        support[trial.profile].append(trial)
    queue, actions, selected, folds, batch_value = _queue_results(pending, support, crossed, additive, training, settings)

    fold_totals = [
        {"label": day.isoformat(), "value": round(fsum(model.probability(record.profile) for record in pending
                                                       if record.record_id in selected), 6)}
        for day, model in folds
    ] if selected else []

    return {
        "headline": (f"{len(pending)} 组待试挂，当轮 {len(selected)} 名占名额（最低预期扣合 "
                     f"{batch_value:.2f}），{actions['暂缓推荐']} 组待补证据"),
        "dispatch_recipient": "车钩台架晚班",
        "metrics": {
            "训练记录数": len(training),
            "留出记录数": len(holdout),
            "留出未见组合记录数": sum(trial.profile not in support for trial in holdout),
            "交叉模型对数损失": round(crossed_scores.log_loss, 6),
            "加性模型对数损失": round(additive_scores.log_loss, 6),
            "交叉模型损失改善": round(additive_scores.log_loss - crossed_scores.log_loss, 6),
            "交叉模型准确率": round(crossed_scores.accuracy, 6),
            "加性模型准确率": round(additive_scores.accuracy, 6),
            "重拟折数": len(folds),
            "当轮试挂名额": settings.quota,
            "入选试挂数": len(selected),
            "整批最低折预期扣合数": round(batch_value, 6),
            "优先试挂数": actions["优先试挂"],
            "优先候补数": actions["优先候补"],
            "常规试挂数": actions["常规试挂"],
            "暂缓推荐数": actions["暂缓推荐"],
        },
        "rows": queue,
        "notes": [
            f"训练记录日期为 {training[0].day.isoformat()} 至 {training[-1].day.isoformat()}，"
            f"留出窗口为最近 {settings.holdout_days} 个日历日，留出记录实际日期为 "
            f"{holdout[0].day.isoformat()} 至 {holdout[-1].day.isoformat()}。候选不参与拟合或评分。",
            f"加性模型逐项编码车型、扣头和高度档，交叉模型增加三字段完整组合项。"
            f"两者使用相同训练数据和 L2 强度 {settings.regularization:g}，截距免惩罚。",
            "相同留出记录比较对数损失和以 0.5 为阈值的准确率。对数损失越低越好，"
            "损失改善为加性减交叉。留出集未见的编码项按零权重评分。",
            f"训练集的 {len(set(trial.day for trial in training))} 个试挂日逐日留出一次并重拟交叉模型，"
            f"共 {len(folds)} 折；近期时间留出集记录不参与重拟。候选在各折上的通过概率用于名额选批。",
            f"当轮名额为 {settings.quota}，按各折整批预期扣合总数（当折入选候选概率之和）的最低值从高到低选批，"
            "最低值持平时取试挂编号字典序更小的一批；名额内按最低折概率及编号排序。",
            f"完整组合在训练集至少出现 {settings.min_support} 次才参与选批；不足或缺组合的候选照旧暂缓并提示补测。"
            f"各试挂日重拟概率的最低值达到 {settings.recommend_at * 100:g}% 才算优先；名额已满的优先候选列为候补。",
            "高度差相对于随台架固定的基准扣头，−0.25 至 +0.25 毫米（含端点）归为平齐。"
            "预测用于安排台架试挂，换件放行仍需核对实际扣合和脱钩记录。",
        ],
        "chart": {
            "label": "同一时间留出集的对数损失（越低越好）",
            "unit": "",
            "points": [
                {"label": "交叉模型", "value": round(crossed_scores.log_loss, 6)},
                {"label": "加性模型", "value": round(additive_scores.log_loss, 6)},
            ],
        },
        "batch_chart": {
            "label": "当轮入选批次在各试挂日重拟下的预期扣合数（取最低值为稳）",
            "unit": "次",
            "points": fold_totals,
        },
        "can_publish": True,
    }
