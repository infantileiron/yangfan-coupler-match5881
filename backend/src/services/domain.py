"""Run a dated holdout comparison and prioritize the pending coupler bench queue."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite

from .domain_data import PENDING, TRIAL, BenchRecord, Profile, demo_rows, validate_records
from .domain_model import LogisticModel, evaluate, fit


@dataclass(frozen=True)
class RunOptions:
    holdout_days: int
    regularization: float
    min_support: int
    recommend_at: float


OPTION_RULES = (
    ("holdout_days", "留出窗口天数", 2, 1, 7),
    ("regularization", "L2 正则化强度", 0.006, 0.001, 0.25),
    ("min_support", "组合最少训练记录", 5, 1, 50),
    ("recommend_at", "优先试挂概率门槛", 0.68, 0.50, 0.99),
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


def _queue_results(
    pending: list[BenchRecord],
    support: dict[Profile, list[BenchRecord]],
    crossed: LogisticModel,
    additive: LogisticModel,
    settings: RunOptions,
) -> tuple[list[dict], Counter[str]]:
    ranked = []
    actions = Counter()
    for candidate in pending:
        profile = candidate.profile
        evidence = support.get(profile, [])
        probability = None
        additive_probability = None
        if not evidence:
            action = "暂缓推荐"
            reason = "训练集未见完整组合；请先补做该车型、扣头和高度档的台架试挂。"
        elif len(evidence) < settings.min_support:
            action = "暂缓推荐"
            needed = settings.min_support - len(evidence)
            reason = f"该组合还差 {needed} 条已试挂训练记录；请补测后更新训练窗口。"
        else:
            probability = crossed.probability(profile)
            additive_probability = additive.probability(profile)
            if probability >= settings.recommend_at:
                action = "优先试挂"
                reason = "模型概率达到门槛；安排台架复试并记录实测结果。"
            else:
                action = "常规试挂"
                reason = "模型概率低于优先门槛；排在优先队列之后，并检查高度与扣头状态。"
        actions[action] += 1
        ranked.append((probability, candidate.record_id, {
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
        }))
    ranked.sort(key=lambda item: (-(item[0] if item[0] is not None else -1), item[1]))
    return [row for _, _, row in ranked], actions


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
    queue, actions = _queue_results(pending, support, crossed, additive, settings)

    return {
        "headline": f"{len(pending)} 组待试挂，{actions['优先试挂']} 组优先、{actions['暂缓推荐']} 组待补证据",
        "metrics": {
            "训练记录数": len(training),
            "留出记录数": len(holdout),
            "留出未见组合记录数": sum(trial.profile not in support for trial in holdout),
            "交叉模型对数损失": round(crossed_scores.log_loss, 6),
            "加性模型对数损失": round(additive_scores.log_loss, 6),
            "交叉模型损失改善": round(additive_scores.log_loss - crossed_scores.log_loss, 6),
            "交叉模型准确率": round(crossed_scores.accuracy, 6),
            "加性模型准确率": round(additive_scores.accuracy, 6),
            "优先试挂数": actions["优先试挂"],
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
            f"完整组合在训练集至少出现 {settings.min_support} 次才显示预测。"
            f"交叉模型通过概率达到 {settings.recommend_at * 100:g}% 的候选优先试挂。",
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
        "can_publish": True,
    }
