"""Coupler bench CSV fields, input validation, and deterministic demo records."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from math import isfinite
from random import Random

VEHICLES = ("短轴货车", "长轴客车", "转向架货车")
HEADS = ("短柄扣", "长柄扣", "摆臂扣")
BANDS = ("偏低", "平齐", "偏高")
TRIAL = "已试挂"
PENDING = "待试挂"
UNTESTED = -1
LEVEL_LIMIT_MM = 0.25
MAX_OFFSET_MM = 2.0


@dataclass(frozen=True)
class Profile:
    vehicle: str
    head: str
    height_band: str


@dataclass(frozen=True)
class BenchRecord:
    record_id: str
    day: date
    profile: Profile
    offset_mm: float
    stage: str
    passed: int


def height_band(offset_mm: float) -> str:
    if offset_mm < -LEVEL_LIMIT_MM:
        return BANDS[0]
    if offset_mm > LEVEL_LIMIT_MM:
        return BANDS[2]
    return BANDS[1]


def validate_records(rows: Iterable[dict]) -> list[BenchRecord]:
    records = []
    seen_ids = set()
    for position, row in enumerate(rows, start=1):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"第 {position} 条缺少试挂编号，请填写唯一编号。")
        record_id = record_id.strip()
        if record_id in seen_ids:
            raise ValueError(f"试挂编号 {record_id} 重复，请合并重复记录或使用不同编号。")
        seen_ids.add(record_id)

        day_text = row.get("date")
        try:
            day = date.fromisoformat(day_text)
            if day.isoformat() != day_text:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"记录 {record_id} 的日期无效，请填写 YYYY-MM-DD 格式的有效日期。") from None

        vehicle = row.get("vehicle")
        if vehicle not in VEHICLES:
            raise ValueError(f"记录 {record_id} 的车型无效，请填写 {'、'.join(VEHICLES)}。")
        head = row.get("head")
        if head not in HEADS:
            raise ValueError(f"记录 {record_id} 的扣头无效，请填写 {'、'.join(HEADS)}。")
        offset = row.get("offset_mm")
        if type(offset) not in (int, float) or not -MAX_OFFSET_MM <= offset <= MAX_OFFSET_MM or not isfinite(offset):
            raise ValueError(f"记录 {record_id} 的安装高度差超出台架录入范围，请复测并填写 −2 至 2 毫米的有限数值。")

        stage = row.get("stage")
        passed = row.get("passed")
        if stage not in (TRIAL, PENDING):
            raise ValueError(f"记录 {record_id} 的状态无效，请填写「已试挂」或「待试挂」。")
        allowed_outcomes = (0, 1) if stage == TRIAL else (UNTESTED,)
        if type(passed) is not int or passed not in allowed_outcomes:
            raise ValueError(f"记录 {record_id} 的结果与状态不符，已试挂请填 0 或 1，待试挂请填 −1。")
        records.append(BenchRecord(record_id, day, Profile(vehicle, head, height_band(offset)), float(offset), stage, passed))
    return sorted(records, key=lambda record: (record.day, record.record_id))


def _row(record_id: str, day: date, setup: tuple[int, int, int], offset: float, stage: str, passed: int) -> dict:
    vehicle, head, _ = setup
    return {
        "record_id": record_id,
        "date": day.isoformat(),
        "vehicle": VEHICLES[vehicle],
        "head": HEADS[head],
        "offset_mm": offset,
        "stage": stage,
        "passed": passed,
    }


def demo_rows() -> list[dict]:
    rng = Random(20261124)
    first_day = date(2026, 11, 16)
    unseen_setup = (2, 2, 2)
    rare_setup = (1, 0, 2)
    rows = []
    for day_index in range(8):
        day = first_day + timedelta(days=day_index)
        for setup in product(range(3), repeat=3):
            if setup == unseen_setup and day_index < 6:
                continue
            if setup == rare_setup and 0 < day_index < 6:
                continue
            vehicle, head, height = setup
            offset = round((-0.65, 0.0, 0.65)[height] + rng.uniform(-0.12, 0.12), 2)
            # Each marginal appears in all three outcome regimes across complete crossings.
            coupling_probability = (0.97, 0.48, 0.05)[(vehicle + head + height) % 3]
            passed = int(rng.random() < coupling_probability)
            record_id = f"TR-{day:%m%d}-{vehicle + 1}{head + 1}{height + 1}"
            rows.append(_row(record_id, day, setup, offset, TRIAL, passed))

    queue_day = date(2026, 11, 24)
    queue = (
        ((0, 0, 0), -0.66),
        ((1, 1, 1), 0.08),
        ((0, 1, 1), -0.12),
        ((2, 0, 2), 0.62),
        (rare_setup, 0.72),
        (unseen_setup, 0.70),
        ((2, 1, 0), -0.54),
    )
    for index, (setup, offset) in enumerate(queue, start=1):
        rows.append(_row(f"PLAN-1124-{index:02}", queue_day, setup, offset, PENDING, UNTESTED))
    return rows
