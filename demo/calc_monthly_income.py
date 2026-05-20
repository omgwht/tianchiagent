"""计算 2026 年 3 月每个司机累计收益（与仿真结果 JSONL 对齐）。

输出 `monthly_income_202603.json`：`drivers` 为数组，每项含该司机 `driver_id`、`income`、`token_usage`；
全量汇总在 `summary`。偏好罚分按 `server/data/drivers.json` 中每条规则的 penalty 字段计。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEMO_PACKAGE_ROOT = _REPO_ROOT / "demo"
if str(_DEMO_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_PACKAGE_ROOT))

from simkit.simulation_actions import haversine_km


_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)

# --- 地理与时间常量（与 drivers.json 描述一致）---
SHENZHEN_LAT_MIN = 22.42
SHENZHEN_LAT_MAX = 22.89
SHENZHEN_LNG_MIN = 113.74
SHENZHEN_LNG_MAX = 114.66

MAR10_DAY_IDX = 9
MAR10_10_MIN = MAR10_DAY_IDX * 1440 + 10 * 60
MAR13_DAY_IDX = 12
MAR13_22_MIN = MAR13_DAY_IDX * 1440 + 22 * 60
# D010 家事：自 3/10 10:00 起 72 小时窗口（至 3/13 10:00）
D010_HOME_WINDOW_END_MIN = MAR10_10_MIN + 72 * 60

D010_PICKUP_LAT, D010_PICKUP_LNG = 23.21, 113.37
D010_HOME_LAT, D010_HOME_LNG = 23.19, 113.36

D003_FORBIDDEN_ZONE_LAT = 23.30
D003_FORBIDDEN_ZONE_LNG = 113.52
D003_FORBIDDEN_ZONE_RADIUS_KM = 20.0


def _resolve_config_json(server_config_dir: Path) -> Path:
    primary = server_config_dir / "config.json"
    if primary.is_file():
        return primary
    fallback = server_config_dir / "config.example.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"缺少 server 配置: {primary} 或 {fallback}")


def _parse_epoch_minutes(ts: str) -> int:
    return int((_SIMULATION_EPOCH.fromisoformat(ts.strip().replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)


@dataclass(frozen=True)
class PreferenceRuleSpec:
    """drivers.json 中单条 preference 的评测口径字段。"""

    content: str
    start_minutes: int
    end_minutes: int
    penalty_amount: float
    penalty_cap: float | None


def load_cargo_map(path: Path) -> dict[str, dict[str, Any]]:
    cargo_map: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cargo_id = str(item.get("cargo_id", "")).strip()
            if not cargo_id:
                continue
            try:
                start = item.get("start", {})
                end = item.get("end", {})
                distance_km = haversine_km(
                    float(start["lat"]),
                    float(start["lng"]),
                    float(end["lat"]),
                    float(end["lng"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"货源文件第 {line_no} 行 start/end 坐标无效") from exc
            create_time = str(item.get("create_time", "")).strip()
            remove_time = str(item.get("remove_time", "")).strip()
            if not create_time or not remove_time:
                raise ValueError(f"货源文件第 {line_no} 行缺少 create_time/remove_time")
            create_minutes = _parse_epoch_minutes(create_time)
            remove_minutes = _parse_epoch_minutes(remove_time)
            cost_time_minutes = int(item.get("cost_time_minutes", 0) or 0)
            load_window = item.get("load_time")
            load_start_minutes: int | None = None
            load_end_minutes: int | None = None
            if load_window is not None:
                if not isinstance(load_window, list) or len(load_window) != 2:
                    raise ValueError(f"货源文件第 {line_no} 行 load_time 格式无效")
                left = str(load_window[0]).strip()
                right = str(load_window[1]).strip()
                if not left or not right:
                    raise ValueError(f"货源文件第 {line_no} 行 load_time 为空")
                load_start_minutes = _parse_epoch_minutes(left)
                load_end_minutes = _parse_epoch_minutes(right)
                if load_end_minutes < load_start_minutes:
                    load_start_minutes = None
                    load_end_minutes = None
            cargo_name = str(item.get("cargo_name", "") or "").strip()
            cargo_map[cargo_id] = {
                "price": float(item.get("price", 0.0)) / 100.0,
                "distance_km": distance_km,
                "create_minutes": create_minutes,
                "remove_minutes": remove_minutes,
                "start_lat": float(start["lat"]),
                "start_lng": float(start["lng"]),
                "end_lat": float(end["lat"]),
                "end_lng": float(end["lng"]),
                "cost_time_minutes": cost_time_minutes,
                "load_start_minutes": load_start_minutes,
                "load_end_minutes": load_end_minutes,
                "cargo_name": cargo_name,
            }
    return cargo_map


def load_driver_cost_map(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("drivers.json 必须为数组")
    cost_map: dict[str, float] = {}
    for item in raw:
        driver_id = str(item.get("driver_id", "")).strip()
        if not driver_id:
            continue
        cost_per_km = float(item.get("cost_per_km", 0.0))
        if cost_per_km < 0:
            raise ValueError(f"driver {driver_id} 的 cost_per_km 不能为负数")
        cost_map[driver_id] = cost_per_km
    return cost_map


def _preference_entry_to_rule(entry: Any) -> PreferenceRuleSpec | None:
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            return None
        return PreferenceRuleSpec(
            content=text,
            start_minutes=0,
            end_minutes=31 * 1440,
            penalty_amount=0.0,
            penalty_cap=0.0,
        )
    if not isinstance(entry, dict):
        return None
    raw_text = entry.get("content")
    if raw_text is None:
        raw_text = entry.get("text")
    content = str(raw_text or "").strip()
    if not content:
        return None
    st = str(entry.get("start_time", "") or "").strip()
    et = str(entry.get("end_time", "") or "").strip()
    start_minutes = _parse_epoch_minutes(st) if st else 0
    end_minutes = _parse_epoch_minutes(et) if et else 31 * 1440
    penalty_amount = float(entry.get("penalty_amount", 0.0) or 0.0)
    cap_raw = entry.get("penalty_cap")
    penalty_cap: float | None
    if cap_raw is None:
        penalty_cap = None
    else:
        penalty_cap = float(cap_raw)
    return PreferenceRuleSpec(
        content=content,
        start_minutes=start_minutes,
        end_minutes=end_minutes,
        penalty_amount=penalty_amount,
        penalty_cap=penalty_cap,
    )


def load_driver_preference_rules(path: Path) -> dict[str, list[PreferenceRuleSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("drivers.json 必须为数组")
    out: dict[str, list[PreferenceRuleSpec]] = {}
    for item in raw:
        driver_id = str(item.get("driver_id", "")).strip()
        if not driver_id:
            continue
        prefs = item.get("preferences") or []
        if not isinstance(prefs, list):
            prefs = []
        rules: list[PreferenceRuleSpec] = []
        for p in prefs:
            spec = _preference_entry_to_rule(p)
            if spec is not None:
                rules.append(spec)
        out[driver_id] = rules
    return out


def load_driver_preferences_map(path: Path) -> dict[str, list[str]]:
    rules_map = load_driver_preference_rules(path)
    return {did: [r.content for r in rules] for did, rules in rules_map.items()}


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("actions_202603_*.jsonl"))


def load_simulate_time_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("simulate_time_seconds")
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return None


def load_reposition_speed_km_per_hour(config_path: Path) -> float:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    value = raw.get("reposition_speed_km_per_hour")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{config_path.name} 缺少有效字段 reposition_speed_km_per_hour")
    return float(value)


def load_simulation_duration_days(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("simulation_duration_days")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("run_summary_202603.json 缺少有效字段 simulation_duration_days")
    return min(int(value), 30)


def _nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= eps


def _distance_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    if distance_km <= 0:
        return 1
    return max(1, int(math.ceil((distance_km / speed_km_per_hour) * 60.0)))


def _iter_day_segments(start_min: int, end_min: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if end_min <= start_min:
        return out
    cur = start_min
    while cur < end_min:
        day_idx = cur // 1440
        day_end = (day_idx + 1) * 1440
        seg_end = min(day_end, end_min)
        out.append((day_idx, seg_end - cur))
        cur = seg_end
    return out


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def _longest_merged_span_minutes(intervals: list[tuple[int, int]]) -> int:
    longest = 0
    for s, e in _merge_intervals(intervals):
        longest = max(longest, e - s)
    return longest


def _in_shenzhen(lat: float, lng: float) -> bool:
    return SHENZHEN_LAT_MIN <= lat <= SHENZHEN_LAT_MAX and SHENZHEN_LNG_MIN <= lng <= SHENZHEN_LNG_MAX


def _night_windows_23_to_6(day: int) -> tuple[int, int, int, int]:
    w1s = day * 1440 + 23 * 60
    w1e = (day + 1) * 1440
    w2s = (day + 1) * 1440
    w2e = (day + 1) * 1440 + 6 * 60
    return w1s, w1e, w2s, w2e


def _sum_deadhead_km(ctxs: list[dict[str, Any]]) -> float:
    total = 0.0
    for c in ctxs:
        res = c.get("result") if isinstance(c.get("result"), dict) else {}
        if c["action_name"] == "reposition":
            total += float(res.get("distance_km", 0.0) or 0.0)
        elif c["action_name"] == "take_order" and bool(res.get("accepted", False)):
            total += float(res.get("pickup_deadhead_km", 0.0) or 0.0)
    return total


def _wait_intervals_for_day(ctxs: list[dict[str, Any]], day: int) -> list[tuple[int, int]]:
    d0 = day * 1440
    d1 = d0 + 1440
    intervals: list[tuple[int, int]] = []
    for c in ctxs:
        if c["action_name"] != "wait" or c["action_exec_cost"] <= 0:
            continue
        s = max(c["step_start"], d0)
        e = min(c["step_end"], d1)
        if e > s:
            intervals.append((s, e))
    return intervals


def _active_minutes_by_day(ctxs: list[dict[str, Any]], days: list[int]) -> dict[int, int]:
    active: dict[int, int] = {d: 0 for d in days}
    for c in ctxs:
        if c["action_name"] not in {"take_order", "reposition"}:
            continue
        for d, seg in _iter_day_segments(c["action_start"], c["action_end"]):
            active[d] = active.get(d, 0) + seg
    return active


def _calendar_weekday_202603(day_idx: int) -> int:
    """0=周一 … 6=周日。"""
    base = date(2026, 3, 1)
    return (base + timedelta(days=day_idx)).weekday()


def _build_step_contexts(file_path: Path) -> list[dict[str, Any]]:
    ctxs: list[dict[str, Any]] = []
    prev_end_minutes = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record: dict[str, Any] = json.loads(row)
            step_elapsed = int(record.get("step_elapsed_minutes", -1))
            query_scan_cost = int(record.get("query_scan_cost_minutes", -1))
            action_exec_cost = int(record.get("action_exec_cost_minutes", -1))
            result = record.get("result", {})
            end_minutes = int(result.get("simulation_progress_minutes", -1))
            if min(step_elapsed, query_scan_cost, action_exec_cost, end_minutes) < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少步骤耗时字段")
            step_start = prev_end_minutes
            action_start = step_start + query_scan_cost
            action_end = action_start + action_exec_cost
            pos_before = record.get("position_before", {}) or {}
            pos_after = record.get("position_after", {}) or {}
            action_obj = record.get("action") or {}
            ctxs.append(
                {
                    "line_no": line_no,
                    "action_name": str(action_obj.get("action", "")).strip().lower(),
                    "params": action_obj.get("params", {}) or {},
                    "result": result if isinstance(result, dict) else {},
                    "step_start": step_start,
                    "action_start": action_start,
                    "action_end": action_end,
                    "step_end": end_minutes,
                    "action_exec_cost": action_exec_cost,
                    "before_lat": float(pos_before.get("lat", 0.0)),
                    "before_lng": float(pos_before.get("lng", 0.0)),
                    "after_lat": float(pos_after.get("lat", 0.0)),
                    "after_lng": float(pos_after.get("lng", 0.0)),
                }
            )
            prev_end_minutes = end_minutes
    return ctxs


class DriverPreferenceCalculatorBase(ABC):
    """司机偏好罚分计算器基类。"""

    driver_id: str

    @abstractmethod
    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError


class DriverD001PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D001"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2 = rules[0], rules[1], rules[2]

        violation_rest = 0
        for day in days:
            intervals = _wait_intervals_for_day(ctxs, day)
            if _longest_merged_span_minutes(intervals) < 8 * 60:
                violation_rest += 1
        pen0 = min(violation_rest * r0.penalty_amount, r0.penalty_cap or float("inf"))
        total += pen0
        detail_rules.append(
            {"rule": "每日连续熄火休息≥8小时", "violations": violation_rest, "penalty": round(pen0, 2), "preference_text": r0.content}
        )

        forbidden = {"化工塑料", "煤炭矿产"}
        bad_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order":
                continue
            if not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            name = str(cargo_map.get(cid, {}).get("cargo_name", "") or "")
            if name in forbidden:
                bad_orders += 1
        pen1 = min(bad_orders * r1.penalty_amount, r1.penalty_cap or float("inf"))
        total += pen1
        detail_rules.append({"rule": "禁接化工塑料/煤炭矿产", "violations": bad_orders, "penalty": round(pen1, 2), "preference_text": r1.content})

        sz_violations = 0
        for c in ctxs:
            if not _in_shenzhen(c["before_lat"], c["before_lng"]) or not _in_shenzhen(c["after_lat"], c["after_lng"]):
                sz_violations += 1
        pen2 = min(sz_violations * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "深圳市范围内行驶与停车", "violations": sz_violations, "penalty": round(pen2, 2), "preference_text": r2.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD002PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D002"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2 = rules[0], rules[1], rules[2]

        # 规则：自然月内至少 4 个自然日「无成交接单」；空驶/ reposition、wait 均不抵消该日计数。
        free_days = 0
        for day in days:
            took_accepted = False
            for c in ctxs:
                if c["action_name"] != "take_order":
                    continue
                if c["step_end"] // 1440 != day:
                    continue
                if bool(c["result"].get("accepted", False)):
                    took_accepted = True
                    break
            if not took_accepted:
                free_days += 1
        pen0 = 0.0 if free_days >= 4 else (r0.penalty_cap or r0.penalty_amount)
        total += pen0
        detail_rules.append({"rule": "自然月至少4整天无成交接单（空驶不计）", "free_days": free_days, "penalty": round(pen0, 2), "preference_text": r0.content})

        veg_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            if str(cargo_map.get(cid, {}).get("cargo_name", "") or "") == "蔬菜":
                veg_orders += 1
        pen1 = min(veg_orders * r1.penalty_amount, r1.penalty_cap or float("inf"))
        total += pen1
        detail_rules.append({"rule": "不接蔬菜", "violations": veg_orders, "penalty": round(pen1, 2), "preference_text": r1.content})

        viol_days = 0
        for day in days:
            intervals = _wait_intervals_for_day(ctxs, day)
            if _longest_merged_span_minutes(intervals) < 4 * 60:
                viol_days += 1
        pen2 = min(viol_days * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "每日连续停车歇脚≥4小时", "violations": viol_days, "penalty": round(pen2, 2), "preference_text": r2.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD003PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D003"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2 = rules[0], rules[1], rules[2]

        deadhead = _sum_deadhead_km(ctxs)
        over = max(0.0, deadhead - 100.0)
        pen0 = min(over * r0.penalty_amount, r0.penalty_cap or float("inf"))
        total += pen0
        detail_rules.append({"rule": "月度空驶赶路≤100km（超额按公里计）", "deadhead_km": round(deadhead, 2), "over_km": round(over, 2), "penalty": round(pen0, 2), "preference_text": r0.content})

        zone_violations = 0
        clat, clng, r_km = D003_FORBIDDEN_ZONE_LAT, D003_FORBIDDEN_ZONE_LNG, D003_FORBIDDEN_ZONE_RADIUS_KM
        for c in ctxs:
            if haversine_km(c["before_lat"], c["before_lng"], clat, clng) <= r_km or haversine_km(c["after_lat"], c["after_lng"], clat, clng) <= r_km:
                zone_violations += 1
        pen1 = min(zone_violations * r1.penalty_amount, r1.penalty_cap or float("inf"))
        total += pen1
        detail_rules.append(
            {"rule": "禁入圆区(23.30,113.52)半径20km", "violations": zone_violations, "penalty": round(pen1, 2), "preference_text": r1.content}
        )

        viol_days: set[int] = set()
        for c in ctxs:
            if c["action_name"] not in {"take_order", "reposition"}:
                continue
            a_start, a_end = c["action_start"], c["action_end"]
            for day in days:
                w1s = day * 1440 + 2 * 60
                w1e = day * 1440 + 5 * 60
                if _interval_overlap(a_start, a_end, w1s, w1e):
                    viol_days.add(day)
        pen2 = min(len(viol_days) * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "凌晨2–5点不接单不空驶", "violations": len(viol_days), "penalty": round(pen2, 2), "preference_text": r2.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD004PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D004"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2 = rules[0], rules[1], rules[2]

        late_first = 0
        for day in days:
            first_start: int | None = None
            for c in ctxs:
                if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                    continue
                if c["action_start"] // 1440 != day:
                    continue
                if first_start is None or c["action_start"] < first_start:
                    first_start = c["action_start"]
            if first_start is not None and first_start >= day * 1440 + 12 * 60:
                late_first += 1
        pen0 = min(late_first * r0.penalty_amount, r0.penalty_cap or float("inf"))
        total += pen0
        detail_rules.append({"rule": "有接单则首单开工不晚于当日12:00", "violations": late_first, "penalty": round(pen0, 2), "preference_text": r0.content})

        extra_orders_pen = 0.0
        for day in days:
            cnt = 0
            for c in ctxs:
                if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                    continue
                if c["action_start"] // 1440 != day:
                    continue
                cnt += 1
            if cnt > 3:
                extra_orders_pen += (cnt - 3) * r1.penalty_amount
        total += extra_orders_pen
        detail_rules.append({"rule": "同日接单≤3单", "penalty": round(extra_orders_pen, 2), "preference_text": r1.content})

        lunch_viol = 0
        for c in ctxs:
            if c["action_name"] not in {"take_order", "reposition"}:
                continue
            for day in days:
                ls = day * 1440 + 12 * 60
                le = day * 1440 + 13 * 60
                if _interval_overlap(c["action_start"], c["action_end"], ls, le):
                    lunch_viol += 1
                    break
        pen2 = min(lunch_viol * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "12–13点不接单不空驶", "violations": lunch_viol, "penalty": round(pen2, 2), "preference_text": r2.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD005PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D005"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2 = rules[0], rules[1], rules[2]

        haul_bad = 0
        pickup_bad = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            cg = cargo_map.get(cid)
            if cg is None:
                continue
            haul = float(cg["distance_km"])
            if haul > 100:
                haul_bad += 1
            pk = haversine_km(c["before_lat"], c["before_lng"], float(cg["start_lat"]), float(cg["start_lng"]))
            if pk > 90:
                pickup_bad += 1
        pen0 = haul_bad * r0.penalty_amount
        if r0.penalty_cap is not None:
            pen0 = min(pen0, r0.penalty_cap)
        pen1 = pickup_bad * r1.penalty_amount
        if r1.penalty_cap is not None:
            pen1 = min(pen1, r1.penalty_cap)
        total += pen0 + pen1
        detail_rules.append({"rule": "单笔装卸距离≤100km", "violations": haul_bad, "penalty": round(pen0, 2), "preference_text": r0.content})
        detail_rules.append({"rule": "赴装货点空驶≤90km", "violations": pickup_bad, "penalty": round(pen1, 2), "preference_text": r1.content})

        viol_days: set[int] = set()
        for c in ctxs:
            if c["action_name"] not in {"take_order", "reposition"}:
                continue
            for day in days:
                w1s, w1e, w2s, w2e = _night_windows_23_to_6(day)
                if _interval_overlap(c["action_start"], c["action_end"], w1s, w1e) or _interval_overlap(
                    c["action_start"], c["action_end"], w2s, w2e
                ):
                    viol_days.add(day)
        pen2 = min(len(viol_days) * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "每日23–次日6不接单不空驶", "violations": len(viol_days), "penalty": round(pen2, 2), "preference_text": r2.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD006PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D006"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2, r3 = rules[0], rules[1], rules[2], rules[3]

        viol_days = 0
        for day in days:
            intervals = _wait_intervals_for_day(ctxs, day)
            if _longest_merged_span_minutes(intervals) < 5 * 60:
                viol_days += 1
        pen0 = min(viol_days * r0.penalty_amount, r0.penalty_cap or float("inf"))
        total += pen0
        detail_rules.append({"rule": "每日连续停车休息≥5小时", "violations": viol_days, "penalty": round(pen0, 2), "preference_text": r0.content})

        fish_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            if str(cargo_map.get(cid, {}).get("cargo_name", "") or "") == "鲜活水产品":
                fish_orders += 1
        pen1 = min(fish_orders * r1.penalty_amount, r1.penalty_cap or float("inf"))
        total += pen1
        detail_rules.append({"rule": "不接鲜活水产品", "violations": fish_orders, "penalty": round(pen1, 2), "preference_text": r1.content})

        haul_bad = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            cg = cargo_map.get(cid)
            if cg is None:
                continue
            if float(cg["distance_km"]) > 150:
                haul_bad += 1
        pen2 = min(haul_bad * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "单笔装卸距离≤150km", "violations": haul_bad, "penalty": round(pen2, 2), "preference_text": r2.content})

        active = _active_minutes_by_day(ctxs, days)
        off_days = sum(1 for d in days if active.get(d, 0) == 0)
        pen3 = 0.0 if off_days >= 2 else (r3.penalty_cap or r3.penalty_amount)
        total += pen3
        detail_rules.append({"rule": "每月至少2整天不接单且不外跑", "off_days": off_days, "penalty": round(pen3, 2), "preference_text": r3.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD007PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D007"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2, r3 = rules[0], rules[1], rules[2], rules[3]

        viol_days_set: set[int] = set()
        for c in ctxs:
            if c["action_name"] not in {"take_order", "reposition"}:
                continue
            for day in days:
                w1s = day * 1440 + 23 * 60
                w1e = (day + 1) * 1440
                w2s = (day + 1) * 1440
                w2e = (day + 1) * 1440 + 4 * 60
                if _interval_overlap(c["action_start"], c["action_end"], w1s, w1e) or _interval_overlap(
                    c["action_start"], c["action_end"], w2s, w2e
                ):
                    viol_days_set.add(day)
        pen0 = min(len(viol_days_set) * r0.penalty_amount, r0.penalty_cap or float("inf"))
        total += pen0
        detail_rules.append({"rule": "每日23–次日4不接单不空驶", "violations": len(viol_days_set), "penalty": round(pen0, 2), "preference_text": r0.content})

        me_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            if str(cargo_map.get(cid, {}).get("cargo_name", "") or "") == "机械设备":
                me_orders += 1
        pen1 = min(me_orders * r1.penalty_amount, r1.penalty_cap or float("inf"))
        total += pen1
        detail_rules.append({"rule": "不接机械设备", "violations": me_orders, "penalty": round(pen1, 2), "preference_text": r1.content})

        haul_bad = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            cg = cargo_map.get(cid)
            if cg is None:
                continue
            if float(cg["distance_km"]) > 180:
                haul_bad += 1
        pen2 = min(haul_bad * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "单笔装卸距离≤180km", "violations": haul_bad, "penalty": round(pen2, 2), "preference_text": r2.content})

        free_full_days = 0
        for day in days:
            ordered = False
            for c in ctxs:
                if c["action_name"] != "take_order":
                    continue
                if c["step_end"] // 1440 != day:
                    continue
                if bool(c["result"].get("accepted", False)):
                    ordered = True
                    break
            if not ordered:
                free_full_days += 1
        pen3 = 0.0 if free_full_days >= 1 else (r3.penalty_cap or r3.penalty_amount)
        total += pen3
        detail_rules.append({"rule": "自然月至少放空一整天不接单", "free_days": free_full_days, "penalty": round(pen3, 2), "preference_text": r3.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD008PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D008"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        r0, r1, r2, r3 = rules[0], rules[1], rules[2], rules[3]

        active = _active_minutes_by_day(ctxs, days)
        off_days = sum(1 for d in days if active.get(d, 0) == 0)
        pen0 = 0.0 if off_days >= 2 else (r0.penalty_cap or r0.penalty_amount)
        total += pen0
        detail_rules.append({"rule": "自然月至少2天完全歇着", "off_days": off_days, "penalty": round(pen0, 2), "preference_text": r0.content})

        weekday_viol = 0
        for day in days:
            if _calendar_weekday_202603(day) >= 5:
                continue
            intervals = _wait_intervals_for_day(ctxs, day)
            if _longest_merged_span_minutes(intervals) < 4 * 60:
                weekday_viol += 1
        pen1 = min(weekday_viol * r1.penalty_amount, r1.penalty_cap or float("inf"))
        total += pen1
        detail_rules.append({"rule": "平日连续停车休息≥4小时", "violations": weekday_viol, "penalty": round(pen1, 2), "preference_text": r1.content})

        food_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            if str(cargo_map.get(cid, {}).get("cargo_name", "") or "") == "食品饮料":
                food_orders += 1
        pen2 = min(food_orders * r2.penalty_amount, r2.penalty_cap or float("inf"))
        total += pen2
        detail_rules.append({"rule": "尽量不拉食品饮料", "violations": food_orders, "penalty": round(pen2, 2), "preference_text": r2.content})

        pickup_bad = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            cg = cargo_map.get(cid)
            if cg is None:
                continue
            pk = haversine_km(c["before_lat"], c["before_lng"], float(cg["start_lat"]), float(cg["start_lng"]))
            if pk > 50:
                pickup_bad += 1
        pen3 = min(pickup_bad * r3.penalty_amount, r3.penalty_cap or float("inf"))
        total += pen3
        detail_rules.append({"rule": "赴装货点空驶≤50km", "violations": pickup_bad, "penalty": round(pen3, 2), "preference_text": r3.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD009PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D009"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))

        temp_rule = rules[0]
        took_temp = False
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            if cid != "240646":
                continue
            if _interval_overlap(temp_rule.start_minutes, temp_rule.end_minutes + 1, c["action_start"], c["action_end"]):
                took_temp = True
                break
        pen_temp = 0.0 if took_temp else min(temp_rule.penalty_amount, temp_rule.penalty_cap or float("inf"))
        total += pen_temp
        detail_rules.append({"rule": "临时约定熟货240646", "satisfied": took_temp, "penalty": round(pen_temp, 2), "preference_text": temp_rule.content})

        r_home = rules[1]
        home_lat, home_lng = 23.12, 113.28
        violation_days = 0
        for day in days:
            t23 = day * 1440 + 23 * 60
            t8_next = (day + 1) * 1440 + 8 * 60
            last_pos: tuple[float, float] | None = None
            for c in ctxs:
                if c["step_end"] <= t23:
                    last_pos = (c["after_lat"], c["after_lng"])
            if last_pos is None and ctxs:
                last_pos = (ctxs[0]["before_lat"], ctxs[0]["before_lng"])
            home_ok = last_pos is not None and haversine_km(last_pos[0], last_pos[1], home_lat, home_lng) <= 1.0
            quiet_ok = True
            for c in ctxs:
                if c["action_name"] in {"take_order", "reposition"} and _interval_overlap(c["action_start"], c["action_end"], t23, t8_next):
                    quiet_ok = False
                    break
            if not (home_ok and quiet_ok):
                violation_days += 1
        pen_home = min(violation_days * r_home.penalty_amount, r_home.penalty_cap or float("inf"))
        total += pen_home
        detail_rules.append(
            {"rule": "每日23点前到家且夜间不接单不空驶", "violations": violation_days, "penalty": round(pen_home, 2), "preference_text": r_home.content}
        )

        r_express = rules[2]
        express_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            cg = cargo_map.get(cid)
            if cg is None:
                continue
            if str(cg.get("cargo_name", "") or "") == "快递快运搬家":
                express_orders += 1
        pen_e = min(express_orders * r_express.penalty_amount, r_express.penalty_cap or float("inf"))
        total += pen_e
        detail_rules.append({"rule": "不接快递快运搬家", "violations": express_orders, "penalty": round(pen_e, 2), "preference_text": r_express.content})

        return round(total, 2), {"rules": detail_rules}


class DriverD010PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D010"

    def compute(
        self,
        ctxs: list[dict[str, Any]],
        cargo_map: dict[str, dict[str, Any]],
        rules: list[PreferenceRuleSpec],
        simulation_duration_days: int,
    ) -> tuple[float, dict[str, Any]]:
        detail_rules: list[dict[str, Any]] = []
        total = 0.0
        days = list(range(simulation_duration_days))
        family_rule = rules[0]

        pen_family, family_detail = self._evaluate_family_event(ctxs, family_rule)
        total += pen_family
        detail_rules.append({**family_detail, "preference_text": family_rule.content})

        r_visit = rules[1]
        visit_days: set[int] = set()
        for c in ctxs:
            if haversine_km(c["after_lat"], c["after_lng"], 23.13, 113.26) <= 1.0:
                visit_days.add(c["step_end"] // 1440)
        pen_visit = 0.0 if len(visit_days) >= 5 else (r_visit.penalty_cap or r_visit.penalty_amount)
        total += pen_visit
        detail_rules.append({"rule": "月度至少5日到访指定点半径1km", "visit_days": len(visit_days), "penalty": round(pen_visit, 2), "preference_text": r_visit.content})

        r_rest = rules[2]
        viol_rest = 0
        for day in days:
            intervals = _wait_intervals_for_day(ctxs, day)
            if _longest_merged_span_minutes(intervals) < 3 * 60:
                viol_rest += 1
        pen_rest = min(viol_rest * r_rest.penalty_amount, r_rest.penalty_cap or float("inf"))
        total += pen_rest
        detail_rules.append({"rule": "每日连续停车休息≥3小时", "violations": viol_rest, "penalty": round(pen_rest, 2), "preference_text": r_rest.content})

        r_soft = rules[3]
        cloth_orders = 0
        for c in ctxs:
            if c["action_name"] != "take_order" or not bool(c["result"].get("accepted", False)):
                continue
            cid = str((c["params"] or {}).get("cargo_id", "")).strip()
            if str(cargo_map.get(cid, {}).get("cargo_name", "") or "") == "服饰纺织皮革":
                cloth_orders += 1
        pen_soft = min(cloth_orders * r_soft.penalty_amount, r_soft.penalty_cap or float("inf"))
        total += pen_soft
        detail_rules.append({"rule": "尽量不拉服饰纺织皮革", "violations": cloth_orders, "penalty": round(pen_soft, 2), "preference_text": r_soft.content})

        return round(total, 2), {"rules": detail_rules}

    def _evaluate_family_event(self, ctxs: list[dict[str, Any]], rule: PreferenceRuleSpec) -> tuple[float, dict[str, Any]]:
        """家事：72h 窗内未在家按分钟×5 元；未接配偶或提前离家（含全程未返抵老家）满足任一项仅扣一次 9000。"""
        radius_km = 1.0
        w0, w1 = MAR10_10_MIN, D010_HOME_WINDOW_END_MIN

        def near_pick(lat: float, lng: float) -> bool:
            return haversine_km(lat, lng, D010_PICKUP_LAT, D010_PICKUP_LNG) <= radius_km

        def near_home(lat: float, lng: float) -> bool:
            return haversine_km(lat, lng, D010_HOME_LAT, D010_HOME_LNG) <= radius_km

        pickup_done_time: int | None = None
        pickup_run = 0
        for c in ctxs:
            if c["step_end"] <= MAR10_10_MIN:
                continue
            if c["action_name"] == "wait":
                if near_pick(c["after_lat"], c["after_lng"]):
                    pickup_run += c["action_exec_cost"]
                    if pickup_run >= 10 and pickup_done_time is None:
                        pickup_done_time = c["step_end"]
                else:
                    pickup_run = 0
            elif not (near_pick(c["before_lat"], c["before_lng"]) and near_pick(c["after_lat"], c["after_lng"])):
                pickup_run = 0

        first_home_arrival: int | None = None
        for c in ctxs:
            if c["step_end"] < MAR10_10_MIN:
                continue
            if near_home(c["after_lat"], c["after_lng"]):
                first_home_arrival = c["step_end"]
                break

        sequence_ok = (
            pickup_done_time is not None
            and first_home_arrival is not None
            and pickup_done_time < first_home_arrival
        )

        minutes_not_home = 0
        for c in ctxs:
            seg_a = max(c["step_start"], w0)
            seg_b = min(c["step_end"], w1)
            if seg_b <= seg_a:
                continue
            at_home = (
                c["action_name"] == "wait"
                and near_home(c["before_lat"], c["before_lng"])
                and near_home(c["after_lat"], c["after_lng"])
            )
            if not at_home:
                minutes_not_home += seg_b - seg_a

        pen_absence = float(minutes_not_home) * 5.0

        never_arrived_home = first_home_arrival is None
        left_after_arrival = False
        if first_home_arrival is not None:
            for c in ctxs:
                if c["step_start"] < first_home_arrival:
                    continue
                if c["step_start"] >= MAR13_22_MIN:
                    break
                if c["action_name"] in {"take_order", "reposition"}:
                    left_after_arrival = True
                    break

        # 提前离家：全程未返抵老家亦视为提前离家（与「到家后又出车」并列）
        early_leave = never_arrived_home or left_after_arrival

        pen_fixed_9000 = (
            float(rule.penalty_amount) if (not sequence_ok or early_leave) else 0.0
        )

        raw_total = pen_absence + pen_fixed_9000
        total = raw_total if rule.penalty_cap is None else min(raw_total, float(rule.penalty_cap))

        return total, {
            "rule": "家事临时约定(3/10–3/13)",
            "sequence_ok": sequence_ok,
            "home_window": [w0, w1],
            "minutes_not_home_in_window": minutes_not_home,
            "pickup_done_minute": pickup_done_time,
            "first_home_minute": first_home_arrival,
            "penalty_absence_minutes": round(pen_absence, 2),
            "triggers_spouse_incomplete": not sequence_ok,
            "never_arrived_home": never_arrived_home,
            "left_after_arrival": left_after_arrival,
            "triggers_early_leave": early_leave,
            "penalty_fixed_9000": round(pen_fixed_9000, 2),
            "penalty": round(total, 2),
        }


_PREFERENCE_CALCULATORS: dict[str, DriverPreferenceCalculatorBase] = {
    "D001": DriverD001PreferenceCalculator(),
    "D002": DriverD002PreferenceCalculator(),
    "D003": DriverD003PreferenceCalculator(),
    "D004": DriverD004PreferenceCalculator(),
    "D005": DriverD005PreferenceCalculator(),
    "D006": DriverD006PreferenceCalculator(),
    "D007": DriverD007PreferenceCalculator(),
    "D008": DriverD008PreferenceCalculator(),
    "D009": DriverD009PreferenceCalculator(),
    "D010": DriverD010PreferenceCalculator(),
}

_MIN_PREFERENCE_RULES_PER_DRIVER: dict[str, int] = {
    "D001": 3,
    "D002": 3,
    "D003": 3,
    "D004": 3,
    "D005": 3,
    "D006": 4,
    "D007": 4,
    "D008": 4,
    "D009": 3,
    "D010": 4,
}


def _evaluate_preferences(
    driver_id: str,
    file_path: Path,
    rules: list[PreferenceRuleSpec],
    cargo_map: dict[str, dict[str, Any]],
    simulation_duration_days: int,
) -> tuple[float, dict[str, Any]]:
    calc = _PREFERENCE_CALCULATORS.get(driver_id)
    if calc is None:
        return 0.0, {"rules": []}
    need = _MIN_PREFERENCE_RULES_PER_DRIVER.get(driver_id, 0)
    if need and len(rules) < need:
        raise ValueError(f"{driver_id} 的 preferences 在 drivers.json 中至少需要 {need} 条，当前 {len(rules)}")
    ctxs = _build_step_contexts(file_path)
    if not ctxs:
        return 0.0, {"rules": []}
    return calc.compute(ctxs, cargo_map, rules, simulation_duration_days)


def _validate_and_compute_income_by_driver(
    file_path: Path,
    cargo_map: dict[str, dict[str, Any]],
    cost_per_km: float,
    reposition_speed_km_per_hour: float,
    simulation_horizon_minutes: int | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    prev_end_minutes = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record: dict[str, Any] = json.loads(row)
            action = record.get("action", {})
            params = action.get("params", {})
            result = record.get("result", {})
            action_name = str(action.get("action", "")).strip().lower()
            if action_name not in {"wait", "reposition", "take_order"}:
                raise ValueError(f"{file_path.name} 第 {line_no} 行 action 非法: {action_name}")

            raw_usage = record.get("token_usage", {})
            if not isinstance(raw_usage, dict):
                raise ValueError(f"{file_path.name} 第 {line_no} 行 token_usage 非法")
            token_usage["prompt_tokens"] += int(raw_usage.get("prompt_tokens", 0))
            token_usage["completion_tokens"] += int(raw_usage.get("completion_tokens", 0))
            token_usage["reasoning_tokens"] += int(raw_usage.get("reasoning_tokens", 0))
            token_usage["total_tokens"] += int(raw_usage.get("total_tokens", 0))

            end_minutes = int(result.get("simulation_progress_minutes", -1))
            if end_minutes < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少 simulation_progress_minutes")
            step_elapsed = int(record.get("step_elapsed_minutes", -1))
            if step_elapsed < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少 step_elapsed_minutes")
            query_scan_cost = int(record.get("query_scan_cost_minutes", -1))
            action_exec_cost = int(record.get("action_exec_cost_minutes", -1))
            if query_scan_cost < 0 or action_exec_cost < 0:
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少 query/action cost 字段")
            if step_elapsed != query_scan_cost + action_exec_cost:
                raise ValueError(f"{file_path.name} 第 {line_no} 行耗时不一致")
            if end_minutes - prev_end_minutes != step_elapsed:
                raise ValueError(f"{file_path.name} 第 {line_no} 行时间推进不一致")
            step_start_minutes = prev_end_minutes
            action_start_minutes = step_start_minutes + query_scan_cost

            pos_before = record.get("position_before", {})
            pos_after = record.get("position_after", {})
            if not isinstance(pos_before, dict) or not isinstance(pos_after, dict):
                raise ValueError(f"{file_path.name} 第 {line_no} 行缺少位置字段")
            before_lat = float(pos_before.get("lat"))
            before_lng = float(pos_before.get("lng"))
            after_lat = float(pos_after.get("lat"))
            after_lng = float(pos_after.get("lng"))

            if action_name == "wait":
                wait_minutes = int((params or {}).get("duration_minutes", 1))
                if action_exec_cost != wait_minutes:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 时间不一致")
                if (not _nearly_equal(before_lat, after_lat)) or (not _nearly_equal(before_lng, after_lng)):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 不应改变位置")

            elif action_name == "reposition":
                target_lat = float((params or {}).get("latitude"))
                target_lng = float((params or {}).get("longitude"))
                if (not _nearly_equal(after_lat, target_lat)) or (not _nearly_equal(after_lng, target_lng)):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 reposition 终点位置错误")
                expected_km = haversine_km(before_lat, before_lng, target_lat, target_lng)
                expected_minutes = _distance_minutes(expected_km, reposition_speed_km_per_hour)
                if action_exec_cost != expected_minutes:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 reposition 时间不一致")
                income["distance_km"] += float(result.get("distance_km", 0.0))

            elif action_name == "take_order":
                cargo_id = str((params or {}).get("cargo_id", "")).strip()
                if not cargo_id:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 take_order 缺少 cargo_id")
                cargo = cargo_map.get(cargo_id)
                if cargo is None:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 cargo_id 不存在: {cargo_id}")
                if not (int(cargo["create_minutes"]) <= action_start_minutes <= int(cargo["remove_minutes"])):
                    if bool(result.get("accepted", False)):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单时点不在货源有效期")

                accepted = bool(result.get("accepted", False))
                if accepted:
                    if (not _nearly_equal(after_lat, float(cargo["end_lat"]))) or (
                        not _nearly_equal(after_lng, float(cargo["end_lng"]))
                    ):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单后位置错误")
                    pickup_km = haversine_km(before_lat, before_lng, float(cargo["start_lat"]), float(cargo["start_lng"]))
                    pickup_minutes = _distance_minutes(pickup_km, reposition_speed_km_per_hour) if pickup_km > 1e-6 else 0
                    arrival_minutes = action_start_minutes + pickup_minutes
                    wait_minutes = 0
                    load_start_minutes = cargo.get("load_start_minutes")
                    load_end_minutes = cargo.get("load_end_minutes")
                    if isinstance(load_start_minutes, int) and isinstance(load_end_minutes, int):
                        if arrival_minutes > load_end_minutes:
                            raise ValueError(f"{file_path.name} 第 {line_no} 行成功接单但已超装货时间窗")
                        wait_minutes = max(0, load_start_minutes - arrival_minutes)
                    expected_exec = pickup_minutes + wait_minutes + int(cargo["cost_time_minutes"])
                    if action_exec_cost != expected_exec:
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单耗时不一致")
                    income_eligible = simulation_horizon_minutes is None or int(end_minutes) <= int(simulation_horizon_minutes)
                    if income_eligible:
                        income["gross_income"] += float(cargo["price"])
                    income["distance_km"] += float(result.get("pickup_deadhead_km", 0.0) or 0.0)
                    haul_km = float(result.get("haul_distance_km", 0.0) or 0.0)
                    if haul_km <= 0:
                        haul_km = float(cargo["distance_km"])
                    income["distance_km"] += haul_km

            prev_end_minutes = end_minutes

    income["cost"] = income["distance_km"] * cost_per_km
    income["net_income"] = income["gross_income"] - income["cost"]
    for key in ("gross_income", "distance_km", "cost", "net_income"):
        income[key] = round(float(income[key]), 2)
    return income, token_usage


def compute_income(
    files: list[Path],
    cargo_map: dict[str, dict[str, Any]],
    driver_cost_map: dict[str, float],
    driver_preference_rules: dict[str, list[PreferenceRuleSpec]],
    reposition_speed_km_per_hour: float,
    simulation_duration_days: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, int], dict[str, str], dict[str, dict[str, Any]]]:
    stats: dict[str, dict[str, float]] = {}
    token_stats: dict[str, dict[str, int]] = {}
    validation_errors: dict[str, str] = {}
    preference_details_by_driver: dict[str, dict[str, Any]] = {}
    zero_income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    simulation_horizon_minutes = int(simulation_duration_days) * 24 * 60
    for file_path in files:
        driver_id = file_path.name.split("_")[2]
        cost_per_km = float(driver_cost_map.get(driver_id, 0.0))
        try:
            income_item, token_item = _validate_and_compute_income_by_driver(
                file_path,
                cargo_map,
                cost_per_km=cost_per_km,
                reposition_speed_km_per_hour=reposition_speed_km_per_hour,
                simulation_horizon_minutes=simulation_horizon_minutes,
            )
            preference_penalty, preference_details = _evaluate_preferences(
                driver_id,
                file_path,
                driver_preference_rules.get(driver_id, []),
                cargo_map,
                simulation_duration_days=simulation_duration_days,
            )
            income_item["preference_penalty"] = round(float(preference_penalty), 2)
            income_item["net_income"] = round(float(income_item["net_income"] - preference_penalty), 2)
            preference_details_by_driver[driver_id] = preference_details
            stats[driver_id] = income_item
            token_stats[driver_id] = token_item
        except Exception as exc:
            stats[driver_id] = dict(zero_income)
            token_stats[driver_id] = dict(zero_tokens)
            validation_errors[driver_id] = f"{type(exc).__name__}: {exc}"
            preference_details_by_driver[driver_id] = {"rules": []}
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    for token_item in token_stats.values():
        total_token_usage["prompt_tokens"] += int(token_item["prompt_tokens"])
        total_token_usage["completion_tokens"] += int(token_item["completion_tokens"])
        total_token_usage["reasoning_tokens"] += int(token_item["reasoning_tokens"])
        total_token_usage["total_tokens"] += int(token_item["total_tokens"])
    return stats, token_stats, total_token_usage, validation_errors, preference_details_by_driver


def build_drivers_payload(
    income: dict[str, dict[str, float]],
    token_by_driver: dict[str, dict[str, int]],
    validation_errors: dict[str, str],
    preference_details_by_driver: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """按司机合并收入与 Token，输出稳定排序的列表。"""
    default_income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "preference_penalty": 0.0, "net_income": 0.0}
    default_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    driver_ids = sorted(set(income) | set(token_by_driver))
    rows: list[dict[str, Any]] = []
    for driver_id in driver_ids:
        inc = {**default_income, **income.get(driver_id, {})}
        tok = {**default_tokens, **token_by_driver.get(driver_id, {})}
        rows.append(
            {
                "driver_id": driver_id,
                "income": inc,
                "token_usage": tok,
                "calculation_aborted": driver_id in validation_errors,
                "validation_error": validation_errors.get(driver_id),
                "preference_check": preference_details_by_driver.get(driver_id, {"rules": []}),
            }
        )
    return rows


def main(project_root: Path | None = None, results_dir: Path | None = None) -> None:
    layout_root = (project_root if project_root is not None else _SCRIPT_DIR).resolve()
    results_dir_path = (results_dir.resolve() if results_dir is not None else layout_root / "results").resolve()
    server_root = layout_root / "server"
    cargo_dataset = server_root / "data" / "cargo_dataset.jsonl"
    drivers_dataset = server_root / "data" / "drivers.json"
    config_path = _resolve_config_json(server_root / "config")
    output_file = results_dir_path / "monthly_income_202603.json"
    run_summary_file = results_dir_path / "run_summary_202603.json"

    if not cargo_dataset.is_file():
        raise FileNotFoundError(f"缺少货源数据: {cargo_dataset}")
    if not drivers_dataset.is_file():
        raise FileNotFoundError(f"缺少司机数据: {drivers_dataset}")
    results_dir_path.mkdir(parents=True, exist_ok=True)
    cargo_map = load_cargo_map(cargo_dataset)
    driver_cost_map = load_driver_cost_map(drivers_dataset)
    driver_preference_rules = load_driver_preference_rules(drivers_dataset)
    reposition_speed_km_per_hour = load_reposition_speed_km_per_hour(config_path)
    simulation_duration_days = load_simulation_duration_days(run_summary_file)
    result_files = iter_result_files(results_dir_path)
    income, token_by_driver, total_token_usage, validation_errors, preference_details_by_driver = compute_income(
        result_files,
        cargo_map,
        driver_cost_map,
        driver_preference_rules,
        reposition_speed_km_per_hour=reposition_speed_km_per_hour,
        simulation_duration_days=simulation_duration_days,
    )
    drivers = build_drivers_payload(income, token_by_driver, validation_errors, preference_details_by_driver)
    total_net_income = round(sum(float(d["income"]["net_income"]) for d in drivers), 2)
    total_preference_penalty = round(sum(float(d["income"].get("preference_penalty", 0.0)) for d in drivers), 2)
    simulate_time_seconds = load_simulate_time_seconds(run_summary_file)
    payload = {
        "month": "2026-03",
        "simulate_time_seconds": simulate_time_seconds,
        "result_files_count": len(result_files),
        "drivers": drivers,
        "summary": {
            "total_net_income_all_drivers": total_net_income,
            "total_preference_penalty": total_preference_penalty,
            "total_token_usage": total_token_usage,
            "failed_driver_count": len(validation_errors),
            "failed_drivers": validation_errors,
        },
        "cost_meaning": "cost = distance_km * cost_per_km (driver cost per km)",
        "cost_metric": "net_income = gross_income - (distance_km * cost_per_km)",
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 2026 年 3 月司机累计收益（读取 config 空驶速度）")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="含 server/data 的布局根目录；默认脚本所在目录（其下需有 server/data）",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="仿真结果目录（含 actions_202603_*.jsonl、run_summary_202603.json）；默认 <project-root>/results",
    )
    args = parser.parse_args()
    main(project_root=args.project_root, results_dir=args.results_dir)
