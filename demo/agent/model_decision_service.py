from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from simkit.ports import SimulationApiPort


SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
DEFAULT_COST_PER_KM = 1.5
REPOSITION_SPEED_KMPH = 60.0
SAFE_WAIT_MINUTES = 30

HUBS: tuple[tuple[str, float, float], ...] = (
    ("广州白云", 23.30, 113.34),
    ("佛山南海", 23.05, 113.14),
    ("佛山顺德", 22.84, 113.25),
    ("深圳龙岗", 22.72, 114.25),
    ("深圳宝安", 22.68, 113.88),
    ("惠州惠城", 23.09, 114.40),
    ("东莞厚街", 22.94, 113.67),
    ("东莞常平", 22.98, 114.00),
)

CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass
class PreferenceConstraints:
    forbidden_cargo_names: set[str] = field(default_factory=set)
    soft_avoid_cargo_names: set[str] = field(default_factory=set)
    daily_rest_minutes: int | None = None
    monthly_off_days: int | None = None
    no_drive_windows: list[tuple[int, int]] = field(default_factory=list)
    lunch_no_drive_windows: list[tuple[int, int]] = field(default_factory=list)
    max_pickup_deadhead_km: float | None = None
    max_order_distance_km: float | None = None
    max_monthly_deadhead_km: float | None = None
    max_daily_orders: int | None = None
    first_order_before_minute: int | None = None
    must_stay_in_bbox: dict[str, float] | None = None
    forbidden_circle: dict[str, float] | None = None
    must_take_cargo_id: str | None = None
    must_take_cargo: dict[str, Any] | None = None
    home_location: dict[str, float] | None = None
    special_home_task: dict[str, Any] | None = None
    must_visit_locations: list[dict[str, Any]] = field(default_factory=list)
    llm_parse: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriverMemory:
    last_seen_time: int | None = None
    last_position: tuple[float, float] | None = None
    daily_order_count: dict[int, int] = field(default_factory=dict)
    daily_first_work_time: dict[int, int] = field(default_factory=dict)
    daily_rest_segments: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    monthly_off_days_candidate: set[int] = field(default_factory=set)
    estimated_deadhead_km: float = 0.0
    bad_query_count: int = 0
    no_good_cargo_count: int = 0
    last_hub_reposition_time: int | None = None
    parsed_preferences_cache: tuple[str, PreferenceConstraints] | None = None
    active_minutes_by_day: dict[int, int] = field(default_factory=dict)
    planned_visit_days: dict[str, set[int]] = field(default_factory=dict)


@dataclass
class EvaluatedCargo:
    cargo_id: str
    score: float
    expected_profit: float
    profit_per_hour: float
    finish_minute: int
    pickup_deadhead_km: float
    order_distance_km: float


class ModelDecisionService:
    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._strategy = DeterministicStrategy(api, self._logger)

    def decide(self, driver_id: str) -> dict[str, Any]:
        try:
            action = self._strategy.decide(driver_id)
            return validate_action(action)
        except Exception:
            self._logger.exception("deterministic strategy failed, fallback to wait driver_id=%s", driver_id)
            return {"action": "wait", "params": {"duration_minutes": SAFE_WAIT_MINUTES}}


class DeterministicStrategy:
    def __init__(self, api: SimulationApiPort, logger: logging.Logger) -> None:
        self._api = api
        self._logger = logger
        self._memories: dict[str, DriverMemory] = {}
        self._parser = LLMPreferenceParser(api, logger)
        self._evaluator = CandidateEvaluator()

    def decide(self, driver_id: str) -> dict[str, Any]:
        memory = self._memories.setdefault(driver_id, DriverMemory())
        status = self._safe_status(driver_id)
        self._refresh_memory(driver_id, memory, status)
        constraints = self._parse_preferences(memory, status)
        now_min = parse_sim_minutes(status)
        lat, lng = read_position(status)
        memory.last_seen_time = now_min
        memory.last_position = (lat, lng)

        action = self._plan_special_home_task(now_min, lat, lng, constraints, memory)
        if action is not None:
            return action

        action = self._plan_home_return(now_min, lat, lng, constraints)
        if action is not None:
            return action

        action = self._plan_must_take_cargo(now_min, lat, lng, constraints)
        if action is not None:
            return action

        wait_until = current_no_drive_window_end(now_min, all_no_drive_windows(constraints))
        if wait_until is not None:
            return wait_action(wait_until - now_min, max_minutes=540)

        action = self._plan_monthly_off_day(now_min, memory, constraints)
        if action is not None:
            return action

        action = self._plan_daily_rest_before_query(now_min, memory, constraints)
        if action is not None:
            return action

        action = self._plan_must_visit_locations(now_min, lat, lng, constraints, memory)
        if action is not None:
            return action

        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
        items = cargo_resp.get("items", [])
        if not isinstance(items, list):
            items = []
        after_query_status = self._safe_status(driver_id)
        after_query_min = parse_sim_minutes(after_query_status)
        after_query_lat, after_query_lng = read_position(after_query_status)

        best, must_take = self._evaluator.choose_best(
            items=items,
            status=after_query_status,
            constraints=constraints,
            memory=memory,
        )
        if must_take is not None:
            memory.no_good_cargo_count = 0
            return {"action": "take_order", "params": {"cargo_id": must_take.cargo_id}}
        if best is not None and best.expected_profit > 0 and best.score > 0:
            memory.no_good_cargo_count = 0
            return {"action": "take_order", "params": {"cargo_id": best.cargo_id}}

        memory.no_good_cargo_count += 1

        action = self._plan_daily_rest_after_query(after_query_min, memory, constraints)
        if action is not None:
            return action

        next_window_wait = upcoming_no_drive_wait(after_query_min, all_no_drive_windows(constraints), threshold_minutes=45)
        if next_window_wait is not None:
            return wait_action(next_window_wait, max_minutes=540)

        if self._is_low_value_wait_time(after_query_min):
            return wait_action(60)

        if memory.no_good_cargo_count >= 3:
            action = self._plan_hub_reposition(after_query_min, after_query_lat, after_query_lng, constraints, memory)
            if action is not None:
                memory.no_good_cargo_count = 0
                return action

        return wait_action(SAFE_WAIT_MINUTES)

    def _safe_status(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        return status if isinstance(status, dict) else {}

    def _parse_preferences(self, memory: DriverMemory, status: dict[str, Any]) -> PreferenceConstraints:
        raw_preferences = status.get("preferences") or []
        key = repr(raw_preferences)
        if memory.parsed_preferences_cache and memory.parsed_preferences_cache[0] == key:
            return memory.parsed_preferences_cache[1]
        constraints = self._parser.parse(raw_preferences)
        enrich_must_take_from_raw_preferences(raw_preferences, constraints)
        memory.parsed_preferences_cache = (key, constraints)
        return constraints

    def _refresh_memory(self, driver_id: str, memory: DriverMemory, status: dict[str, Any]) -> None:
        history = self._safe_history(driver_id)
        memory.daily_order_count.clear()
        memory.daily_first_work_time.clear()
        memory.daily_rest_segments.clear()
        memory.monthly_off_days_candidate.clear()
        memory.active_minutes_by_day.clear()
        memory.estimated_deadhead_km = 0.0

        all_days = set(range(parse_sim_minutes(status) // 1440))
        active_days: set[int] = set()
        order_days: set[int] = set()
        prev_end = 0
        for record in history:
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            action = record.get("action") if isinstance(record.get("action"), dict) else {}
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            action_name = str(action.get("action", "")).strip().lower()
            end_min = safe_int(result.get("simulation_progress_minutes"), prev_end)
            step_elapsed = safe_int(record.get("step_elapsed_minutes"), max(0, end_min - prev_end))
            query_cost = safe_int(record.get("query_scan_cost_minutes"), 0)
            action_cost = safe_int(record.get("action_exec_cost_minutes"), max(0, step_elapsed - query_cost))
            step_start = max(0, end_min - step_elapsed)
            action_start = step_start + query_cost
            action_end = action_start + action_cost

            if action_name == "wait" and action_end > action_start:
                for day, start, end in split_by_day(action_start, action_end):
                    memory.daily_rest_segments.setdefault(day, []).append((start, end))
            elif action_name in {"take_order", "reposition"}:
                for day, start, end in split_by_day(action_start, action_end):
                    active_days.add(day)
                    memory.active_minutes_by_day[day] = memory.active_minutes_by_day.get(day, 0) + (end - start)

            if action_name == "take_order" and bool(result.get("accepted", False)):
                day = action_start // 1440
                order_days.add(day)
                memory.daily_order_count[day] = memory.daily_order_count.get(day, 0) + 1
                memory.daily_first_work_time[day] = min(
                    memory.daily_first_work_time.get(day, action_start), action_start
                )
                memory.estimated_deadhead_km += safe_float(result.get("pickup_deadhead_km"), 0.0)
            elif action_name == "reposition":
                memory.estimated_deadhead_km += safe_float(result.get("distance_km"), 0.0)

            if action_name in {"take_order", "reposition"}:
                day = action_start // 1440
                memory.daily_first_work_time[day] = min(
                    memory.daily_first_work_time.get(day, action_start), action_start
                )

            _ = params
            prev_end = end_min

        for day in all_days:
            if day not in active_days and day not in order_days:
                memory.monthly_off_days_candidate.add(day)

    def _safe_history(self, driver_id: str) -> list[dict[str, Any]]:
        try:
            resp = self._api.query_decision_history(driver_id, -1)
            records = resp.get("records", []) if isinstance(resp, dict) else []
            return [r for r in records if isinstance(r, dict)]
        except Exception:
            self._logger.exception("query_decision_history failed driver_id=%s", driver_id)
            return []

    def _plan_special_home_task(
        self,
        now_min: int,
        lat: float,
        lng: float,
        constraints: PreferenceConstraints,
        memory: DriverMemory,
    ) -> dict[str, Any] | None:
        task = constraints.special_home_task
        if not task:
            return None
        start_min = int(task.get("start_minute", 0))
        stay_until = int(task.get("stay_until_minute", start_min))
        if now_min < start_min:
            if start_min - now_min <= 180:
                return wait_action(start_min - now_min)
            return None
        if now_min >= stay_until:
            return None

        pickup = task.get("pickup") or {}
        home = task.get("home") or {}
        pickup_lat = safe_float(pickup.get("lat"), lat)
        pickup_lng = safe_float(pickup.get("lng"), lng)
        home_lat = safe_float(home.get("lat"), lat)
        home_lng = safe_float(home.get("lng"), lng)
        pickup_done = self._pickup_wait_done(memory, pickup_lat, pickup_lng, start_min)
        at_pickup = haversine_km(lat, lng, pickup_lat, pickup_lng) <= 1.0
        at_home = haversine_km(lat, lng, home_lat, home_lng) <= 1.0

        if not pickup_done:
            if not at_pickup:
                return {"action": "reposition", "params": {"latitude": pickup_lat, "longitude": pickup_lng}}
            return wait_action(int(task.get("pickup_wait_minutes", 10)), min_minutes=10)
        if not at_home:
            return {"action": "reposition", "params": {"latitude": home_lat, "longitude": home_lng}}
        return wait_action(stay_until - now_min, max_minutes=480)

    def _pickup_wait_done(self, memory: DriverMemory, lat: float, lng: float, start_min: int) -> bool:
        for day, intervals in memory.daily_rest_segments.items():
            day_base = day * 1440
            for start, end in intervals:
                if day_base + end < start_min:
                    continue
                if end - start >= 10:
                    return True
        _ = lat, lng
        return False

    def _plan_home_return(
        self,
        now_min: int,
        lat: float,
        lng: float,
        constraints: PreferenceConstraints,
    ) -> dict[str, Any] | None:
        home = constraints.home_location
        if not home:
            return None
        home_lat = float(home["lat"])
        home_lng = float(home["lng"])
        radius = float(home.get("radius_km", 1.0))
        return_before = int(home.get("return_before_minute", 23 * 60))
        quiet_end = int(home.get("quiet_end_minute", 8 * 60))
        mod = now_min % 1440
        at_home = haversine_km(lat, lng, home_lat, home_lng) <= radius
        travel_minutes = distance_minutes(haversine_km(lat, lng, home_lat, home_lng))

        if mod >= return_before or mod < quiet_end:
            if not at_home:
                return {"action": "reposition", "params": {"latitude": home_lat, "longitude": home_lng}}
            end_abs = absolute_next_minute(now_min, quiet_end)
            return wait_action(end_abs - now_min, max_minutes=540)
        if not at_home and mod + travel_minutes >= return_before - 90:
            return {"action": "reposition", "params": {"latitude": home_lat, "longitude": home_lng}}
        return None

    def _plan_must_take_cargo(
        self,
        now_min: int,
        lat: float,
        lng: float,
        constraints: PreferenceConstraints,
    ) -> dict[str, Any] | None:
        task = constraints.must_take_cargo
        if not task:
            return None
        start = location_from_mapping(
            task,
            "start",
            "pickup",
            "origin",
            "location",
            "start_location",
            "pickup_location",
            "origin_location",
            "loading_location",
        )
        if start is None:
            return None
        start_lat, start_lng = start
        if not point_allowed(start_lat, start_lng, constraints):
            return None
        visible_from = minute_from_value(task.get("visible_from_minute") or task.get("create_minute"))
        visible_until = minute_from_value(task.get("visible_until_minute") or task.get("remove_minute") or task.get("deadline_minute"))
        if visible_until is not None and now_min > visible_until:
            return None

        distance = haversine_km(lat, lng, start_lat, start_lng)
        travel_minutes = distance_minutes(distance) if distance > 1.0 else 0
        if visible_from is None:
            if distance > 8.0:
                return {"action": "reposition", "params": {"latitude": start_lat, "longitude": start_lng}}
            return None

        latest_start = max(0, visible_from - travel_minutes - 20)
        if now_min < latest_start:
            if latest_start - now_min <= 300:
                return wait_action(latest_start - now_min)
            return None
        if distance > 3.0:
            return {"action": "reposition", "params": {"latitude": start_lat, "longitude": start_lng}}
        if now_min < visible_from:
            return wait_action(visible_from - now_min)
        return None

    def _plan_monthly_off_day(
        self,
        now_min: int,
        memory: DriverMemory,
        constraints: PreferenceConstraints,
    ) -> dict[str, Any] | None:
        needed = constraints.monthly_off_days
        if not needed or len(memory.monthly_off_days_candidate) >= needed:
            return None
        day = now_min // 1440
        mod = now_min % 1440
        active_today = memory.active_minutes_by_day.get(day, 0) > 0 or memory.daily_order_count.get(day, 0) > 0
        if active_today:
            return None
        if day <= needed + 1:
            return wait_action(1440 - mod, max_minutes=480)
        remaining_days = max(0, 30 - day)
        missing = needed - len(memory.monthly_off_days_candidate)
        if remaining_days <= missing:
            return wait_action(1440 - mod, max_minutes=480)
        return None

    def _plan_daily_rest_before_query(
        self,
        now_min: int,
        memory: DriverMemory,
        constraints: PreferenceConstraints,
    ) -> dict[str, Any] | None:
        need = constraints.daily_rest_minutes
        if not need or daily_rest_satisfied(now_min // 1440, memory, need):
            return None
        mod = now_min % 1440
        if mod >= max(0, 1440 - need) or mod < 5 * 60:
            return wait_action(need, max_minutes=480)
        return None

    def _plan_daily_rest_after_query(
        self,
        now_min: int,
        memory: DriverMemory,
        constraints: PreferenceConstraints,
    ) -> dict[str, Any] | None:
        need = constraints.daily_rest_minutes
        if not need or daily_rest_satisfied(now_min // 1440, memory, need):
            return None
        mod = now_min % 1440
        if mod >= max(18 * 60, 1440 - need) or memory.no_good_cargo_count >= 2:
            return wait_action(need, max_minutes=480)
        return None

    def _plan_must_visit_locations(
        self,
        now_min: int,
        lat: float,
        lng: float,
        constraints: PreferenceConstraints,
        memory: DriverMemory,
    ) -> dict[str, Any] | None:
        if not constraints.must_visit_locations:
            return None
        day = now_min // 1440
        if day >= 30:
            return None
        mod = now_min % 1440
        for index, location_rule in enumerate(constraints.must_visit_locations):
            target = lat_lng_from_value(location_rule)
            if target is None:
                continue
            target_lat, target_lng = target
            if not point_allowed(target_lat, target_lng, constraints):
                continue
            required_days = positive_int(
                location_rule.get("required_days")
                or location_rule.get("required_visit_days")
                or location_rule.get("min_days")
                or location_rule.get("min_visit_days")
                or location_rule.get("monthly_days")
                or location_rule.get("different_days")
                or location_rule.get("days")
            ) or 1
            key = str(location_rule.get("id") or f"visit-{index}-{target_lat:.4f}-{target_lng:.4f}")
            done_days = memory.planned_visit_days.setdefault(key, set())
            if len(done_days) >= required_days or day in done_days:
                continue
            distance = haversine_km(lat, lng, target_lat, target_lng)
            radius = positive_float(location_rule.get("radius_km")) or 1.0
            if distance <= radius:
                done_days.add(day)
                return wait_action(10, min_minutes=5, max_minutes=30)
            travel_minutes = distance_minutes(distance)
            if overlap_no_drive_window(now_min, now_min + travel_minutes, all_no_drive_windows(constraints)):
                continue
            if constraints.max_monthly_deadhead_km is not None:
                if memory.estimated_deadhead_km + distance > constraints.max_monthly_deadhead_km:
                    continue
            home = constraints.home_location
            if home and mod < int(home.get("return_before_minute", 23 * 60)):
                home_lat = float(home["lat"])
                home_lng = float(home["lng"])
                travel_home = distance_minutes(haversine_km(target_lat, target_lng, home_lat, home_lng))
                if mod + travel_minutes + travel_home > int(home.get("return_before_minute", 23 * 60)) - 30:
                    continue
            remaining_days = max(0, 30 - day)
            missing_days = required_days - len(done_days)
            urgent = remaining_days <= missing_days + 1 or day <= required_days + 2 or mod < 10 * 60 or distance <= 30
            if not urgent:
                continue
            if distance > 160 and remaining_days > missing_days + 1:
                continue
            done_days.add(day)
            return {"action": "reposition", "params": {"latitude": target_lat, "longitude": target_lng}}
        return None

    def _plan_hub_reposition(
        self,
        now_min: int,
        lat: float,
        lng: float,
        constraints: PreferenceConstraints,
        memory: DriverMemory,
    ) -> dict[str, Any] | None:
        if memory.last_hub_reposition_time is not None and now_min - memory.last_hub_reposition_time < 180:
            return None
        if current_no_drive_window_end(now_min, all_no_drive_windows(constraints)) is not None:
            return None
        best: tuple[float, float, float] | None = None
        for _, hlat, hlng in HUBS:
            if not point_allowed(hlat, hlng, constraints):
                continue
            if constraints.home_location and now_min % 1440 >= 18 * 60:
                home = constraints.home_location
                if haversine_km(hlat, hlng, float(home["lat"]), float(home["lng"])) > 60:
                    continue
            dist = haversine_km(lat, lng, hlat, hlng)
            if dist < 5 or dist > 80:
                continue
            if constraints.max_monthly_deadhead_km is not None:
                if memory.estimated_deadhead_km + dist > constraints.max_monthly_deadhead_km:
                    continue
            if best is None or dist < best[0]:
                best = (dist, hlat, hlng)
        if best is None:
            return None
        memory.last_hub_reposition_time = now_min
        return {"action": "reposition", "params": {"latitude": best[1], "longitude": best[2]}}

    @staticmethod
    def _is_low_value_wait_time(now_min: int) -> bool:
        mod = now_min % 1440
        return mod < 6 * 60 or mod >= 22 * 60


class PreferenceParser:
    def parse(self, preferences: Any) -> PreferenceConstraints:
        constraints = PreferenceConstraints()
        texts = self._preference_texts(preferences)
        for text in texts:
            self._parse_cargo_names(text, constraints)
            self._parse_rest(text, constraints)
            self._parse_off_days(text, constraints)
            self._parse_windows(text, constraints)
            self._parse_distance_limits(text, constraints)
            self._parse_daily_order_limits(text, constraints)
            self._parse_areas(text, constraints)
            self._parse_must_take(text, constraints)
            self._parse_home(text, constraints)
            self._parse_special_home_task(text, constraints)
        return constraints

    def _preference_texts(self, preferences: Any) -> list[str]:
        if not isinstance(preferences, list):
            return []
        texts: list[str] = []
        for item in preferences:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("content") or item.get("text") or "").strip()
            else:
                text = ""
            if text:
                texts.append(text)
        return texts

    def _parse_cargo_names(self, text: str, constraints: PreferenceConstraints) -> None:
        names = set(re.findall(r"「([^」]+)」", text))
        if not names:
            return
        if "尽量不" in text:
            constraints.soft_avoid_cargo_names.update(names)
        elif "不接" in text or "不得接" in text or "禁止接" in text:
            constraints.forbidden_cargo_names.update(names)

    def _parse_rest(self, text: str, constraints: PreferenceConstraints) -> None:
        if "每天" not in text and "平日" not in text:
            return
        if not any(word in text for word in ("休息", "停车", "歇")):
            return
        match = re.search(r"(?:连续|连着).*?(?:休息|停车|歇).*?(\d+|[一二两三四五六七八九十]+)\s*小时", text)
        if not match:
            match = re.search(r"(?:休息|停车|歇).*?(?:至少|满)(\d+|[一二两三四五六七八九十]+)\s*小时", text)
        if match:
            hours = parse_number(match.group(1))
            if hours > 0:
                minutes = hours * 60
                constraints.daily_rest_minutes = max(constraints.daily_rest_minutes or 0, minutes)

    def _parse_off_days(self, text: str, constraints: PreferenceConstraints) -> None:
        if "自然月" not in text and "月内" not in text:
            return
        if not any(word in text for word in ("整天", "完全", "放空")):
            return
        match = re.search(r"至少(?:要有)?(\d+|[一二两三四五六七八九十]+)\s*(?:个)?(?:整天|天)", text)
        if not match and "一整天" in text:
            constraints.monthly_off_days = max(constraints.monthly_off_days or 0, 1)
            return
        if match:
            days = parse_number(match.group(1))
            if days > 0:
                constraints.monthly_off_days = max(constraints.monthly_off_days or 0, days)

    def _parse_windows(self, text: str, constraints: PreferenceConstraints) -> None:
        if "不接单" not in text or not any(word in text for word in ("空驶", "空车", "空跑", "赶路")):
            return
        for start, end in extract_time_windows(text):
            if start == end:
                continue
            if 11 * 60 <= start <= 13 * 60:
                constraints.lunch_no_drive_windows.append((start, end))
            constraints.no_drive_windows.append((start, end))

    def _parse_distance_limits(self, text: str, constraints: PreferenceConstraints) -> None:
        value = extract_km_limit(text)
        if value is None:
            return
        if "装货点" in text and "卸货点" in text:
            constraints.max_order_distance_km = min_optional(constraints.max_order_distance_km, value)
        elif "赴装货点" in text or "接单后" in text:
            constraints.max_pickup_deadhead_km = min_optional(constraints.max_pickup_deadhead_km, value)
        elif "月" in text and "空驶" in text:
            constraints.max_monthly_deadhead_km = min_optional(constraints.max_monthly_deadhead_km, value)

    def _parse_daily_order_limits(self, text: str, constraints: PreferenceConstraints) -> None:
        match = re.search(r"同(?:一)?天接单不得超过(\d+|[一二两三四五六七八九十]+)\s*单", text)
        if match:
            constraints.max_daily_orders = parse_number(match.group(1))
        if "首单" in text and ("不晚于" in text or "不得晚于" in text):
            minute = parse_hhmm_from_text(text)
            if minute is not None:
                constraints.first_order_before_minute = minute

    def _parse_areas(self, text: str, constraints: PreferenceConstraints) -> None:
        bbox = re.search(
            r"北纬\s*([0-9.]+)\s*至\s*([0-9.]+).*?东经\s*([0-9.]+)\s*至\s*([0-9.]+)",
            text,
        )
        if bbox:
            constraints.must_stay_in_bbox = {
                "lat_min": float(bbox.group(1)),
                "lat_max": float(bbox.group(2)),
                "lng_min": float(bbox.group(3)),
                "lng_max": float(bbox.group(4)),
            }
        if "圆心" in text and "半径" in text:
            coords = extract_coordinates(text)
            radius = extract_km_limit(text)
            if coords and radius is not None:
                constraints.forbidden_circle = {"lat": coords[0][0], "lng": coords[0][1], "radius_km": radius}

    def _parse_must_take(self, text: str, constraints: PreferenceConstraints) -> None:
        if not any(word in text for word in ("指定", "熟货", "不接则", "老客户")):
            return
        task = dict(constraints.must_take_cargo or {})
        match = re.search(r"(?:货源编号|编号|cargo_id\s*=?)\s*([0-9A-Za-z_-]+)", text)
        if match:
            cargo_id = match.group(1)
            constraints.must_take_cargo_id = cargo_id
            task["cargo_id"] = cargo_id
        names = re.findall(r"品类[为是]?「([^」]+)」", text)
        if names:
            task["cargo_name"] = names[0]
        coords = extract_coordinates(text)
        if coords:
            task.setdefault("pickup", {"lat": coords[0][0], "lng": coords[0][1]})
        create_match = re.search(r"上架时间[：:]\s*(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)", text)
        if create_match:
            create_min = parse_optional_wall_time(create_match.group(1))
            if create_min is not None:
                task["visible_from_minute"] = create_min
        due_match = re.search(r"(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)前", text)
        if due_match:
            due_min = parse_optional_wall_time(due_match.group(1))
            if due_min is not None:
                task["deadline_minute"] = due_min
        if task:
            constraints.must_take_cargo = task

    def _parse_home(self, text: str, constraints: PreferenceConstraints) -> None:
        if "自家位置" not in text and "回家" not in text:
            return
        coords = extract_coordinates(text)
        if not coords:
            return
        return_before = 23 * 60
        match = re.search(r"每天\s*(\d{1,2})\s*点前", text)
        if match:
            return_before = int(match.group(1)) * 60
        quiet_end = 8 * 60
        quiet = re.search(r"次日\s*(?:早)?(\d{1,2})\s*点", text)
        if quiet:
            quiet_end = int(quiet.group(1)) * 60
        constraints.home_location = {
            "lat": coords[0][0],
            "lng": coords[0][1],
            "radius_km": 1.0,
            "return_before_minute": return_before,
            "quiet_end_minute": quiet_end,
        }
        if (23 * 60, quiet_end) not in constraints.no_drive_windows:
            constraints.no_drive_windows.append((return_before, quiet_end))

    def _parse_special_home_task(self, text: str, constraints: PreferenceConstraints) -> None:
        if "家中急事" not in text and "接上配偶" not in text:
            return
        coords = extract_coordinates(text)
        if len(coords) < 2:
            return
        datetimes = extract_chinese_datetimes(text)
        if not datetimes:
            return
        start_dt = datetimes[0]
        stay_until = datetimes[-1]
        due = None
        due_match = re.search(r"须在(\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2})前", text)
        if due_match:
            due = parse_chinese_datetime(due_match.group(1))
        constraints.special_home_task = {
            "start_minute": datetime_to_sim_minutes(start_dt),
            "due_minute": datetime_to_sim_minutes(due) if due else datetime_to_sim_minutes(stay_until),
            "stay_until_minute": datetime_to_sim_minutes(stay_until),
            "pickup": {"lat": coords[0][0], "lng": coords[0][1]},
            "home": {"lat": coords[1][0], "lng": coords[1][1]},
            "pickup_wait_minutes": 10,
        }


class LLMPreferenceParser:
    def __init__(self, api: SimulationApiPort, logger: logging.Logger) -> None:
        self._api = api
        self._logger = logger
        self._fallback = PreferenceParser()
        self._cache: dict[str, PreferenceConstraints] = {}

    def parse(self, preferences: Any) -> PreferenceConstraints:
        key = self._cache_key(preferences)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        fallback = self._fallback.parse(preferences)
        texts = self._fallback._preference_texts(preferences)
        if not texts:
            self._cache[key] = fallback
            return fallback
        parsed = self._call_llm(texts)
        if parsed is None:
            self._cache[key] = fallback
            return fallback
        constraints = self._constraints_from_llm(parsed, fallback)
        self._cache[key] = constraints
        return constraints

    @staticmethod
    def _cache_key(preferences: Any) -> str:
        try:
            return json.dumps(preferences, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return repr(preferences)

    def _call_llm(self, texts: list[str]) -> dict[str, Any] | None:
        payload = {
            "messages": self._build_messages(texts),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 900,
            "enable_thinking": False,
        }
        try:
            response = self._api.model_chat_completion(payload)
            content = self._extract_content(response)
            parsed = self._parse_json_content(content)
        except Exception as exc:
            self._logger.warning("LLM preference parse unavailable, fallback to rules: %s", exc.__class__.__name__)
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _build_messages(texts: list[str]) -> list[dict[str, str]]:
        schema = {
            "version": "preference_schema_v1",
            "constraints": {
                "forbidden_cargo_names": [],
                "soft_avoid_cargo_names": [],
                "daily_rest_minutes": None,
                "monthly_off_days": None,
                "no_drive_windows": [{"start_minute": 1380, "end_minute": 240}],
                "lunch_no_drive_windows": [],
                "max_pickup_deadhead_km": None,
                "max_order_distance_km": None,
                "max_monthly_deadhead_km": None,
                "max_daily_orders": None,
                "first_order_before_minute": None,
                "must_stay_in_bbox": None,
                "forbidden_circle": None,
                "home_location": None,
                "must_take_cargo": {"cargo_id": None, "cargo_name": None, "pickup_location": None, "visible_from_minute": None, "deadline_minute": None},
                "special_home_task": None,
                "must_visit_locations": [{"target_location": None, "radius_km": 1, "required_visit_days": None}],
            },
            "rules": [],
        }
        system_prompt = (
            "你是卡车司机偏好解析器。只返回紧凑 JSON 对象，不要解释、Markdown、代码块或推理过程。"
            "只抽取明示约束；未知字段填 null 或 []，禁止猜测。"
            "禁行/不接单窗口不要推断成休息时长。循环日内时间用 0-1439 分钟；跨天窗口 end_minute 小于 start_minute。"
            "带具体日期的时间转为相对 2026-03-01 00:00:00 的绝对分钟。经纬度统一 lat/lng。"
            "指定熟货/老客户货源写入 must_take_cargo；若原文出现装货点/装货地坐标，必须放入 pickup_location。"
            "月内到访指定点写入 must_visit_locations；不同自然日数量写入 required_visit_days。"
            "必须输出 version、constraints、rules 三个顶层字段；rules 可为空。"
            f"schema={json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )
        user_prompt = "请解析这些司机偏好，并严格返回 JSON 对象：\n" + json.dumps(texts, ensure_ascii=False)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("invalid choice")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = first.get("text")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty content")
        return content

    @staticmethod
    def _parse_json_content(content: str) -> Any:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    def _constraints_from_llm(
        self,
        parsed: dict[str, Any],
        fallback: PreferenceConstraints,
    ) -> PreferenceConstraints:
        constraints = fallback
        constraints.llm_parse = parsed
        raw_constraints = parsed.get("constraints")
        source = raw_constraints if isinstance(raw_constraints, dict) else {}
        self._apply_constraint_object(source, constraints)
        self._apply_rule_objects(parsed.get("rules"), constraints)
        constraints.no_drive_windows = dedupe_windows(constraints.no_drive_windows)
        constraints.lunch_no_drive_windows = dedupe_windows(constraints.lunch_no_drive_windows)
        return constraints

    def _apply_constraint_object(self, source: dict[str, Any], constraints: PreferenceConstraints) -> None:
        constraints.forbidden_cargo_names.update(strings_from_value(source.get("forbidden_cargo_names")))
        constraints.soft_avoid_cargo_names.update(strings_from_value(source.get("soft_avoid_cargo_names")))
        daily_rest = positive_int(source.get("daily_rest_minutes"))
        if daily_rest is not None:
            constraints.daily_rest_minutes = max(constraints.daily_rest_minutes or 0, daily_rest)
        off_days = positive_int(source.get("monthly_off_days"))
        if off_days is not None:
            constraints.monthly_off_days = max(constraints.monthly_off_days or 0, off_days)
        constraints.no_drive_windows.extend(windows_from_value(source.get("no_drive_windows")))
        constraints.lunch_no_drive_windows.extend(windows_from_value(source.get("lunch_no_drive_windows")))
        pickup_limit = positive_float(source.get("max_pickup_deadhead_km"))
        if pickup_limit is not None:
            constraints.max_pickup_deadhead_km = min_optional(constraints.max_pickup_deadhead_km, pickup_limit)
        order_limit = positive_float(source.get("max_order_distance_km"))
        if order_limit is not None:
            constraints.max_order_distance_km = min_optional(constraints.max_order_distance_km, order_limit)
        monthly_deadhead = positive_float(source.get("max_monthly_deadhead_km"))
        if monthly_deadhead is not None:
            constraints.max_monthly_deadhead_km = min_optional(constraints.max_monthly_deadhead_km, monthly_deadhead)
        max_daily_orders = positive_int(source.get("max_daily_orders"))
        if max_daily_orders is not None:
            constraints.max_daily_orders = max_daily_orders
        first_before = minute_from_value(source.get("first_order_before_minute") or source.get("first_order_before_hhmm"))
        if first_before is not None:
            constraints.first_order_before_minute = first_before
        bbox = bbox_from_value(source.get("must_stay_in_bbox"))
        if bbox is not None:
            constraints.must_stay_in_bbox = bbox
        circle = circle_from_value(source.get("forbidden_circle"))
        if circle is not None:
            constraints.forbidden_circle = circle
        home = home_from_value(source.get("home_location"))
        if home is not None:
            constraints.home_location = home
            constraints.no_drive_windows.append((int(home.get("return_before_minute", 23 * 60)), int(home.get("quiet_end_minute", 8 * 60))))
        must_take = normalized_task_from_value(source.get("must_take_cargo"))
        if must_take is not None:
            constraints.must_take_cargo = merge_task(constraints.must_take_cargo, must_take)
            cargo_id = str(constraints.must_take_cargo.get("cargo_id") or "").strip()
            if cargo_id:
                constraints.must_take_cargo_id = cargo_id
        special_home = normalized_task_from_value(source.get("special_home_task"))
        if special_home is not None:
            constraints.special_home_task = special_home
        constraints.must_visit_locations.extend(locations_from_value(source.get("must_visit_locations")))

    def _apply_rule_objects(self, rules: Any, constraints: PreferenceConstraints) -> None:
        if not isinstance(rules, list):
            return
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            parsed = rule.get("parsed") if isinstance(rule.get("parsed"), dict) else {}
            category = str(rule.get("category") or "").strip()
            hardness = str(rule.get("hardness") or "").strip()
            if category == "cargo":
                if hardness == "soft":
                    constraints.soft_avoid_cargo_names.update(strings_from_value(parsed.get("cargo_names")))
                else:
                    constraints.forbidden_cargo_names.update(strings_from_value(parsed.get("cargo_names")))
            elif category == "must_take_cargo":
                task = normalized_task_from_value(parsed)
                if task is not None:
                    constraints.must_take_cargo = merge_task(constraints.must_take_cargo, task)
                    cargo_id = str(constraints.must_take_cargo.get("cargo_id") or "").strip()
                    if cargo_id:
                        constraints.must_take_cargo_id = cargo_id
            elif category == "visit_location":
                constraints.must_visit_locations.extend(locations_from_value(parsed.get("locations") or parsed))


def enrich_must_take_from_raw_preferences(preferences: Any, constraints: PreferenceConstraints) -> None:
    if not isinstance(preferences, list):
        return
    for item in preferences:
        if isinstance(item, dict):
            text = str(item.get("content") or item.get("text") or "")
            end_time = item.get("end_time")
        elif isinstance(item, str):
            text = item
            end_time = None
        else:
            continue
        if not any(word in text for word in ("指定", "熟货", "老客户", "不接则")):
            continue
        task = dict(constraints.must_take_cargo or {})
        match = re.search(r"(?:货源编号|编号|cargo_id\s*=?)\s*([0-9A-Za-z_-]+)", text)
        if match:
            cargo_id = match.group(1)
            constraints.must_take_cargo_id = cargo_id
            task["cargo_id"] = cargo_id
        names = re.findall(r"品类[为是]?「([^」]+)」", text)
        if names:
            task["cargo_name"] = names[0]
        coords = extract_coordinates(text)
        if coords:
            task["pickup"] = {"lat": coords[0][0], "lng": coords[0][1]}
        create_match = re.search(r"上架时间[：:]\s*(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)", text)
        if create_match:
            create_min = parse_optional_wall_time(create_match.group(1))
            if create_min is not None:
                task["visible_from_minute"] = create_min
        end_min = parse_optional_wall_time(end_time) if isinstance(end_time, str) else None
        if end_min is not None:
            task["visible_until_minute"] = end_min
        if task:
            constraints.must_take_cargo = task


def strings_from_value(value: Any) -> set[str]:
    if isinstance(value, str):
        text = value.strip()
        return {text} if text else set()
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            if isinstance(item, str) and item.strip():
                out.add(item.strip())
        return out
    return set()


def positive_int(value: Any) -> int | None:
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def positive_float(value: Any) -> float | None:
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def minute_from_value(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        minute = int(value)
        return minute if minute >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() == "null":
        return None
    if text.isdigit():
        return int(text)
    if any(marker in text for marker in ("年", "-", "/")):
        wall = parse_optional_wall_time(text)
        if wall is not None:
            return wall
        try:
            return datetime_to_sim_minutes(parse_chinese_datetime(text))
        except ValueError:
            pass
    hhmm = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", text)
    if hhmm:
        hour = int(hhmm.group(1))
        minute = int(hhmm.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour * 60 + minute
    hour_match = re.search(r"(\d{1,2})\s*点", text)
    if hour_match:
        hour = int(hour_match.group(1))
        if 0 <= hour <= 24:
            return (hour % 24) * 60
    wall = parse_optional_wall_time(text)
    if wall is not None:
        return wall
    try:
        return datetime_to_sim_minutes(parse_chinese_datetime(text))
    except ValueError:
        return None


def windows_from_value(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        return []
    out: list[tuple[int, int]] = []
    for item in value:
        start = end = None
        if isinstance(item, dict):
            start = minute_from_value(item.get("start_minute") or item.get("start") or item.get("from"))
            end = minute_from_value(item.get("end_minute") or item.get("end") or item.get("to"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = minute_from_value(item[0])
            end = minute_from_value(item[1])
        if start is not None and end is not None and start != end:
            out.append((start % 1440, end % 1440))
    return dedupe_windows(out)


def lat_lng_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        lat = value.get("lat") if "lat" in value else value.get("latitude")
        lng = value.get("lng") if "lng" in value else value.get("longitude")
        if lat is None or lng is None:
            return None
        parsed_lat = safe_float(lat, 999.0)
        parsed_lng = safe_float(lng, 999.0)
        if -90 <= parsed_lat <= 90 and -180 <= parsed_lng <= 180:
            return parsed_lat, parsed_lng
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        parsed_lat = safe_float(value[0], 999.0)
        parsed_lng = safe_float(value[1], 999.0)
        if -90 <= parsed_lat <= 90 and -180 <= parsed_lng <= 180:
            return parsed_lat, parsed_lng
    return None


def location_from_mapping(source: dict[str, Any], *keys: str) -> tuple[float, float] | None:
    direct = lat_lng_from_value(source)
    if direct is not None:
        return direct
    for key in keys:
        value = source.get(key)
        parsed = lat_lng_from_value(value)
        if parsed is not None:
            return parsed
    return None


def bbox_from_value(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        bbox = {
            "lat_min": float(value["lat_min"]),
            "lat_max": float(value["lat_max"]),
            "lng_min": float(value["lng_min"]),
            "lng_max": float(value["lng_max"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if bbox["lat_min"] > bbox["lat_max"] or bbox["lng_min"] > bbox["lng_max"]:
        return None
    return bbox


def circle_from_value(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    location = lat_lng_from_value(value.get("center") or value)
    radius = positive_float(value.get("radius_km") or value.get("radius"))
    if location is None or radius is None:
        return None
    return {"lat": location[0], "lng": location[1], "radius_km": radius}


def home_from_value(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    location = lat_lng_from_value(value.get("location") or value)
    if location is None:
        return None
    return_before = minute_from_value(value.get("return_before_minute") or value.get("return_before"))
    quiet_end = minute_from_value(value.get("quiet_end_minute") or value.get("quiet_end"))
    return {
        "lat": location[0],
        "lng": location[1],
        "radius_km": positive_float(value.get("radius_km")) or 1.0,
        "return_before_minute": return_before if return_before is not None else 23 * 60,
        "quiet_end_minute": quiet_end if quiet_end is not None else 8 * 60,
    }


def normalized_task_from_value(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    task = {k: v for k, v in value.items() if v is not None}
    for key in (
        "visible_from_minute",
        "visible_until_minute",
        "create_minute",
        "remove_minute",
        "deadline_minute",
        "start_minute",
        "due_minute",
        "stay_until_minute",
    ):
        if key in task:
            parsed = minute_from_value(task[key])
            if parsed is not None:
                task[key] = parsed
    for key in (
        "start",
        "pickup",
        "origin",
        "end",
        "destination",
        "home",
        "location",
        "start_location",
        "pickup_location",
        "origin_location",
        "loading_location",
    ):
        if key in task:
            location = lat_lng_from_value(task[key])
            if location is not None:
                task[key] = {"lat": location[0], "lng": location[1]}
    for alias, target in (
        ("start_location", "start"),
        ("pickup_location", "pickup"),
        ("origin_location", "origin"),
        ("loading_location", "pickup"),
        ("destination_location", "destination"),
        ("dropoff_location", "destination"),
        ("home_location", "home"),
    ):
        if target not in task and alias in task:
            task[target] = task[alias]
    location = location_from_mapping(
        task,
        "start",
        "pickup",
        "origin",
        "location",
        "start_location",
        "pickup_location",
        "origin_location",
        "loading_location",
    )
    cargo_id = str(task.get("cargo_id") or "").strip()
    cargo_name = str(task.get("cargo_name") or "").strip()
    if not task or (location is None and not cargo_id and not cargo_name and "home" not in task):
        return None
    if cargo_id:
        task["cargo_id"] = cargo_id
    return task


def merge_task(base: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None:
        return update
    if update is None:
        return base
    merged = dict(base)
    merged.update(update)
    for key in ("start", "pickup", "origin", "location", "home", "destination"):
        if key not in merged and key in base:
            merged[key] = base[key]
    for key in (
        "visible_from_minute",
        "visible_until_minute",
        "create_minute",
        "remove_minute",
        "deadline_minute",
        "start_minute",
        "due_minute",
        "stay_until_minute",
    ):
        base_minute = base.get(key)
        merged_minute = merged.get(key)
        if isinstance(base_minute, int) and isinstance(merged_minute, int):
            if base_minute >= 1440 and 0 <= merged_minute < 1440:
                merged[key] = base_minute
    return merged


def locations_from_value(value: Any) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else [value]
    out: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        location = lat_lng_from_value(
            item.get("location")
            or item.get("target_location")
            or item.get("visit_location")
            or item.get("point")
            or item.get("center")
            or item
        )
        if location is None:
            continue
        parsed = dict(item)
        parsed["lat"] = location[0]
        parsed["lng"] = location[1]
        for key in ("arrive_after_minute", "arrive_before_minute", "stay_until_minute"):
            if key in parsed:
                minute = minute_from_value(parsed[key])
                if minute is not None:
                    parsed[key] = minute
        out.append(parsed)
    return out


class CandidateEvaluator:
    def choose_best(
        self,
        items: list[Any],
        status: dict[str, Any],
        constraints: PreferenceConstraints,
        memory: DriverMemory,
    ) -> tuple[EvaluatedCargo | None, EvaluatedCargo | None]:
        now_min = parse_sim_minutes(status)
        current_lat, current_lng = read_position(status)
        truck_length = str(status.get("truck_length") or "").strip()
        best: EvaluatedCargo | None = None
        must_take: EvaluatedCargo | None = None
        for item in items:
            cargo = item.get("cargo") if isinstance(item, dict) else None
            if not isinstance(cargo, dict):
                continue
            evaluated = self._evaluate(cargo, now_min, current_lat, current_lng, truck_length, constraints, memory)
            if evaluated is None:
                continue
            if constraints.must_take_cargo_id and evaluated.cargo_id == constraints.must_take_cargo_id:
                if must_take is None or evaluated.score > must_take.score:
                    must_take = evaluated
            if best is None or evaluated.score > best.score:
                best = evaluated
        return best, must_take

    def _evaluate(
        self,
        cargo: dict[str, Any],
        now_min: int,
        current_lat: float,
        current_lng: float,
        truck_length: str,
        constraints: PreferenceConstraints,
        memory: DriverMemory,
    ) -> EvaluatedCargo | None:
        cargo_id = str(cargo.get("cargo_id") or "").strip()
        if not cargo_id:
            return None
        cargo_name = str(cargo.get("cargo_name") or "").strip()
        if cargo_name in constraints.forbidden_cargo_names:
            return None
        if not truck_length_matches(truck_length, cargo.get("truck_length")):
            return None

        start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        try:
            start_lat, start_lng = float(start["lat"]), float(start["lng"])
            end_lat, end_lng = float(end["lat"]), float(end["lng"])
        except (KeyError, TypeError, ValueError):
            return None

        if not all(point_allowed(a, b, constraints) for a, b in ((current_lat, current_lng), (start_lat, start_lng), (end_lat, end_lng))):
            return None

        create_min = parse_optional_wall_time(cargo.get("create_time"))
        remove_min = parse_optional_wall_time(cargo.get("remove_time"))
        if create_min is not None and now_min < create_min:
            return None
        if remove_min is not None and now_min > remove_min:
            return None

        pickup_deadhead_km = haversine_km(current_lat, current_lng, start_lat, start_lng)
        order_distance_km = haversine_km(start_lat, start_lng, end_lat, end_lng)
        if constraints.max_pickup_deadhead_km is not None and pickup_deadhead_km > constraints.max_pickup_deadhead_km:
            return None
        if constraints.max_order_distance_km is not None and order_distance_km > constraints.max_order_distance_km:
            return None
        if constraints.max_monthly_deadhead_km is not None:
            if memory.estimated_deadhead_km + pickup_deadhead_km > constraints.max_monthly_deadhead_km:
                return None

        pickup_minutes = distance_minutes(pickup_deadhead_km) if pickup_deadhead_km > 1e-6 else 0
        arrive_pickup = now_min + pickup_minutes
        load_start, load_end = parse_load_window(cargo.get("load_time"))
        if load_end is not None and arrive_pickup > load_end:
            return None
        wait_minutes = max(0, (load_start or arrive_pickup) - arrive_pickup)
        cost_time_minutes = safe_int(cargo.get("cost_time_minutes"), 0)
        if cost_time_minutes <= 0:
            return None
        total_minutes = pickup_minutes + wait_minutes + cost_time_minutes
        finish_minute = now_min + total_minutes

        if finish_minute > 30 * 1440:
            return None
        if now_min >= 29 * 1440 and total_minutes > 8 * 60:
            return None
        if overlap_no_drive_window(now_min, finish_minute, all_no_drive_windows(constraints)):
            return None
        if blocks_upcoming_must_take(cargo_id, now_min, finish_minute, current_lat, current_lng, constraints):
            return None
        home = constraints.home_location
        if home:
            return_before = int(home.get("return_before_minute", 23 * 60))
            finish_mod = finish_minute % 1440
            if finish_mod < return_before:
                home_lat = float(home["lat"])
                home_lng = float(home["lng"])
                travel_home = distance_minutes(haversine_km(end_lat, end_lng, home_lat, home_lng))
                if finish_mod + travel_home > return_before - 30:
                    return None

        day = now_min // 1440
        if constraints.daily_rest_minutes and not daily_rest_satisfied(day, memory, constraints.daily_rest_minutes):
            latest_finish = (day + 1) * 1440 - constraints.daily_rest_minutes
            if finish_minute > latest_finish:
                return None
        if constraints.first_order_before_minute is not None and memory.daily_order_count.get(day, 0) == 0:
            if now_min % 1440 >= constraints.first_order_before_minute:
                return None
        if constraints.max_daily_orders is not None:
            if memory.daily_order_count.get(day, 0) >= constraints.max_daily_orders:
                return None

        revenue = safe_float(cargo.get("price"), 0.0)
        cost = (pickup_deadhead_km + order_distance_km) * DEFAULT_COST_PER_KM
        expected_profit = revenue - cost
        profit_per_hour = expected_profit / max(total_minutes / 60.0, 0.1)
        preference_risk = 0.0
        if cargo_name in constraints.soft_avoid_cargo_names:
            preference_risk += 500.0
        preference_risk += wait_minutes * 0.4
        if constraints.daily_rest_minutes and total_minutes > 8 * 60:
            preference_risk += 120.0
        if upcoming_no_drive_wait(finish_minute, all_no_drive_windows(constraints), threshold_minutes=60) is not None:
            preference_risk += 200.0
        destination_bonus = destination_hub_bonus(end_lat, end_lng, constraints)
        score = expected_profit + 0.35 * profit_per_hour - wait_minutes * 0.4 - preference_risk + destination_bonus
        return EvaluatedCargo(
            cargo_id=cargo_id,
            score=score,
            expected_profit=expected_profit,
            profit_per_hour=profit_per_hour,
            finish_minute=finish_minute,
            pickup_deadhead_km=pickup_deadhead_km,
            order_distance_km=order_distance_km,
        )


def validate_action(action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("action must be a dict")
    action_name = str(action.get("action", "")).strip().lower()
    params = action.get("params")
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    if action_name == "take_order":
        cargo_id = str(params.get("cargo_id", "")).strip()
        if not cargo_id:
            raise ValueError("take_order requires cargo_id")
        return {"action": "take_order", "params": {"cargo_id": cargo_id}}
    if action_name == "reposition":
        return {
            "action": "reposition",
            "params": {"latitude": float(params["latitude"]), "longitude": float(params["longitude"])},
        }
    if action_name == "wait":
        return wait_action(safe_int(params.get("duration_minutes"), SAFE_WAIT_MINUTES))
    raise ValueError(f"unknown action: {action_name}")


def wait_action(duration_minutes: int, min_minutes: int = 5, max_minutes: int = 480) -> dict[str, Any]:
    duration = max(min_minutes, min(max_minutes, int(duration_minutes)))
    return {"action": "wait", "params": {"duration_minutes": duration}}


def parse_sim_minutes(status: dict[str, Any]) -> int:
    raw = status.get("simulation_progress_minutes")
    if isinstance(raw, (int, float)):
        return max(0, int(raw))
    wall = status.get("simulation_wall_time")
    parsed = parse_optional_wall_time(wall)
    return parsed if parsed is not None else 0


def parse_datetime(text: str) -> datetime:
    stripped = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(stripped, fmt)
        except ValueError:
            pass
    raise ValueError(f"invalid datetime: {text}")


def parse_optional_wall_time(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime_to_sim_minutes(parse_datetime(value))
    except ValueError:
        return None


def datetime_to_sim_minutes(value: datetime) -> int:
    return int((value - SIMULATION_EPOCH).total_seconds() // 60)


def minutes_of_day(value: datetime) -> int:
    return value.hour * 60 + value.minute


def read_position(status: dict[str, Any]) -> tuple[float, float]:
    if "current_lat" in status and "current_lng" in status:
        return safe_float(status.get("current_lat"), 0.0), safe_float(status.get("current_lng"), 0.0)
    pos = status.get("position") if isinstance(status.get("position"), dict) else {}
    return safe_float(pos.get("lat"), 0.0), safe_float(pos.get("lng"), 0.0)


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def distance_minutes(distance_km: float, speed_km_per_hour: float = REPOSITION_SPEED_KMPH) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil(distance_km / speed_km_per_hour * 60.0))


def parse_load_window(value: Any) -> tuple[int | None, int | None]:
    if not isinstance(value, list) or len(value) != 2:
        return None, None
    return parse_optional_wall_time(value[0]), parse_optional_wall_time(value[1])


def parse_number(value: str) -> int:
    text = value.strip()
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        return (CHINESE_NUMBERS.get(left, 1) * 10) + CHINESE_NUMBERS.get(right, 0)
    return CHINESE_NUMBERS.get(text, 0)


def extract_km_limit(text: str) -> float | None:
    match = re.search(r"(?:超过|半径)\s*([0-9.]+)\s*公里", text)
    if not match:
        return None
    return float(match.group(1))


def extract_coordinates(text: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    pattern = r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]"
    for lat, lng in re.findall(pattern, text):
        out.append((float(lat), float(lng)))
    return out


def parse_hhmm_from_text(text: str) -> int | None:
    if "中午" in text:
        match = re.search(r"中午\s*(\d{1,2})\s*点", text)
        if match:
            hour = int(match.group(1))
            if hour < 12:
                hour += 12 if hour == 1 else 0
            return hour * 60
        return 12 * 60
    match = re.search(r"(\d{1,2})\s*点", text)
    if not match:
        return None
    return int(match.group(1)) * 60


def extract_time_windows(text: str) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    normalized = text.replace("凌晨", "").replace("早", "").replace("晚上", "").replace("每晚", "每天")
    patterns = [
        r"(\d{1,2})\s*点\s*(?:至|到|-)\s*次日\s*(\d{1,2})\s*点",
        r"(\d{1,2})\s*点\s*(?:至|到|-)\s*(?:下午)?\s*(\d{1,2})\s*点",
    ]
    for idx, pattern in enumerate(patterns):
        for start_s, end_s in re.findall(pattern, normalized):
            start = int(start_s) * 60
            end_hour = int(end_s)
            if idx == 1 and "下午" in normalized and end_hour < 12:
                end_hour += 12
            end = end_hour * 60
            windows.append((start, end))
    return dedupe_windows(windows)


def dedupe_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in windows:
        if item not in out:
            out.append(item)
    return out


def all_no_drive_windows(constraints: PreferenceConstraints) -> list[tuple[int, int]]:
    return dedupe_windows([*constraints.no_drive_windows, *constraints.lunch_no_drive_windows])


def current_no_drive_window_end(now_min: int, windows: list[tuple[int, int]]) -> int | None:
    for start, end in windows:
        candidates = window_absolute_ranges(now_min - 1440, now_min + 1440, start, end)
        for abs_start, abs_end in candidates:
            if abs_start <= now_min < abs_end:
                return abs_end
    return None


def upcoming_no_drive_wait(now_min: int, windows: list[tuple[int, int]], threshold_minutes: int) -> int | None:
    for start, end in windows:
        for abs_start, abs_end in window_absolute_ranges(now_min, now_min + threshold_minutes + 1440, start, end):
            if now_min <= abs_start <= now_min + threshold_minutes:
                return abs_end - now_min
    return None


def overlap_no_drive_window(start_min: int, end_min: int, windows: list[tuple[int, int]]) -> bool:
    if end_min <= start_min:
        return False
    for start, end in windows:
        for abs_start, abs_end in window_absolute_ranges(start_min - 1440, end_min + 1440, start, end):
            if max(start_min, abs_start) < min(end_min, abs_end):
                return True
    return False


def window_absolute_ranges(range_start: int, range_end: int, start: int, end: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    first_day = max(-1, range_start // 1440 - 1)
    last_day = range_end // 1440 + 1
    for day in range(first_day, last_day + 1):
        abs_start = day * 1440 + start
        abs_end = day * 1440 + end
        if end <= start:
            abs_end += 1440
        if abs_end >= range_start and abs_start <= range_end:
            out.append((abs_start, abs_end))
    return out


def absolute_next_minute(now_min: int, minute_of_day: int) -> int:
    day_base = now_min // 1440 * 1440
    candidate = day_base + minute_of_day
    if candidate <= now_min:
        candidate += 1440
    return candidate


def split_by_day(start_min: int, end_min: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    cur = start_min
    while cur < end_min:
        day = cur // 1440
        day_start = day * 1440
        seg_end = min(end_min, day_start + 1440)
        out.append((day, cur - day_start, seg_end - day_start))
        cur = seg_end
    return out


def daily_rest_satisfied(day: int, memory: DriverMemory, need_minutes: int) -> bool:
    intervals = sorted(memory.daily_rest_segments.get(day, []))
    if not intervals:
        return False
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return any(end - start >= need_minutes for start, end in merged)


def point_allowed(lat: float, lng: float, constraints: PreferenceConstraints) -> bool:
    bbox = constraints.must_stay_in_bbox
    if bbox and not (bbox["lat_min"] <= lat <= bbox["lat_max"] and bbox["lng_min"] <= lng <= bbox["lng_max"]):
        return False
    circle = constraints.forbidden_circle
    if circle:
        dist = haversine_km(lat, lng, float(circle["lat"]), float(circle["lng"]))
        if dist <= float(circle["radius_km"]):
            return False
    return True


def blocks_upcoming_must_take(
    cargo_id: str,
    now_min: int,
    finish_minute: int,
    current_lat: float,
    current_lng: float,
    constraints: PreferenceConstraints,
) -> bool:
    task = constraints.must_take_cargo
    if not task:
        return False
    must_cargo_id = str(task.get("cargo_id") or constraints.must_take_cargo_id or "").strip()
    if must_cargo_id and cargo_id == must_cargo_id:
        return False
    start = location_from_mapping(
        task,
        "start",
        "pickup",
        "origin",
        "location",
        "start_location",
        "pickup_location",
        "origin_location",
        "loading_location",
    )
    visible_from = minute_from_value(task.get("visible_from_minute") or task.get("create_minute"))
    if start is None or visible_from is None or now_min >= visible_from:
        return False
    travel_minutes = distance_minutes(haversine_km(current_lat, current_lng, start[0], start[1]))
    reserve_start = max(0, visible_from - travel_minutes - 20)
    return finish_minute > reserve_start and visible_from - now_min <= 720


def truck_length_matches(driver_truck: str, cargo_truck: Any) -> bool:
    if not driver_truck or cargo_truck is None:
        return True
    if isinstance(cargo_truck, list):
        return driver_truck in {str(x) for x in cargo_truck}
    if isinstance(cargo_truck, str):
        return driver_truck == cargo_truck or driver_truck in cargo_truck
    return True


def destination_hub_bonus(lat: float, lng: float, constraints: PreferenceConstraints) -> float:
    best = None
    for _, hlat, hlng in HUBS:
        if not point_allowed(hlat, hlng, constraints):
            continue
        dist = haversine_km(lat, lng, hlat, hlng)
        best = dist if best is None else min(best, dist)
    if best is None:
        return 0.0
    if best <= 10:
        return 80.0
    if best <= 30:
        return 40.0
    if best >= 100:
        return -60.0
    return 0.0


def min_optional(current: float | None, value: float) -> float:
    return value if current is None else min(current, value)


def parse_chinese_datetime(text: str) -> datetime:
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2}):(\d{2})", text)
    if not match:
        raise ValueError(f"invalid datetime: {text}")
    year, month, day, hour, minute = map(int, match.groups())
    return datetime(year, month, day, hour, minute, 0)


def extract_chinese_datetimes(text: str) -> list[datetime]:
    out: list[datetime] = []
    for match in re.finditer(r"\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}", text):
        out.append(parse_chinese_datetime(match.group(0)))
    return out
