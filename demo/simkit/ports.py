"""跨模块协议：仅依赖 typing，避免与 server/agent 实现相互引用。"""

from __future__ import annotations

from typing import Any, Protocol


class AgentDecisionPort(Protocol):
    """评测主循环调用的固定决策接口。"""

    def decide(self, driver_id: str) -> dict[str, Any]: ...


class SimulationApiPort(Protocol):
    """决策服务在一步决策中所需的外部能力（状态 + 模型）。"""

    def get_driver_status(self, driver_id: str) -> dict[str, Any]: ...

    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict[str, Any]: ...

    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]:
        """当前评测会话内存中的历史（与编排器写入 jsonl 前追加的条目一致）；不读磁盘。"""
        ...

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...
