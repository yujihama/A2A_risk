from __future__ import annotations

import abc
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field


class NodeResult(BaseModel):
    """ノードが返却する結果パケット。Engine が state へ apply する。"""

    observation: Optional[Any] = None  # allow flexible types (dict, str, list)
    # state へ適用する patch (dict の再帰マージ想定) ※ None なら適用なし
    patch: Dict[str, Any] = Field(default_factory=dict)
    # history へ追加する汎用イベント。Engine 側で型付けせずにそのまま push
    events: List[Dict[str, Any]] = Field(default_factory=list)
    # 後続エッジ Condition へのヒント
    suggestions: Optional[Dict[str, Any]] = None


class Node(abc.ABC):
    """全ノード共通の抽象基底クラス"""

    id: str = ""

    def __init__(self, *, node_id: Optional[str] = None):
        if node_id:
            self.id = node_id

    @abc.abstractmethod
    async def run(self, state: "DynamicAgentState", toolbox: "ToolBox") -> NodeResult:  # noqa: F821
        """ノードのメイン処理。state を直接 mutate しない。"""
        raise NotImplementedError 