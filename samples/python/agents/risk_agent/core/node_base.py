from __future__ import annotations

import abc
from typing import Any, Dict, Optional, List
import datetime

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
    
class ToolBox: 
    def __init__(self, *, llm=None, eval_llm=None, smart_a2a_client=None, data_analyzer=None):
        self.llm = llm
        self.eval_llm = eval_llm
        self.smart_a2a_client = smart_a2a_client
        self.data_client = smart_a2a_client
        self.data_analyzer = data_analyzer

def make_history_entry(entry_type: str, content: Any, state: dict) -> dict:
    """
    履歴エントリを共通生成するユーティリティ。
    必須: type, content, timestamp, currently_investigating_hypothesis_id
    """
    return {
        "type": entry_type,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat(),
        "currently_investigating_hypothesis_id": state.get("currently_investigating_hypothesis_id")
    }