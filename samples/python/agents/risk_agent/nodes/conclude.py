import logging
from typing import Any, Dict, List

from ..core.node_base import Node, NodeResult, make_history_entry

logger = logging.getLogger(__name__)


class ConcludeNode(Node):
    id = "conclude"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: Conclude ---")

        objective = state.get("objective", "")
        hypothesis = state.get("current_hypothesis", {})
        # currently_investigating_hypothesis_id が設定されていない、またはnullになっているhistoryを取得
        hyp_history = [h for h in state.get("history", []) if h.get("currently_investigating_hypothesis_id") is not None]

        summary = ""
        if toolbox.llm:
            prompt = f"""
            ### Role
            あなたは内部監査人として、不正の検証過程を詳細に確認して評価を行います。            
            
            ### Task
            不正の検出を行うための検証過程を詳細に確認して評価を行います。
            
            ### Context
            ・OBJECTIVE: {objective}
            ・HYPOTHESES: {hypothesis}
            ・HYPOTHESES HISTORY: {hyp_history}
            
            ### Output
            ・上記のHYPOTHESESの検証過程を詳細に確認し、最終的にOBJECTIVEの不正に該当するデータが存在するかどうかを判断します。
            ・resultというキーにJSONで回答してください。  
            """
            try:
                summary = await toolbox.llm.ainvoke(prompt)
                
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM conclusion failed: %s", e)

        patch = {"hypothesis_result": {"summary": summary}, "next_action": None}
        events = [make_history_entry("node", {"name": "conclude"}, state)]
        return NodeResult(observation=summary, patch=patch, events=events) 