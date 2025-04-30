import logging
import uuid
from typing import Any, Dict, List

from ..core.node_base import Node, NodeResult
from ..prompts import get_generate_hypothesis_prompt

logger = logging.getLogger(__name__)


class GenerateHypothesisNode(Node):
    """仮説生成ノード

    現在のデータ概要と目的を元に LLM に仮説案を生成してもらう。
    LLM が使えない場合はルールベースで１件ダミー仮説を作る。
    """

    id = "generate_hypothesis"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: GenerateHypothesis ---")

        objective: str = state.get("objective", "No objective provided")
        data_summary = state.get("collected_data_summary", {})

        hypos: List[Dict[str, Any]] = []

        if toolbox.llm:
            prompt = get_generate_hypothesis_prompt(state)
            try:
                resp = await toolbox.llm.ainvoke(prompt)
                if isinstance(resp, list):
                    hypos = resp  # assume already list[dict]
                elif isinstance(resp, dict) and ("hypotheses" in resp or "hypothesis" in resp):
                    hypos = resp.get("hypotheses") or resp.get("hypothesis")
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM failed, fallback: %s", e)

        if not hypos:
            # fallback single hypo
            hypos = [{
                "id": str(uuid.uuid4())[:8],
                "text": f"{objective} に関する仮説",
                "priority": 5,
                "status": "new",
            }]

        patch = {
            "current_hypotheses": hypos,
            "next_action": None,
        }

        # console 出力
        logger.info("Generated hypotheses: %s", hypos)

        events = [{"type": "node", "name": "generate_hypothesis", "count": len(hypos)}]

        return NodeResult(observation=hypos, patch=patch, events=events) 