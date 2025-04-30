import logging
from typing import Any, Dict, List

from ..core.node_base import Node, NodeResult

logger = logging.getLogger(__name__)


class ConcludeNode(Node):
    id = "conclude"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: Conclude ---")

        objective = state.get("objective", "")
        hyps: List[Dict[str, Any]] = state.get("current_hypotheses", [])
        analysis = state.get("analysis_result", {})

        summary = "Summary not generated"
        if toolbox.llm:
            prompt = (
                "You are to write a concise conclusion based on objective, hypotheses and analysis results.\n"
                f"OBJECTIVE: {objective}\nHYPOTHESES: {hyps}\nANALYSIS: {analysis}\n"
                "Return JSON summary."
            )
            try:
                resp = await toolbox.llm.ainvoke(prompt)
                if isinstance(resp, str):
                    summary = resp
                elif isinstance(resp, dict):
                    summary = resp.get("text") or summary
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM conclusion failed: %s", e)

        patch = {"final_result": {"summary": summary}, "next_action": None}
        events = [{"type": "node", "name": "conclude"}]
        return NodeResult(observation=summary, patch=patch, events=events) 