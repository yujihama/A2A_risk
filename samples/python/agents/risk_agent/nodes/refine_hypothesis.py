import logging
import uuid
from typing import Any, Dict, List

from ..core.node_base import Node, NodeResult, make_history_entry

logger = logging.getLogger(__name__)


class RefineHypothesisNode(Node):
    id = "refine_hypothesis"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: RefineHypothesis ---")

        hyps: List[Dict[str, Any]] = state.get("current_hypotheses", [])
        supported = [h for h in hyps if h.get("status") == "supported"]
        focus_hyps = supported or hyps[:1]

        new_hyps: List[Dict[str, Any]] = []

        if toolbox.llm and focus_hyps:
            prompt = (
                "You are a scientist. Suggest refined or follow-up hypotheses based on the following:\n"
                f"CURRENT SUPPORTED OR FOCUSED HYPOTHESES: {focus_hyps}\n"
                "Return JSON list with objects {id,text,priority}."
            )
            try:
                resp = await toolbox.llm.ainvoke(prompt)
                if isinstance(resp, list):
                    new_hyps = resp
                elif isinstance(resp, dict) and "hypotheses" in resp:
                    new_hyps = resp["hypotheses"]
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM refine failed: %s", e)

        if not new_hyps:
            # fallback simple variant
            for h in focus_hyps:
                new_hyps.append({
                    "id": str(uuid.uuid4())[:8],
                    "text": h["text"] + " (refined)",
                    "priority": h.get("priority", 5),
                    "status": "new",
                })

        hyps.extend(new_hyps)

        patch = {"current_hypotheses": hyps, "next_action": None}
        events = [make_history_entry("node", {"name": "refine_hypothesis", "new_count": len(new_hyps)}, state)]

        return NodeResult(observation=new_hyps, patch=patch, events=events) 