import logging
import random
from typing import Any, Dict, List
import datetime

from ..core.node_base import Node, NodeResult, make_history_entry
from ..prompts import get_evaluate_hypothesis_prompt

logger = logging.getLogger(__name__)


class EvaluateHypothesisNode(Node):
    """仮説評価ノード

    LLM があれば客観的に評価、無い場合は簡易スコアリング。
    評価結果は `status` と `evaluation_reason` に格納。
    """

    id = "evaluate_hypothesis"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: EvaluateHypothesis ---")

        # currently_investigating_hypothesis_idのみを対象にする
        hyps: List[Dict[str, Any]] = state.get("current_hypotheses", [])
        currently_investigating_hypothesis_id = state.get("currently_investigating_hypothesis_id")
        if currently_investigating_hypothesis_id:
            hyps = [h for h in hyps if h["id"] == currently_investigating_hypothesis_id]

        if not hyps:
            logger.warning("No hypotheses to evaluate.")
            return NodeResult(observation="no_hypotheses", patch={}, events=[])

        if toolbox.llm:
            evaluated: List[Dict[str, Any]] = []
            for h in hyps:
                parameters = {"hypothesis_id": h["id"]}
                prompt = get_evaluate_hypothesis_prompt(state, h, parameters)
                try:
                    resp = await toolbox.eval_llm.ainvoke(prompt)
                    if isinstance(resp, dict) and "evaluation_status" in resp:
                        h["status"] = resp["evaluation_status"]
                        h["evaluation_reason"] = resp.get("reasoning", resp.get("reason", ""))
                        h["required_next_data"] = resp.get("required_next_data")
                        logger.info(
                            f"[AE] 評価結果: id={h['id']}, status={h['status']}, reasoning={h['evaluation_reason']}, "
                            f"required_next_data={h.get('required_next_data')}")
                    else:
                        h["status"] = "inconclusive"
                        h["evaluation_reason"] = ""
                        h["required_next_data"] = None
                        logger.info(
                            f"[AE] 評価失敗: id={h['id']}, status=inconclusive, reasoning=, required_next_data=None")
                except Exception as e:  # noqa: BLE001
                    logger.warning("LLM evaluation failed: %s", e)
                    h["status"] = "inconclusive"
                evaluated.append(h)
            hyps = evaluated
        else:
            # error
            logger.error("LLM is not available")
            raise Exception("LLM is not available")

        events = [make_history_entry(
            "node",
            {"name": "evaluate_hypothesis", "count": len(hyps)},
            state
        )]
        # 各仮説の評価結果を observation として追加
        for h in hyps:
            events.append(make_history_entry(
                "observation",
                {
                    "hypothesis_id": h["id"],
                    "status": h.get("status"),
                    "evaluation_reason": h.get("evaluation_reason"),
                    "required_next_data": h.get("required_next_data")
                },
                state
            ))

        # 評価済みの仮説を state に反映させる
        # patch = {"current_hypotheses": hyps}
        patch = {"next_action": None}

        return NodeResult(observation="hypotheses_evaluated", patch=patch, events=events) 