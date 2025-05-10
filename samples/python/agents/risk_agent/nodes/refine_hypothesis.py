import logging
import uuid
from typing import Any, Dict, List
import asyncio

from ..core.node_base import Node, NodeResult, make_history_entry
from ..prompts import get_refine_hypothesis_prompt
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..nodes.generate_hypothesis import HypothesisList

logger = logging.getLogger(__name__)


class RefineHypothesisNode(Node):
    id = "refine_hypothesis"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: RefineHypothesis ---")

        # フォーカス仮説を取得
        params = state.get("next_action", {}).get("parameters", {})
        hypothesis_id = params.get("hypothesis_id")
        existing_hypotheses: List[Dict] = state.get("current_hypotheses", [])
        target_hypothesis = next((h for h in existing_hypotheses if h.get('id') == hypothesis_id), None)
        
        prompt = get_refine_hypothesis_prompt(state, target_hypothesis)
        new_hyps: List[Dict[str, Any]] = []
        try:
            react_llm = ChatOpenAI(model="gpt-4.1")
            react_agent = create_react_agent(
                model=react_llm,
                tools=[],
                response_format=HypothesisList,
            )
            resp_data = await react_agent.ainvoke({"messages": [("human", prompt)]})
            raw_hypotheses = dict(resp_data["structured_response"]).get("hypotheses", [])
            for i, hypo in enumerate(raw_hypotheses):
                hypo = dict(hypo)
                # refineの場合はidをsubref_+元仮説id+連番に
                hypo["id"] = f"subref_{hypothesis_id}_{i+1}"
                hypo["parent_hypothesis_id"] = hypothesis_id
                if hypo.get("status") != "new":
                    hypo["status"] = "new"
                new_hyps.append(hypo)
        except Exception as e:
            logger.warning(f"LLM failed during refine hypothesis generation, fallback: {e}")

        if not new_hyps:
            fallback_hypo = {
                "id": f"fb_ref_{str(uuid.uuid4())[:8]}",
                "text": f"{target_hypothesis['text']} を見直したサブ仮説",
                "priority": 0.5,
                "status": "new",
                "parent_hypothesis_id": hypothesis_id,
                "supporting_evidence_keys": [],
                "next_validation_step_suggestion": "",
                "metric_definition": ""
            }
            new_hyps.append(fallback_hypo)

        updated_hypotheses = existing_hypotheses + new_hyps
        patch = {
            "current_hypotheses": updated_hypotheses,
            "next_action": None,
        }

        events = [make_history_entry(
            "node",
            {
                "name": "refine_hypothesis",
                "parent_id": hypothesis_id,
                "count": len(new_hyps),
                "generated_hypotheses": [hypo["id"] for hypo in new_hyps],
            },
            state
        )]

        return NodeResult(observation=new_hyps, patch=patch, events=events) 