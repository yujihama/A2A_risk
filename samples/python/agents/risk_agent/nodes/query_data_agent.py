import logging
import random
from typing import Any, Dict

import pandas as pd

from ..core.node_base import Node, NodeResult, make_history_entry
from ..agent import execute_step, PlanStep
from ..prompts import get_query_data_first_step_prompt, get_query_data_step_prompt
from ..prompts import get_query_refinement_prompt, get_query_result_prompt, get_query_data_react_prompt, get_query_data_analysis_prompt

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

import json

logger = logging.getLogger(__name__)

async def call_data_agent(skill_id,query):
    """
    指定されたスキルのデータエージェントからデータを取得する
    Args:
        skill_id: クエリを渡すスキルのID
        query: データエージェントに渡すクエリ（重要：クエリは複雑すぎず、シンプルなデータ取得にしてください）
    Returns:
        str: データエージェントから取得したデータ
    
    """
    logger.info(f"[QDA] call_data_agent skill_id: {skill_id}, query: {query}")

    step = PlanStep(
        id=f"step_{skill_id}",
        description=f"データエージェントへのクエリ: {str(query)}",
        skill_id=skill_id,
        input_data={"input": str(query)},
        parameters={}
    )
    try:
        step_result = await execute_step(step)
    except Exception as e:
        logger.error(f"[QDA] call_data_agent error: {e}")
        step_result = {"error": "データエージェントからデータを取得できませんでした。クエリを変更して再度実行してください。"}

    logger.info(f"[QDA] call_data_agent result: {step_result}")

    return step_result

async def analyze_data(query, output_from_data_agent):
    """
    データエージェントから取得したデータを使ってデータ分析をして、分析結果を返す
    Args:
        query: データエージェントから取得したデータを使って何を分析するかを示す。
        output_from_data_agent: データエージェントから取得したデータ（複数ある場合はdict形式で、それぞれが何のデータかを示す）
    Returns:
        str: データ分析結果
    """
    logger.info(f"[QDA] analyze_data: {query[:50]}")

    tool_llm = ChatOpenAI(model="o3-mini", temperature=0)
    prompt_for_data_analysis = get_query_data_analysis_prompt(query, output_from_data_agent)

    analysis_result = await tool_llm.ainvoke(prompt_for_data_analysis)

    logger.info(f"[QDA] analyze_data result: {analysis_result}")

    return analysis_result


class QueryDataAgentNode(Node):
    """データ取得ノード

    `last_query` または `next_action.parameters` にある `agent_skill_id` / `query` を使って外部データエージェント
    （未実装の場合はダミー）からデータを収集し、要約を `collected_data_summary` に格納する。
    """

    id = "query_data_agent"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: QueryDataAgent ---")

        # パラメータ解析
        params = state.get("next_action", {}).get("parameters", {})
        query_list = params.get("query") 
        if not query_list:
            logger.warning("query_list is empty")
        
        # 現在フォーカスしている仮説 ID（無い場合は None）
        hyp_id = state.get("currently_investigating_hypothesis_id")

        data_summary: Dict[str, Any] = {}

        prompt_for_query = get_query_data_react_prompt(state,query_list)
        react_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        react_agent = create_react_agent(
            model=react_llm, 
            tools=[call_data_agent, analyze_data],
            prompt="ステップバイステップでcall_data_agentツールを使ってエージェントからデータを取得し、結果を回答してください。必要に応じて、analyze_dataツールを使って分析を行い結果を回答してください。データ取得手順ではなく、必ずデータ取得結果を回答してください。",

        )
        # agent_executor を使用して呼び出す
        resp_data = await react_agent.ainvoke({"messages": [("human", prompt_for_query)]})
        
        data_summary = resp_data["messages"][-1].content
        logger.info(f"[QDA] data_summary: {data_summary}")
    
        # 既存の collected_data_summary を仮説 ID ごとに更新
        current_summary = state.get("collected_data_summary", {})
        updated_summary = current_summary.copy() if isinstance(current_summary, dict) else {}
        if hyp_id:
            if hyp_id not in updated_summary:
                updated_summary[hyp_id] = []
            updated_summary[hyp_id].append(data_summary)
        else:
            # フォーカスが無い場合は __global__ に退避
            updated_summary["__global__"] = data_summary

        next_action = None
        if hyp_id:
            next_action = {
                "action_type": "evaluate_hypothesis",
                "parameters": {"hypothesis_id": hyp_id},
            }

        patch = {
            "hyp_id": hyp_id,
            "collected_data_summary": updated_summary,
            "next_action": next_action
        }

        events = [make_history_entry(
            "node",
            {
                "name": "query_data_agent",
                "query": query_list,
                "process":[step.content for step in resp_data["messages"]]
            },
            state
        )]

        return NodeResult(observation=data_summary, patch=patch, events=events) 