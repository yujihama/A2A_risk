import logging
import random
from typing import Any, Dict

import pandas as pd

from ..core.node_base import Node, NodeResult
from ..agent import execute_step, PlanStep
from ..prompts import get_query_data_first_step_prompt, get_query_data_step_prompt
from ..prompts import get_query_refinement_prompt, get_query_result_prompt

logger = logging.getLogger(__name__)


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

        iteration = 0

        data_summary: Dict[str, Any] = {}
        summary_text = None
        data_plan_list = []

        while iteration < 10:
            if iteration == 0:
                prompt_for_query = get_query_data_first_step_prompt(state,query_list)
                logger.info(f"[QDA] LLM first step prompt: {prompt_for_query}")
                data_plan = await toolbox.llm.ainvoke(prompt_for_query)
                logger.info(f"[QDA] LLM first step response: {data_plan}")

            else:
                prompt_for_query = get_query_data_step_prompt(state,query_list,data_plan_list)
                # logger.info(f"[QDA] LLM step prompt: {prompt_for_query}")
                data_plan = await toolbox.llm.ainvoke(prompt_for_query)
                logger.info(f"[QDA] LLM step response: {data_plan}")
                
            skill_id = data_plan.get("skill_id", "")
            query = data_plan.get("query", "")
            if skill_id == "" and query == "":
                break
        
            logger.info("Performing data query via skill_id=%s query=%s (hyp_id=%s)", skill_id, query, hyp_id)

            # --- クエリリファインメント（複数クエリ抽出） ---
            refined_query_data = None
            required_data_list = []
            if toolbox.llm:
                try:
                    prompt_template = get_query_refinement_prompt(query)
                    refined_query_data = await toolbox.llm.ainvoke(prompt_template)
                    logger.info(f"[QDA] query細分化LLM出力: {refined_query_data}")
                    if isinstance(refined_query_data, dict) and "answer" in refined_query_data:
                        for pattern in refined_query_data["answer"]:
                            step_id = pattern.get("step_id", "")
                            required_data = pattern.get("required_data", {}).get("new", {})
                            if required_data != {}:
                                found = False
                                for item in required_data_list:
                                    if item["required_data"] == required_data:
                                        if isinstance(item["step_id"], list):
                                            item["step_id"].append(step_id)
                                        else:
                                            item["step_id"] = [item["step_id"], step_id]
                                        found = True
                                        break
                                if not found:
                                    required_data_list.append({"step_id": step_id, "required_data": required_data})
                    logger.info(f"[QDA] refined required_data_list: {required_data_list}")
                except Exception as e:
                    logger.warning(f"[QDA] クエリリファインメント失敗: {e}")

            # --- データ取得・実行（複数クエリ対応） ---
            
            if required_data_list:
                try:
                    result_text = []
                    raw_data = []
                    for required_data in required_data_list:
                        step = PlanStep(
                            id=f"step_{skill_id}_{required_data['step_id']}",
                            description=f"データエージェントへのクエリ: {str(required_data['required_data'])}",
                            skill_id=skill_id,
                            input_data={"input": str(required_data['required_data']), "return_type": "df"},
                            parameters=params
                        )
                        step_result = await execute_step(step)
                        text = None
                        data = None
                        if hasattr(step_result, 'output_parts') and step_result.output_parts:
                            for part in step_result.output_parts:
                                if hasattr(part, 'type') and part.type == 'text':
                                    text = getattr(part, 'text', None)
                                elif hasattr(part, 'type') and part.type == 'data':
                                    data = getattr(part, 'data', None)
                        elif isinstance(step_result.output_data, dict):
                            text = step_result.output_data.get("text", "")
                            data = step_result.output_data.get("data", None)
                        else:
                            text = str(step_result.output_data)
                            data = None
                        result_text.append(text)
                        raw_data.append({"step_id": required_data['step_id'], "data": data})
                    # LLMでresult_textを要約
                    prompt_template = get_query_result_prompt(query, refined_query_data, result_text) #get_query_result_summary_prompt(result_text, required_data_list)から変更
                    # logger.info(f"[QDA] LLM data analysis prompt: {prompt_template}")
                    result = await toolbox.llm.ainvoke(prompt_template)
                    # logger.info(f"[QDA] LLM data analysis result: {result}")
                    data_summary = {f"text": result_text, "raw_data": raw_data}
                    data_plan["result"] = result
                except Exception as e:
                    logger.warning(f"execute_step failed (multi): {e}")
                    data_summary = {"error": str(e)}
                    data_plan["result"] = data_summary
                finally:
                    data_plan_list.append(data_plan)
                    iteration += 1

    
        # 既存の collected_data_summary を仮説 ID ごとに更新
        current_summary = state.get("collected_data_summary", {})
        updated_summary = current_summary.copy() if isinstance(current_summary, dict) else {}
        if hyp_id:
            updated_summary[hyp_id] = data_summary  # 仮説ごとに上書き／保存
        else:
            # フォーカスが無い場合は __global__ に退避
            updated_summary["__global__"] = data_summary

        patch = {
            "hyp_id": hyp_id,
            "collected_data_summary": updated_summary,
            "next_action": {"action_type": "data_analysis", "parameters": {"query": query_list, "data_plan_list": data_plan_list}},
        }

        events = [{"type": "node", "name": "query_data_agent", "query": query, "refined_query": data_plan_list if data_plan_list else query}]

        return NodeResult(observation=data_summary, patch=patch, events=events) 