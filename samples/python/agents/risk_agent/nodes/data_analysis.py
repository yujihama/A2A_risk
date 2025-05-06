import logging
from typing import Any, Dict

import pandas as pd
import numpy as np

from ..core.node_base import Node, NodeResult, make_history_entry
from ..prompts import get_query_data_analysis_prompt  

logger = logging.getLogger(__name__)

# --- ヘルパー関数を外に定義 ---
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    return data
# --------------------------

class DataAnalysisNode(Node):
    id = "data_analysis"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: DataAnalysis ---")

        # 必要な情報を取得
        query = state.get("next_action", {}).get("parameters", {}).get("query", "")
        data_plan_list = state.get("next_action", {}).get("parameters", {}).get("data_plan_list", [])

        # 現在フォーカスしている仮説 ID
        hyp_id = state.get("currently_investigating_hypothesis_id")

        # 仮説 ID ごとのデータに絞り込む (無い場合は空辞書)
        collected = state.get("collected_data_summary", {})
        data = collected.get(hyp_id) if hyp_id else collected.get("__global__", {})

        analysis: Any = convert_numpy_types({ # デフォルト値も変換
            "note": "analysis not executed",
            "input_query": query,
            "data_plan_list": data_plan_list,
        })

        # LLM が利用可能なら詳細分析を依頼
        if toolbox.llm:
            try:
                prompt = get_query_data_analysis_prompt(query, data_plan_list)
                resp = await toolbox.llm.ainvoke(prompt)
                # --- LLM 応答直後に変換 ---
                analysis = convert_numpy_types(resp) if resp else analysis
                # ------------------------
                logger.info(f"[DataAnalysis] analysis: {analysis['reasoning']}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM data analysis failed: %s", exc)
                # エラー時も analysis がシリアライズ可能であることを保証
                analysis = convert_numpy_types(analysis)

        # --- 次アクション判定 ---
        next_action = None
        if hyp_id:
            next_action = {
                "action_type": "evaluate_hypothesis",
                "parameters": {"hypothesis_id": hyp_id},
            }

        # --- state から読み込んだ既存の analysis_result も変換 ---
        current_analysis = state.get("analysis_result", {})
        updated_analysis = convert_numpy_types(
            current_analysis.copy() if isinstance(current_analysis, dict) else {}
        )
        # ----------------------------------------------------
        if hyp_id:
            updated_analysis[hyp_id] = analysis # 変換済みの analysis を代入
        else:
            updated_analysis["__global__"] = analysis # 変換済みの analysis を代入

        # patch 作成 (中の値は既に変換済みのはず)
        patch = {
            "hyp_id": hyp_id,
            "analysis_result": updated_analysis,
            "next_action": next_action,
        }
        # 以前の変換コードは削除
        # def convert_numpy_types(data): ...
        # converted_patch = convert_numpy_types(patch)

        # events 作成 (中の analysis は変換済み)
        events = [
            make_history_entry("node", {"name": "data_analysis"}, state),
            make_history_entry("observation", {"hypothesis_id": hyp_id, "data_analysis_result": analysis}, state)
        ]
        # observation にも変換済みの analysis を渡す
        return NodeResult(observation=analysis, patch=patch, events=events) 