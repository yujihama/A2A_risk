import logging
import json
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from ..core.node_base import Node, NodeResult, make_history_entry
from ..prompts import get_initial_data_query_prompt

logger = logging.getLogger(__name__)

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

class EDANode(Node):
    """探索的データ分析(EDA)ノード

    - `collected_data_summary` が空なら、LLM に対して初期データ取得タスクを生成させる
    - データがあれば pandas で簡易統計を計算し summary を生成
    - いずれの場合も `data_collected=True` を立てて次ノード( query_data_agent )へ遷移させる
    """

    id = "eda"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: EDA ---")

        data_summary: Dict[str, Any] = state.get("collected_data_summary", {})
        objective: str = state.get("objective", "No objective provided")
        available_agents: List[Dict[str, Any]] = state.get("available_data_agents_and_skills", [])

        patch: Dict[str, Any] = {
            # ルーティング用
            "data_collected": True,
        }

        events = [make_history_entry("node", {"name": "eda"}, state)]

        # データが無い場合は LLM に問い合わせ
        if not data_summary:
            logger.info("No data present. Ask LLM to propose initial data query.")

            prompt_template = get_initial_data_query_prompt(objective, available_agents)

            agent_skill_id = None
            query = None
            decision_data = None
            if toolbox.llm:
                try:
                    decision_data = await toolbox.llm.ainvoke(prompt_template)
                    logger.info(f"[EDA] LLM出力: {decision_data}")
                    if isinstance(decision_data, dict):
                        agent_skill_id = decision_data.get("agent_skill_id") or decision_data.get("skill_id")
                        query = decision_data.get("query")
                except Exception as e:  # noqa: BLE001
                    logger.warning("LLM failed to generate initial query: %s", e)

            if not (agent_skill_id and query):
                # フォールバック: ダミークエリ
                agent_skill_id = "default_agent"
                query = "SELECT * FROM dummy LIMIT 10"
                logger.info("Fallback initial query generated.")

            # skill_idの妥当性チェック
            valid_skill_ids = set()
            for agent_card in available_agents:
                try:
                    # dict 形式
                    if isinstance(agent_card, dict):
                        sid = agent_card.get('skill_id') or agent_card.get('id')
                        if sid:
                            valid_skill_ids.add(sid)
                        continue
                    # pydantic AgentCard オブジェクト
                    if hasattr(agent_card, 'skills'):
                        for sk in agent_card.skills:
                            sid = getattr(sk, 'id', None) or getattr(sk, 'skill_id', None)
                            if sid:
                                valid_skill_ids.add(sid)
                except Exception:
                    continue
            if agent_skill_id not in valid_skill_ids:
                logger.warning(f"LLM generated an unknown agent_skill_id: {agent_skill_id}. Falling back to error.")
                agent_skill_id = "default_agent"
                query = "SELECT * FROM dummy LIMIT 10"

            logger.info(f"[EDA] アクション決定: query_data_agent, skill_id={agent_skill_id}, query={query}")
            patch.update({
                "eda_summary": "EDA要約: データが存在しません。",  # 日本語固定で可
                "eda_stats": {},
                "last_query": query,
            })
            # next_actionをセット
            patch['next_action'] = {
                'action_type': 'query_data_agent',
                'parameters': {
                    'agent_skill_id': agent_skill_id,
                    'query': query
                }
            }

        else:
            # 簡易統計を pandas describe で生成
            logger.info("Data summary present. Computing statistics via pandas.")
            summary_lines: List[str] = []
            summary_dict: Dict[str, Any] = {}

            for key, value in data_summary.items():
                if isinstance(value, pd.DataFrame):
                    df = value
                elif isinstance(value, list):
                    df = pd.DataFrame(value)
                elif isinstance(value, dict):
                    df = pd.DataFrame([value])
                else:
                    summary_lines.append(f"{key}: Unsupported type {type(value)}")
                    continue

                if df.empty:
                    summary_lines.append(f"{key}: DataFrame is empty")
                    continue

                desc = df.describe(include="all").T
                null_pct = df.isnull().mean() * 100

                summary_lines.append(f"{key}: Rows={len(df)}")
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    mean_val = desc.loc[col, "mean"] if "mean" in desc.columns else "N/A"
                    std_val = desc.loc[col, "std"] if "std" in desc.columns else "N/A"
                    missing = null_pct[col]
                    summary_lines.append(
                        f"  - {col} (Type:{col_type}) Mean={mean_val} Std={std_val} Missing={missing:.1f}%"
                    )
                    summary_dict[f"{key}.{col}"] = {
                        "mean": mean_val if pd.notna(mean_val) else None,
                        "std": std_val if pd.notna(std_val) else None,
                        "missing_pct": missing,
                        "dtype": col_type,
                    }

            patch.update({
                "eda_summary": "\n".join(summary_lines),
                "eda_stats": summary_dict,
            })

            # データがある場合は next_action は設定しない (DecisionMaker に判断を委ねる)
            patch['next_action'] = None # None を明示するか、この行自体を削除

        logger.info("[EDA] patch keys: %s", list(patch.keys()))

        # --- 追加: patch 内の NumPy 型を Python ネイティブ型に変換 --- 
        converted_patch = convert_numpy_types(patch)
        # --- ここまで追加 ---

        # 変換後の patch を返す
        return NodeResult(observation=converted_patch.get("eda_summary", "EDA done"), patch=converted_patch, events=events) 