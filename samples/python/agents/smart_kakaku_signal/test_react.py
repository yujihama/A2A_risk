import datetime
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional, Literal
from functools import partial

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

import asyncio
import sys
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_INCONCLUSIVE_BEFORE_GENERATE = 2
STAGNATION_THRESHOLD = 0.8
SIM_THRESHOLD = 3
EVAL_REPEAT_LIMIT = 10

class Hypothesis(TypedDict):
    id: str
    text: str
    priority: float
    status: Literal["new", "investigating", "supported", "rejected", "needs_revision", "inconclusive"]
    evaluation_reasoning: Optional[str] = None
    required_next_data: Optional[str] = None
    complexity_score: Optional[float] = None

class NextAction(TypedDict):
    action_type: Literal[
        "query_data_agent",
        "generate_hypothesis",
        "evaluate_hypothesis",
        "refine_plan",
        "execute_plan_step",
        "conclude",
        "error",
        "refine_hypothesis",
        "triangulate_data",
        "escalate_to_expert",
        "sample_and_test"
    ]
    parameters: Dict[str, Any]

class HistoryEntry(TypedDict):
    type: Literal["thought", "action", "observation"]
    content: Any
    timestamp: str

class DynamicAgentState(TypedDict):
    objective: str
    history: List[HistoryEntry]
    current_hypotheses: List[Hypothesis]
    collected_data_summary: Dict[str, Any]
    active_plan: Optional[Dict]
    next_action: Optional[NextAction]
    final_result: Optional[Dict]
    available_actions: List[Dict]
    available_data_agents_and_skills: List[Dict]
    error_message: Optional[str]
    max_iterations: int
    current_iteration: int
    data_points_collected: int
    cycles_since_last_hypothesis: int
    max_queries_without_hypothesis: int
    consecutive_query_count: int
    currently_investigating_hypothesis_id: Optional[str]
    eval_repeat_count: int
    eda_summary: Optional[str]
    eda_stats: Optional[Dict]

def _all_hypotheses_resolved(hypotheses):
    # --- ▼▼▼ 修正: 仮説リストが空の場合は False を返す ▼▼▼ ---
    if not hypotheses:
        return False
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---
    return all(h['status'] in ['supported', 'rejected'] for h in hypotheses)

def _unresolved_ratio(hypotheses):
    if not hypotheses:
        return 0.0
    unresolved = [h for h in hypotheses if h.get('status') not in ['supported', 'rejected']]
    return len(unresolved) / len(hypotheses)

def _count_inconclusive(hypotheses):
    return sum(1 for h in hypotheses if h['status'] in ['inconclusive', 'needs_revision'])

def _summarize_hypotheses(hypotheses):
    return [
        {
            "id": h.get("id"),
            "status": h.get("status"),
            "priority": h.get("priority"),
        }
        for h in hypotheses
    ]

async def decision_maker_node(state: DynamicAgentState, llm: ChatOpenAI) -> DynamicAgentState:
    logger.info("--- Node: Decision Maker ---")
    logger.info(
        f"[DM] 入力state: iter={state.get('current_iteration')}/{state.get('max_iterations')}, "
        f"focus={state.get('currently_investigating_hypothesis_id')}, "
        f"hypotheses_summary={_summarize_hypotheses(state.get('current_hypotheses', []))}"
    )
    current_iteration = state.get('current_iteration', 0)
    max_iterations = state.get('max_iterations', 10)
    max_queries_without_hypothesis = state.get('max_queries_without_hypothesis', 2)
    cycles_since_last_hypothesis = state.get('cycles_since_last_hypothesis', 0)
    consecutive_query_count = state.get('consecutive_query_count', 0)
    current_hypotheses = state.get('current_hypotheses', [])
    currently_investigating_hypothesis_id = state.get('currently_investigating_hypothesis_id')
    eval_repeat_count = state.get('eval_repeat_count', 0)
    collected_data_summary = state.get('collected_data_summary', {})
    eda_stats = state.get('eda_stats', {})

    # --- ▼▼▼ 修正: 前のノードでアクションが決定済みかチェック ▼▼▼ ---
    predefined_action = state.get('next_action')
    if predefined_action:
        # pre_node_name = state.get('history',[])[-1].get('type','unknown') if state.get('history') else 'unknown'
        # logger.info(f"Previous node ({pre_node_name}?) already set the next action to: {predefined_action.get('action_type')}. DM skips its logic.")
        logger.info(f"Previous node already set the next action to: {predefined_action.get('action_type')}. DM skips its logic.")
        # current_iteration は前のノードでインクリメントされている場合があるので、ここでは操作しない
        # state['next_action'] も変更しない
        return state # 既存の next_action を維持して Action Executor へ
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

    # --- ステータス重み付け ---
    status_weight = {'needs_revision': 0, 'inconclusive': 1, 'new': 2}

    # --- 改修: リスクスコア関数（eda_stats > collected_data_summary > 0） ---
    def risk_score(h):
        coverage = 0
        if h['id'] in eda_stats and isinstance(eda_stats[h['id']], dict):
            coverage = eda_stats[h['id']].get('coverage_pct', 0)
        elif h['id'] in collected_data_summary and isinstance(collected_data_summary[h['id']], dict):
            coverage = collected_data_summary[h['id']].get('coverage_pct', 0)
        return h.get('priority', 0) * (1 - coverage)

    # --- 共通: ステータス重み+リスクスコアで降順 ---
    def select_focus_candidate(hypos):
        return sorted(
            hypos,
            key=lambda h: (status_weight.get(h['status'], 99), -risk_score(h)),
        )

    unresolved_ratio = _unresolved_ratio(current_hypotheses)
    stagnation_score = (current_iteration / max_iterations) * unresolved_ratio + (cycles_since_last_hypothesis / SIM_THRESHOLD)
    logger.info(
        f"[Conclude判定直前] iter={current_iteration}/{max_iterations}, "
        f"unresolved_ratio={unresolved_ratio:.3f}, stagnation_score={stagnation_score:.3f}, "
        f"cycles_since_last_hypothesis={cycles_since_last_hypothesis}, "
        f"focus={state.get('currently_investigating_hypothesis_id')}"
    )
    logger.info(f"[Conclude判定直前] hypotheses_summary={_summarize_hypotheses(current_hypotheses)}")
    if _all_hypotheses_resolved(current_hypotheses):
        logger.info("全仮説が解決済み。自動concludeを発火します。")
        action = {'action_type': 'conclude', 'parameters': {}}
        logger.info(f"Next action: {action['action_type']}")
        state['next_action'] = action
        state['current_iteration'] = current_iteration + 1
        return state
    if stagnation_score >= STAGNATION_THRESHOLD:
        logger.info(f"stagnation_score={stagnation_score:.2f} が閾値({STAGNATION_THRESHOLD})を超過。現在の仮説のみ'rejected'にして次へ進みます。")
        if currently_investigating_hypothesis_id:
            for h in current_hypotheses:
                if h['id'] == currently_investigating_hypothesis_id and h['status'] not in ['supported', 'rejected']:
                    h['status'] = 'needs_revision'
                    h['evaluation_reasoning'] = f"stagnation_scoreが閾値({STAGNATION_THRESHOLD})を超過したためneeds_revisionに設定。"
                    logger.info(f"仮説 {currently_investigating_hypothesis_id} をstagnationによりneeds_revisionに設定。")
                    break
        state['currently_investigating_hypothesis_id'] = None
        unresolved = [h for h in current_hypotheses if h['status'] not in ['supported', 'rejected']]
        if unresolved:
            next_focus = select_focus_candidate(unresolved)[0]
            state['currently_investigating_hypothesis_id'] = next_focus['id']
            logger.info(f"次の仮説 {next_focus['id']} (Priority: {next_focus.get('priority')}) にフォーカスを移します。")
            action = {
                "action_type": "evaluate_hypothesis",
                "parameters": {"hypothesis_id": next_focus['id']}
            }
            logger.info(f"Next action: {action['action_type']}")
            thought = f"stagnation_score超過により前仮説をrejected。次の仮説 {next_focus['id']} にフォーカスし評価を開始。"
            state['next_action'] = action
            state['history'] = state.get('history', []) + [{
                'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()
            }]
            state['current_iteration'] = current_iteration + 1
            return state
        else:
            logger.info("全仮説が解決済み（またはrejected）。自動concludeを発火します。")
            action = {'action_type': 'conclude', 'parameters': {}}
            logger.info(f"Next action: {action['action_type']}")
            state['next_action'] = action
            state['current_iteration'] = current_iteration + 1
            return state

    if current_iteration >= max_iterations:
        logger.warning(f"最大イテレーション数 ({max_iterations}) に到達しました。強制終了します。")
        action = {'action_type': 'error', 'parameters': {'message': 'Max iterations reached'}}
        logger.info(f"Next action: {action['action_type']}")
        state['next_action'] = action
        state['error_message'] = 'Max iterations reached'
        return state

    focused_hypothesis = None
    focus_candidates = [h for h in current_hypotheses if h['status'] in ['needs_revision', 'inconclusive', 'new'] and not h.get('required_next_data')]
    focus_candidates = select_focus_candidate(focus_candidates)
    if currently_investigating_hypothesis_id:
        focused_hypothesis = next((h for h in current_hypotheses if h['id'] == currently_investigating_hypothesis_id), None)
        if not focused_hypothesis or focused_hypothesis['status'] in ['supported', 'rejected']:
            if focused_hypothesis:
                 logger.info(f"Focus hypothesis {currently_investigating_hypothesis_id} is already {focused_hypothesis['status']}. Clearing focus.")
            else:
                 logger.warning(f"Focus hypothesis {currently_investigating_hypothesis_id} not found during check. Clearing focus.")
            currently_investigating_hypothesis_id = None
            state['currently_investigating_hypothesis_id'] = None
            focused_hypothesis = None
    if not focused_hypothesis and focus_candidates:
        hypothesis_to_focus = focus_candidates[0]
        currently_investigating_hypothesis_id = hypothesis_to_focus['id']
        state['currently_investigating_hypothesis_id'] = currently_investigating_hypothesis_id
        focused_hypothesis = hypothesis_to_focus
        logger.info(f"No focus. Setting focus to: {currently_investigating_hypothesis_id} (Priority: {hypothesis_to_focus.get('priority')})")
        action = {
            "action_type": "evaluate_hypothesis",
            "parameters": {"hypothesis_id": currently_investigating_hypothesis_id}
        }
        logger.info(f"Next action: {action['action_type']}")
        thought = f"Setting focus on new or nearly-resolved hypothesis {currently_investigating_hypothesis_id} and initiating evaluation."
        state['next_action'] = action
        state['history'] = state.get('history', []) + [{
            'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()
        }]
        state['current_iteration'] = current_iteration + 1
        return state

    inconclusive_count = _count_inconclusive(current_hypotheses)
    # --- ▼▼▼ 修正: inconclusive が続く場合の強制 refine/generate ロジック ▼▼▼ ---
    force_refine_or_generate = False
    if cycles_since_last_hypothesis >= MAX_INCONCLUSIVE_BEFORE_GENERATE or inconclusive_count >= MAX_INCONCLUSIVE_BEFORE_GENERATE:
        # フォーカス中の仮説があり、かつ必要なデータが指定されている場合は、LLMの判断に任せる
        if focused_hypothesis and focused_hypothesis.get('required_next_data'):
             logger.info(f"Inconclusive count ({inconclusive_count}) or cycles ({cycles_since_last_hypothesis}) reached threshold, but required_next_data exists for focus {focused_hypothesis['id']}. Proceeding to LLM decision.")
             pass # LLM判断ロジックへ進む
        else:
            # 必要なデータが不明な inconclusive が続く場合や、フォーカスがない場合に強制実行
            force_refine_or_generate = True
            logger.warning(f"cycles_since_last_hypothesis={cycles_since_last_hypothesis}, inconclusive_count={inconclusive_count} で refine_hypothesis/generate_hypothesis を強制します。(No focus or no required_next_data)")

    if force_refine_or_generate:
        action = None # action を初期化
        thought = None # thought を初期化
        if focused_hypothesis:
            # focused_hypothesis が inconclusive/needs_revision で required_next_data がない場合のみ refine を強制
            if focused_hypothesis['status'] in ['inconclusive', 'needs_revision'] and not focused_hypothesis.get('required_next_data'):
                 action = {'action_type': 'refine_hypothesis', 'parameters': {'hypothesis_id': focused_hypothesis['id']}}
                 logger.info(f"Next action: {action['action_type']} (Forced due to inconclusive/no required data)")
                 thought = f"Forcing 'refine_hypothesis' for {focused_hypothesis['id']} due to persistent inconclusive status without clear required data."
            else:
                 # required_next_dataがある場合や他のステータス場合はLLMに任せる
                 logger.info(f"Force refine condition met, but focused hypothesis {focused_hypothesis['id']} has required_next_data or is not inconclusive/needs_revision. Proceeding to LLM decision.")
                 pass # LLM判断ロジックへ進む (action は None のまま)
        else:
            # フォーカスがない場合は generate_hypothesis を強制
            action = {'action_type': 'generate_hypothesis', 'parameters': {}}
            logger.info(f"Next action: {action['action_type']} (Forced due to inconclusive count/cycles with no focus)")
            thought = f"Forcing 'generate_hypothesis' due to inconclusive/needs_revision cycles or count (no focus)."

        # actionが決定された場合のみ state を更新して return
        if action:
             state['next_action'] = action
             if thought: # thought が設定されている場合のみ履歴に追加
                 state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
             state['current_iteration'] = current_iteration + 1
             # 強制実行後は cycles_since_last_hypothesis をリセットすべきか検討 (ここではリセットしない)
             # state['cycles_since_last_hypothesis'] = 0
             return state
        # actionが決定されなかった場合 (LLM判断に任せる場合) はそのまま下に流れる
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

    if not focused_hypothesis:
        if consecutive_query_count >= max_queries_without_hypothesis:
            logger.warning(f"query_data_agent が {consecutive_query_count} 回連続しました (no focus, no new)。generate_hypothesis を強制します。")
            action = {'action_type': 'generate_hypothesis', 'parameters': {}}
            logger.info(f"Next action: {action['action_type']}")
            thought = f"Forcing 'generate_hypothesis' due to {consecutive_query_count} consecutive queries without focus or new hypotheses."
            state['next_action'] = action
            state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
            state['current_iteration'] = current_iteration + 1
            return state
        logger.info("Attempting to generate new hypotheses as no 'new' ones exist.")
        action = {'action_type': 'generate_hypothesis', 'parameters': {}}
        logger.info(f"Next action: {action['action_type']}")
        thought = "No focus and no 'new' hypotheses. Attempting to generate new ones."
        state['next_action'] = action
        state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
        state['current_iteration'] = current_iteration + 1
        return state

    if focused_hypothesis:
        logger.info(f"Continuing focus on hypothesis: {currently_investigating_hypothesis_id} (Status: {focused_hypothesis['status']})")
        last_action = None
        if state.get('history'):
            for entry in reversed(state['history']):
                if entry['type'] == 'action':
                    last_action = entry['content']
                    break
        if last_action and last_action.get('action_type') == 'evaluate_hypothesis' and last_action.get('parameters', {}).get('hypothesis_id') == currently_investigating_hypothesis_id:
            eval_repeat_count += 1
        else:
            eval_repeat_count = 1
        state['eval_repeat_count'] = eval_repeat_count
        if eval_repeat_count >= EVAL_REPEAT_LIMIT:
            logger.warning(f"evaluate_hypothesis が同一仮説 {currently_investigating_hypothesis_id} で {eval_repeat_count} 回連続しました。フォーカス解除し generate_hypothesis へ。")
            action = {'action_type': 'generate_hypothesis', 'parameters': {}}
            logger.info(f"Next action: {action['action_type']}")
            state['currently_investigating_hypothesis_id'] = None
            state['eval_repeat_count'] = 0
            thought = f"Loop guard: Too many repeated evaluations for {currently_investigating_hypothesis_id}. Forcing generate_hypothesis."
            state['next_action'] = action
            state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
            state['current_iteration'] = current_iteration + 1
            return state
        # --- inconclusive/needs_revision時の分岐強化 ---
        if focused_hypothesis['status'] in ['inconclusive', 'needs_revision']:
            # 最新Observationを取得
            last_obs = None
            for entry in reversed(state.get('history', [])):
                if entry['type'] == 'observation':
                    last_obs = entry['content']
                    break
            action = None
            thought = None
            if last_obs and isinstance(last_obs, dict):
                agent_skill = last_obs.get('agent_skill_id')
                # Observationにエラーやescalateワードがあればエスカレーション
                obs_str = str(last_obs)
                if 'error' in obs_str.lower() or 'escalate' in obs_str.lower():
                    action = {'action_type': 'escalate_to_expert', 'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}}
                    thought = f"Observationにエラー/エスカレート指示が含まれるためescalate_to_expertを選択"
                elif agent_skill in ['query_expense', 'query_payment', 'query_inventory']:
                    action = {'action_type': 'triangulate_data', 'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}}
                    thought = f"Observationが支払/在庫/経費系なのでtriangulate_dataを選択"
                elif agent_skill == 'sample_test':
                    action = {'action_type': 'sample_and_test', 'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}}
                    thought = f"Observationがサンプル系なのでsample_and_testを選択"
                # else: # ★削除★
                #     action = {'action_type': 'refine_hypothesis', 'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}}
                #     thought = f"Observation内容からrefine_hypothesisを選択"
            # else: # ★削除★
            #     action = {'action_type': 'refine_hypothesis', 'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}}
            #     thought = f"Observationがないのでrefine_hypothesisを選択"

            # 上記の分岐で action が決定された場合のみ state を更新して return
            if action:
                 logger.info(f"Next action (determined by observation analysis): {action['action_type']}")
                 state['next_action'] = action
                 if thought:
                      state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
                 state['current_iteration'] = current_iteration + 1
                 return state
            # action が決定されなかった場合は、下の LLM プロンプトによる判断に進む
            logger.info("No specific action determined by observation analysis for inconclusive/needs_revision status. Proceeding to LLM decision.")

        # --- 既存の LLM によるアクション決定分岐 ---
        objective = state['objective']
        history_for_prompt = json.dumps(state.get('history', [])[-3:], ensure_ascii=False, indent=2)
        data_summary_for_prompt = json.dumps(state.get('collected_data_summary', {}), ensure_ascii=False, indent=2)
        available_actions = state.get('available_actions', [])
        available_agents = state.get('available_data_agents_and_skills', [])
        data_points_collected = state.get('data_points_collected', 0)
        focused_hypothesis_for_prompt = json.dumps(focused_hypothesis, ensure_ascii=False, indent=2)
        prompt_template = f"""
### ROLE
あなたは経験豊富な内部監査人／調査エージェント AI です。仮説駆動型アプローチに従ってください。現在フォーカスしている仮説の検証を進めます。

### GOAL
{objective}

### CURRENT FOCUS (重要: 現在はこの仮説の検証に集中しています)
- CURRENTLY_INVESTIGATING_HYPOTHESIS: {focused_hypothesis_for_prompt}

### CONTEXT
- HISTORY_JSON: {history_for_prompt} # 直近の履歴。Observationに直前のデータ収集結果が含まれる点に注意。
- COLLECTED_DATA_SUMMARY_JSON: {data_summary_for_prompt}
- CYCLES_SINCE_LAST_HYPOTHESIS_ACTION: {cycles_since_last_hypothesis}
- DATA_POINTS_COLLECTED_IN_LAST_QUERY: {data_points_collected}
- CONSECUTIVE_QUERY_COUNT: {consecutive_query_count} # 参考情報

### AVAILABLE_RESOURCES
- ACTION_TYPES_JSON: {json.dumps(available_actions, ensure_ascii=False, indent=2)}
- DATA_AGENTS_JSON: {json.dumps(available_agents, ensure_ascii=False, indent=2)}

### ACTION-SELECTION POLICY (このポリシーに従ってください)
**Focus Hypothesis is {focused_hypothesis['status']}. Determine the next step:**
*   If focus_hypothesis.status == "new" → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
*   If focus_hypothesis.status ∈ {{"inconclusive", "needs_revision"}}:
            *   Look at the latest entry in HISTORY_JSON.
            *   If the latest entry is an Observation from 'query_data_agent' related to the focus_hypothesis → **Analyze if the collected data (in observation.content.result) is sufficient to evaluate the hypothesis.**
                *   If **sufficient** → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
                *   If **insufficient** → **must** determine **what specific information is still missing** based on the focus_hypothesis.text, focus_hypothesis.required_next_data, and the collected data. Then, select "query_data_agent" with a **new, more specific query** to obtain the missing information. Avoid repeating the exact same query.
            *   If the latest entry is NOT a relevant Observation → **must** select "query_data_agent" based on focus_hypothesis.required_next_data to get the initially requested data.
*   If focus_hypothesis.status == "investigating" → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
*   *Do not select "generate_hypothesis" or "conclude" while focusing on a hypothesis.*
*   **If the hypothesis is large or complex (complexity_score >= 0.7), you may select 'refine_plan' to break down the investigation into steps.**

### TASK
1. 上記 ACTION-SELECTION POLICY と状況に基づき、現在フォーカスしている仮説 ({currently_investigating_hypothesis_id}) の検証を進めるための *単一* の行動 (action_type) を選定し、必要パラメータを具体値で生成せよ。
2. 選定理由を "thought" フィールドに 1–3 文で要約せよ (POLICY をどう解釈したか、Focus の状態、履歴の考慮点を含めること)。
3. 失敗や不整合がある場合は action_type に "error" を指定せよ。
4. action_type に "evaluate_hypothesis" を選択する場合、parameters は {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }} の形式で出力せよ。
5. action_type に "query_data_agent" を選択する場合、parameters は {{ "agent_skill_id": "<データエージェントのID>", "query": "<具体的で新しい問い合わせ内容>" }} の形式で出力せよ。
6. action_type に "refine_plan" を選択する場合、parameters は {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }} の形式で出力せよ。
7. action_type に "execute_plan_step" を選択する場合、parameters は {{ "step_id": "<ステップID>" }} の形式で出力せよ。

### FEW-SHOT EXAMPLES (Focus: {currently_investigating_hypothesis_id})
Example 1: Status is inconclusive, **no recent relevant observation**.
{{
  "thought": "Currently focusing on '{currently_investigating_hypothesis_id}'. Its status is 'inconclusive' and it requires data 'XYZ' (from required_next_data field). The history does not show recent data collection for this. Policy requires querying the necessary data. Selecting 'query_data_agent'.",
  "action": {{
    "action_type": "query_data_agent",
    "parameters": {{ "agent_skill_id": "some_agent_skill", "query": "Retrieve XYZ data related to {currently_investigating_hypothesis_id} based on evaluation requirement." }}
  }}
}}
Example 2: Status is inconclusive, **recent observation contains data 'XYZ'**.
{{
  "thought": "Currently focusing on '{currently_investigating_hypothesis_id}', status 'inconclusive'. The latest history entry is an observation containing relevant data 'XYZ' for this hypothesis. Policy requires re-evaluating the hypothesis with the new data. Selecting 'evaluate_hypothesis' for {currently_investigating_hypothesis_id}.",
  "action": {{
    "action_type": "evaluate_hypothesis",
    "parameters": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}
  }}
}}
Example 3: Status inconclusive, **recent observation data is insufficient** (e.g., missing specific field 'ABC').
{{
  "thought": "Focusing on '{currently_investigating_hypothesis_id}', status 'inconclusive'. Recent observation provided some data, but analysis indicates field 'ABC' is still missing to fully evaluate. Policy requires querying for the missing specific information. Selecting 'query_data_agent' with a refined query.",
  "action": {{
    "action_type": "query_data_agent",
    "parameters": {{ "agent_skill_id": "some_agent_skill", "query": "Retrieve specific field 'ABC' details for the records related to {currently_investigating_hypothesis_id} obtained in the previous query." }}
  }}
}}
Example 4: Status is complex, select refine_plan.
{{
  "thought": "Currently focusing on '{currently_investigating_hypothesis_id}', which is large/complex (complexity_score >= 0.7). Policy allows breaking down the investigation. Selecting 'refine_plan'.",
  "action": {{
    "action_type": "refine_plan",
    "parameters": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}
  }}
}}

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{
  "thought": "<why this action is optimal for {currently_investigating_hypothesis_id}, referencing the policy, focus state, and latest history observation>",
  "action": {{
    "action_type": "<one_of: {', '.join([a['type'] for a in available_actions if a['type'] not in ['generate_hypothesis', 'conclude']] )}, refine_plan, execute_plan_step>",
    "parameters": {{<key_value_pairs_or_empty_object>}}
  }}
}}
        """
        logger.info(f"LLMに次の行動決定を依頼中... (Focus: {currently_investigating_hypothesis_id})")
        response = await llm.ainvoke(prompt_template)
        llm_output_text = response.content
        logger.info(f"[DM] LLM出力: {llm_output_text}")
        try:
            decision_data = json.loads(llm_output_text)
            thought = decision_data.get("thought", "No thought provided.")
            action = decision_data.get("action")
            if not action or "action_type" not in action:
                raise ValueError("LLM response missing 'action' or 'action_type'")
            if action.get('action_type') == 'evaluate_hypothesis':
                 llm_hyp_id = action.get('parameters', {}).get('hypothesis_id')
                 if llm_hyp_id != currently_investigating_hypothesis_id:
                     logger.warning(f"LLM decided to evaluate hypothesis {llm_hyp_id} but focus is on {currently_investigating_hypothesis_id}. Overriding to focus ID.")
                     action['parameters']['hypothesis_id'] = currently_investigating_hypothesis_id
            logger.info(f"Next action: {action.get('action_type')}")
            state['history'] = state.get('history', []) + [{
                'type': 'thought',
                'content': thought,
                'timestamp': datetime.datetime.now().isoformat()
            }]
            state['next_action'] = action
            action_type = action.get('action_type')
            params = action.get('parameters', {})
            logger.info(f"[DM] LLM thought: {thought}")
            logger.info(f"[DM] LLM判断: action_type={action_type}, parameters={params} (Focus: {currently_investigating_hypothesis_id})")
        except Exception as e:
            logger.error(f"Decision Maker LLM応答の解析エラー: {e}\nLLM Output: {llm_output_text} (Focus: {currently_investigating_hypothesis_id})")
            state['next_action'] = {'action_type': 'error', 'parameters': {'message': f'Failed to parse Decision Maker LLM response: {e}'}}
            state['error_message'] = 'Failed to parse Decision Maker LLM response'
            state['history'] = state.get('history', []) + [{
                'type': 'thought',
                'content': f"Error: Failed to determine next action due to parsing error. LLM Output: {llm_output_text}",
                'timestamp': datetime.datetime.now().isoformat()
            }]
        state['current_iteration'] = current_iteration + 1
        return state

    logger.error("Decision Maker reached an unexpected state (no focus and no action determined by logic).")
    action = {'action_type': 'error', 'parameters': {'message': 'Unexpected state in Decision Maker'}}
    logger.info(f"Next action: {action['action_type']}")
    state['next_action'] = action
    state['error_message'] = 'Unexpected state in Decision Maker'
    state['current_iteration'] = current_iteration + 1
    return state

async def action_executor_node(state: DynamicAgentState, llm: ChatOpenAI, smart_a2a_client, data_analyzer) -> DynamicAgentState:
    """
    決定されたアクションを実行し、関連するStateフィールドを更新するノード。
    仮説評価後にフォーカスを解除するロジックを追加。
    """
    logger.info("--- Node: Action Executor ---")
    logger.info(
        f"[AE] 入力state: iter={state.get('current_iteration')}/{state.get('max_iterations')}, "
        f"focus={state.get('currently_investigating_hypothesis_id')}, "
        f"hypotheses_summary={_summarize_hypotheses(state.get('current_hypotheses', []))}"
    )
    action_details = state.get('next_action')
    if not action_details:
        logger.error("Action Executor: No action defined in state.")
        state['history'] = state.get('history', []) + [{
            'type': 'observation',
            'content': {'error': 'No action specified by Decision Maker'},
            'timestamp': datetime.datetime.now().isoformat()
        }]
        return state
    action_type = action_details.get('action_type')
    parameters = action_details.get('parameters', {})
    observation = None
    error_occurred = False
    data_points_collected_this_run = 0
    currently_investigating_hypothesis_id = state.get('currently_investigating_hypothesis_id')

    if action_type != 'query_data_agent':
        state['consecutive_query_count'] = 0
    if action_type in ['generate_hypothesis', 'evaluate_hypothesis']:
        state['cycles_since_last_hypothesis'] = 0
    else:
        state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) + 1

    state['history'] = state.get('history', []) + [{
        'type': 'action',
        'content': action_details,
        'timestamp': datetime.datetime.now().isoformat()
    }]

    exec_log_msg = f"Executing action: {action_type}"
    if action_type == 'evaluate_hypothesis':
        exec_log_msg += f" for hypothesis_id={parameters.get('hypothesis_id')}"
    exec_log_msg += f" with params: {parameters}"
    if currently_investigating_hypothesis_id:
        exec_log_msg += f" (Focus: {currently_investigating_hypothesis_id})"
    logger.info(exec_log_msg)
    logger.info(f"[AE] 実行action_details: {action_details}")
    try:
        if action_type == "query_data_agent":
            agent_skill_id = parameters.get("agent_skill_id")
            query = parameters.get("query")
            logger.info(f"[AE] query_data_agent: skill_id={agent_skill_id}, query={query}")
            if not agent_skill_id or not query:
                raise ValueError("Missing 'agent_skill_id' or 'query' parameter for query_data_agent")

            logger.warning(f"query_data_agent(skill={agent_skill_id}): ダミー結果を返します")
            res_sample = llm.invoke(f"あなたはデータエージェントです。以下の問い合わせ内容に対する結果を生成してください。（ダミーの結果で良いですが、実際のデータかのように回答してください。またqueryの要求事項が多い、またはハイレベルな場合はあえて不備・不足のある回答も混ぜてください。）\\n\\n問い合わせ内容: {query}")
            dummy_result = res_sample.content
            observation = {'result': dummy_result, 'agent_skill_id': agent_skill_id, 'query': query}
            data_points_collected_this_run = len(dummy_result.split()) // 10
            logger.info(f"推定データポイント数: {data_points_collected_this_run}")

            state['consecutive_query_count'] = state.get('consecutive_query_count', 0) + 1
        elif action_type == "generate_hypothesis":
            eda_summary = state.get('eda_summary', None)
            eda_stats = state.get('eda_stats', {})
            # EDAサマリーは先頭5行のみ
            if eda_summary:
                eda_summary_lines = eda_summary.splitlines()
                eda_summary_head = "\n".join(eda_summary_lines[:5])
            else:
                eda_summary_head = None
            prompt_template = f"""
### ROLE
あなたは洞察力のあるデータアナリスト／リスク分析官 AI です。
### GOAL
新規または改訂リスク仮説を生成し、優先度づけせよ。
### CONTEXT
- OBJECTIVE: {state['objective']}
- EDA_SUMMARY_TEXT: {json.dumps(eda_summary_head, ensure_ascii=False, indent=2) if eda_summary_head else 'null'}
- EDA_STATS_JSON: {json.dumps(eda_stats, ensure_ascii=False, indent=2)}
- RECENT_OBSERVATIONS_JSON: {json.dumps(state.get('history', [])[-3:], ensure_ascii=False, indent=2)}
- DATA_SUMMARY_JSON: {json.dumps(state.get('collected_data_summary', {}), ensure_ascii=False, indent=2)}
- EXISTING_HYPOTHESES_JSON: {json.dumps(state.get('current_hypotheses', []), ensure_ascii=False, indent=2)}
### TASK
1. 0 個以上の仮説オブジェクトを生成せよ。既存仮説と重複しないように考慮すること。
2. 各仮説には **ユニークな** id (例: hyp_00X), text, priority (0–1), status ('new'で初期化) を含めよ。
3. 根拠データキー(supporting_evidence_keys)、次の検証案(next_validation_step_suggestion)も推奨。
- EDA_SUMMARY_TEXTがnullの場合は、まず対象データの代表サンプルを抽出して傾向把握する仮説を作成してください。
### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
[ {{ "id": "hyp_###", "text": "...", "priority": 0.##, "status": "new", ...}}, ... ]
            """
            logger.info("LLMに仮説生成を依頼中...")
            logger.debug(f"[AE] LLMプロンプト: {prompt_template}")
            response = await llm.ainvoke(prompt_template)
            llm_output_text = response.content
            logger.debug(f"[AE] LLM出力: {llm_output_text}")
            try:
                new_hypotheses_list = json.loads(llm_output_text)
                observation = {'generated_hypotheses': new_hypotheses_list}
                current_hypotheses = state.get('current_hypotheses', [])
                existing_ids = {h['id'] for h in current_hypotheses}
                added_count = 0
                updated_hypotheses = []
                for h_new in new_hypotheses_list:
                    if 'id' not in h_new or not isinstance(h_new.get('id'), str):
                       logger.warning(f"Skipping hypothesis due to missing or invalid ID: {h_new}")
                       continue
                    if h_new['id'] in existing_ids:
                        logger.warning(f"Hypothesis ID {h_new['id']} already exists. Skipping.")
                    else:
                        h_new.setdefault('status', 'new')
                        if 'text' in h_new and 'priority' in h_new:
                             updated_hypotheses.append(h_new)
                             existing_ids.add(h_new['id'])
                             added_count += 1
                        else:
                             logger.warning(f"Skipping hypothesis due to missing required fields (text, priority): {h_new}")
                state['current_hypotheses'] = current_hypotheses + updated_hypotheses
                logger.info(f"仮説を新規生成しました: {added_count}件")
                logger.info(f"生成された仮説: {updated_hypotheses}")
                observation['generated_hypotheses'] = updated_hypotheses
            except Exception as e:
                logger.error(f"[AE] 仮説生成LLM応答の解析エラー: {e}\nLLM Output: {llm_output_text}")
                raise ValueError(f"仮説生成LLM応答の解析エラー: {e}\nLLM Output: {llm_output_text}")
        elif action_type == "evaluate_hypothesis":
            hypothesis_id_to_evaluate = parameters.get("hypothesis_id")
            logger.info(f"[AE] evaluate_hypothesis: hypothesis_id={hypothesis_id_to_evaluate}")
            if not hypothesis_id_to_evaluate:
                raise ValueError("Missing 'hypothesis_id' parameter for evaluate_hypothesis")
            current_hypotheses = state.get('current_hypotheses', [])
            hypothesis_to_evaluate = next((h for h in current_hypotheses if h['id'] == hypothesis_id_to_evaluate), None)
            if not hypothesis_to_evaluate:
                 raise ValueError(f"Hypothesis with id '{hypothesis_id_to_evaluate}' not found.")
            prompt_template = f"""
### ROLE
あなたは客観的かつ批判的な監査評価者 AI です。
### TARGET_HYPOTHESIS
{json.dumps(hypothesis_to_evaluate, ensure_ascii=False, indent=2)}
### CONTEXT
- OBJECTIVE: {state['objective']}
- RELATED_HISTORY_JSON: {json.dumps(state.get('history', [])[-5:], ensure_ascii=False, indent=2)}
- DATA_SUMMARY_JSON: {json.dumps(state.get('collected_data_summary', {}), ensure_ascii=False, indent=2)}
- ADDITIONAL_PARAMETERS: {json.dumps(parameters, ensure_ascii=False, indent=2)}
### TASK
1. 利用可能なデータに基づき、TARGET_HYPOTHESIS を評価せよ。
2. evaluation_status を supported | rejected | needs_revision | inconclusive から選べ。
3. reasoning は評価の根拠を 2–4 文で述べよ。
4. inconclusive または needs_revision の場合は、評価に必要な次のデータ (required_next_data) を具体的に示せ (null も可)。
### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{ "hypothesis_id": "{hypothesis_id_to_evaluate}", "evaluation_status": "<status>", "reasoning": "...", "required_next_data": "<null_or_specific_data_needed>" }}
            """
            logger.info(f"LLMに仮説評価依頼 (Hypothesis ID: {hypothesis_id_to_evaluate})...")
            logger.debug(f"[AE] LLMプロンプト: {prompt_template}")
            response = await llm.ainvoke(prompt_template)
            llm_output_text = response.content
            logger.debug(f"[AE] LLM出力: {llm_output_text}")
            try:
                evaluation_result = json.loads(llm_output_text)
                observation = {'evaluation_result': evaluation_result}
                evaluation_status = evaluation_result.get('evaluation_status')
                reasoning = evaluation_result.get('reasoning')
                required_data = evaluation_result.get('required_next_data')
                evaluated_id = evaluation_result.get('hypothesis_id')
                logger.info(f"[AE] 評価結果: id={evaluated_id}, status={evaluation_status}, reasoning={reasoning}, required_next_data={required_data}")
                if not evaluated_id or evaluated_id != hypothesis_id_to_evaluate:
                     logger.warning(f"LLM evaluation response ID mismatch. Expected {hypothesis_id_to_evaluate}, got {evaluated_id}. Using expected ID.")
                     evaluated_id = hypothesis_id_to_evaluate

                found_and_updated = False
                for i, h in enumerate(current_hypotheses):
                    if h['id'] == evaluated_id:
                        current_hypotheses[i]['status'] = evaluation_status if evaluation_status else h['status']
                        current_hypotheses[i]['evaluation_reasoning'] = reasoning
                        current_hypotheses[i]['required_next_data'] = required_data
                        found_and_updated = True
                        break
                if not found_and_updated:
                     logger.warning(f"Evaluated hypothesis ID {evaluated_id} not found in current list during update.")
                else:
                    state['current_hypotheses'] = current_hypotheses
                    logger.info(f"仮説評価完了 (Hypothesis ID: {evaluated_id}), status={evaluation_status}")
                    if evaluation_status in ['supported', 'rejected']:
                        if currently_investigating_hypothesis_id == evaluated_id:
                             logger.info(f"Hypothesis {evaluated_id} resolved ({evaluation_status}). Clearing focus.")
                             state['currently_investigating_hypothesis_id'] = None
                        else:
                             logger.warning(f"Hypothesis {evaluated_id} resolved ({evaluation_status}), but it was not the focus ({currently_investigating_hypothesis_id}). Keeping focus unchanged.")
            except Exception as e:
                logger.error(f"[AE] 仮説評価LLM応答の解析エラー: {e}\nLLM Output: {llm_output_text}")
                raise ValueError(f"仮説評価LLM応答の解析エラー: {e}\nLLM Output: {llm_output_text}")
        elif action_type == "refine_plan":
            logger.warning("refine_plan アクションは現在ダミーです。")
            observation = {'plan_generated': 'dummy_plan_details', 'based_on_hypothesis': parameters.get('hypothesis_id')}
            state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) -1
        elif action_type == "execute_plan_step":
             logger.warning("execute_plan_step アクションは現在ダミーです。")
             observation = {'step_executed': parameters.get('step_id'), 'result': 'dummy_step_result'}
             state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) -1
        elif action_type == "conclude":
            logger.info("最終分析 (DataAnalyzer) を呼び出し中...")
            await asyncio.sleep(1)
            dummy_report = {
                "is_anomaly": False,
                "is_data_sufficient": True,
                "analysis": "調査が完了しました。(ダミーレポート)",
                "summary_of_findings": [h for h in state.get('current_hypotheses', []) if h['status'] in ['supported', 'rejected']],
                "recommendations": "定期的なモニタリング継続を推奨します。"
            }
            observation = {'final_report': dummy_report}
            state['final_result'] = dummy_report
            logger.info("最終分析完了。")
            state['next_action'] = None
        elif action_type == "error":
            error_msg = parameters.get('message', 'Undefined error occurred')
            logger.error(f"Error action executed: {error_msg}")
            observation = {'error': error_msg}
            state['error_message'] = error_msg
            state['next_action'] = None
        elif action_type == "refine_hypothesis":
            # サブ仮説分割／再定義
            hypothesis_id = parameters.get("hypothesis_id")
            h = next((h for h in state['current_hypotheses'] if h['id'] == hypothesis_id), None)
            if not h:
                logger.error(f"refine_hypothesis: 指定ID {hypothesis_id} の仮説が見つかりません")
                observation = {'refined_hypotheses': [], 'based_on': hypothesis_id, 'error': 'not found'}
            else:
                prompt = f"""
あなたは監査人です。以下の仮説を、\n- 意味的に重複しない\n- 検証しやすいサブ仮説\nへ分割してください。\n{{ "hypothesis": {json.dumps(h, ensure_ascii=False)} }}
\n【必ずJSON形式（Pythonのlist of dict）で出力してください。コードブロックや説明文は不要です。】
"""
                logger.info("refine_hypothesis: サブ仮説生成をLLMに依頼")
                response = await llm.ainvoke(prompt)
                try:
                    refined = json.loads(response.content)
                    if not isinstance(refined, list):
                        refined = []
                    # ここで各サブ仮説にstatus='new'を必ず付与
                    for h in refined:
                        if 'status' not in h or h['status'] is None:
                            h['status'] = 'new'
                except Exception as e:
                    logger.warning(f"refine_hypothesis: LLM応答パース失敗: {e}")
                    refined = []
                state['current_hypotheses'].extend(refined)
                observation = {'refined_hypotheses': refined, 'based_on': hypothesis_id}
            state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) - 1
        elif action_type == "triangulate_data":
            # データ三角照合
            hypothesis_id = parameters.get("hypothesis_id")
            queries = [
                {"agent_skill_id": "analyze_order",    "query": "購買単価データを取得..."},
                {"agent_skill_id": "query_payment",    "query": "請求金額データを取得..."},
                {"agent_skill_id": "query_inventory",  "query": "在庫出庫データを取得..."},
            ]
            results = {}
            for q in queries:
                try:
                    res = await smart_a2a_client.find_and_send_task(q['agent_skill_id'], q['query'])
                    results[q['agent_skill_id']] = res['result']
                except Exception as e:
                    logger.warning(f"triangulate_data: {q['agent_skill_id']} 取得失敗: {e}")
                    results[q['agent_skill_id']] = f"error: {e}"
            triangulated = {"joined_count": len(results)*10, "details": results}
            observation = {'triangulation_result': triangulated, 'based_on': hypothesis_id}
            state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) - 1
        elif action_type == "sample_and_test":
            # サンプリング検証
            hypothesis_id = parameters.get("hypothesis_id")
            sample_query = f"仮説 {hypothesis_id} 検証用に、購買データからランダムサンプル10件を取得してください。"
            try:
                res = await smart_a2a_client.find_and_send_task("analyze_order", sample_query)
                sample_results = res['result']
            except Exception as e:
                logger.warning(f"sample_and_test: サンプル取得失敗: {e}")
                sample_results = f"error: {e}"
            prompt = f"以下のサンプルデータをもとに、仮説 {hypothesis_id} の初期評価を行ってください。\n{json.dumps(sample_results, ensure_ascii=False)}"
            try:
                resp2 = await llm.ainvoke(prompt)
                sampling_test = resp2.content
            except Exception as e:
                logger.warning(f"sample_and_test: LLM評価失敗: {e}")
                sampling_test = f"error: {e}"
            observation = {'sampling_test_result': sampling_test, 'based_on': hypothesis_id}
            state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) - 1
        elif action_type == "escalate_to_expert":
            # 専門家エスカレーション
            hypothesis_id = parameters.get("hypothesis_id")
            message = f"仮説 {hypothesis_id} について専門家ヒアリングを実施してください。背景: {state['objective']}"
            try:
                task = await smart_a2a_client.find_and_send_task("ExpertAgent", message)
            except Exception as e:
                logger.warning(f"escalate_to_expert: タスク生成失敗: {e}")
                task = {"error": str(e)}
            observation = {'escalation_task': task, 'based_on': hypothesis_id}
            state['cycles_since_last_hypothesis'] = max(0, state.get('cycles_since_last_hypothesis',0) - 1)
        else:
            raise NotImplementedError(f"Unknown action type: {action_type}")
    except Exception as e:
        logger.error(f"Action Executor: Error executing action {action_type}: {e}", exc_info=True)
        observation = {'error': f"Failed to execute action {action_type}: {e}"}
        error_occurred = True
        state['error_message'] = str(e)
        state['next_action'] = None

    state['history'] = state.get('history', []) + [{
        'type': 'observation',
        'content': observation,
        'timestamp': datetime.datetime.now().isoformat()
    }]

    state['data_points_collected'] = data_points_collected_this_run

    if not error_occurred and action_type not in ["error", "conclude"]:
         state['next_action'] = None

    return state

def route_action(state: DynamicAgentState) -> Literal["action_executor", "__end__"]:
    logger.info("--- Routing Action ---")
    if state.get('error_message'):
        logger.error(f"Error detected in state: {state['error_message']}. Ending workflow.")
        return "__end__"

    next_action = state.get('next_action')
    if next_action:
        action_type = next_action.get('action_type')
        log_msg = f"Next action type decided: {action_type}"
        if state.get('currently_investigating_hypothesis_id'):
             log_msg += f" (Focus: {state['currently_investigating_hypothesis_id']})"
        logger.info(log_msg)
        if action_type == "error":
             logger.warning("Routing to __end__ based on 'error' action decided by Decision Maker.")
             return "__end__"
        else:
            logger.info("Routing to action_executor.")
            return "action_executor"
    else:
        logger.warning("No next_action determined by Decision Maker node or forced action. Ending workflow unexpectedly.")
        state['error_message'] = "Decision Maker failed to produce a valid next action."
        return "__end__"

async def eda_node(state: DynamicAgentState, llm: ChatOpenAI) -> DynamicAgentState:
    """
    EDAノード: collected_data_summaryから簡易統計を計算し、state['eda_summary']に格納。
    データがない場合は、LLMに依頼して基本的なデータ収集を指示する。
    """
    logger.info("--- Node: EDA (探索的データ分析) ---")
    summary_lines = []
    summary_dict = {}
    data_summary = state.get('collected_data_summary', {})
    objective = state.get('objective', 'No objective provided') # Objectiveを取得
    available_agents = state.get('available_data_agents_and_skills', []) # 利用可能なエージェント情報を取得

    # --- 共通の状態カウンタ更新 ---
    # ※ next_actionが決まる前に iteration をインクリメントする
    current_iteration = state.get('current_iteration', 0) + 1
    state['current_iteration'] = current_iteration
    state['cycles_since_last_hypothesis'] = 0 # EDA後はリセット

    if not data_summary:
        summary_lines.append("EDA要約: データが存在しません。")
        summary_dict = {}
        logger.info("EDA要約: データが存在しません。LLMに依頼してEDAに必要な基本データを収集します。")
        state['eda_summary'] = "EDA要約: データが存在しません。" # サマリー自体は更新
        state['eda_stats'] = {} # 統計情報も空で更新

        # --- ▼▼▼ 変更点: LLMによるデータ収集指示生成 ▼▼▼ ---
        prompt_template = f"""
### ROLE
あなたは、調査目的を達成するために最適な初期データ収集タスクを計画する AI アシスタントです。

### GOAL
提示された調査目的 (OBJECTIVE) のための初期データ分析 (EDA) に最も適したデータ収集タスク（単一）を決定してください。

### CONTEXT
- OBJECTIVE: {objective}
- AVAILABLE_DATA_AGENTS_JSON: {json.dumps(available_agents, ensure_ascii=False, indent=2)} # 利用可能なデータソースとスキル

### TASK
1. 上記 OBJECTIVE と AVAILABLE_DATA_AGENTS_JSON を考慮し、初期 EDA に最も関連性が高く、基本的な情報を得られると考えられる *単一の* データエージェントスキル (`agent_skill_id`) を選択してください。
2. そのスキルに対して、OBJECTIVE に沿った初期データ取得のための具体的な問い合わせ内容 (`query`) を生成してください。クエリは、広範囲すぎず、かつ具体的すぎない、初期分析に適したものであるべきです。
3. 結果を以下の JSON 形式で出力してください。

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{
  "agent_skill_id": "<selected_agent_skill_id>",
  "query": "<generated_query_for_initial_data>"
}}
"""
        logger.info("LLMに初期データ収集タスクの生成を依頼中...")
        try:
            response = await llm.ainvoke(prompt_template)
            llm_output_text = response.content
            logger.debug(f"[EDA] LLM Raw Output: {llm_output_text}")
            decision_data = json.loads(llm_output_text)

            agent_skill_id = decision_data.get("agent_skill_id")
            query = decision_data.get("query")

            if not agent_skill_id or not query:
                 raise ValueError("LLM response missing 'agent_skill_id' or 'query'")

            # 有効な skill_id かチェック (任意だが推奨)
            valid_skill_ids = {agent['skill_id'] for agent in available_agents}
            if agent_skill_id not in valid_skill_ids:
                logger.warning(f"LLM generated an unknown agent_skill_id: {agent_skill_id}. Falling back to error.")
                raise ValueError(f"LLM generated an unknown agent_skill_id: {agent_skill_id}")


            logger.info(f"LLM generated task: skill_id={agent_skill_id}, query={query}")
            state['next_action'] = {
                'action_type': 'query_data_agent',
                'parameters': {
                    'agent_skill_id': agent_skill_id,
                    'query': query
                }
            }

        except Exception as e:
            logger.error(f"EDA: LLMによる初期データ収集タスク生成に失敗: {e}", exc_info=True)
            # エラー発生時は error アクションを設定
            state['next_action'] = {
                'action_type': 'error',
                'parameters': {'message': f'Failed to generate initial data query via LLM: {e}'}
            }
            state['error_message'] = f'Failed to generate initial data query via LLM: {e}'
        # --- ▲▲▲ 変更点ここまで ▲▲▲ ---

    else:
        # --- 既存のEDA処理 (データがある場合) ---
        try:
            for key, value in data_summary.items():
                # valueがDataFrameの場合
                if isinstance(value, pd.DataFrame):
                    df = value
                # dictやlistの場合はDataFrameに変換を試みる
                elif isinstance(value, (dict, list)):
                    try:
                        df = pd.DataFrame(value)
                    except Exception as df_err:
                         logger.warning(f"EDA ({key}): DataFrame変換失敗: {df_err}")
                         summary_lines.append(f"{key}: DataFrame変換失敗")
                         continue
                else:
                    summary_lines.append(f"{key}: 型不明 ({type(value)})")
                    continue
                if df.empty:
                    summary_lines.append(f"{key}: データフレームは空です。")
                    continue
                desc = df.describe(include='all').T
                null_pct = df.isnull().mean() * 100
                summary_lines.append(f"{key}: (Rows: {len(df)})")
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    mean_val = desc.loc[col, 'mean'] if col in desc.index and 'mean' in desc.columns and pd.notna(desc.loc[col, 'mean']) else 'N/A'
                    std_val = desc.loc[col, 'std'] if col in desc.index and 'std' in desc.columns and pd.notna(desc.loc[col, 'std']) else 'N/A'
                    mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else mean_val
                    std_str = f"{std_val:.2f}" if isinstance(std_val, (int, float)) else std_val
                    missing = null_pct[col]
                    summary_lines.append(f"  - {col} (Type: {col_type}): Mean={mean_str}, Std={std_str}, Missing={missing:.1f}%")
                    summary_dict[f"{key}.{col}"] = {
                        'mean': mean_val if mean_val != 'N/A' else None,
                        'std': std_val if std_val != 'N/A' else None,
                        'missing_pct': missing,
                        'dtype': col_type
                    }
        except Exception as e:
            logger.error(f"EDA処理中にエラー発生: {e}", exc_info=True)
            summary_lines.append(f"EDAエラー: {e}")

        summary = "\n".join(summary_lines)
        logger.info(f"EDA要約:\n{summary}")
        state['eda_summary'] = summary
        state['eda_stats'] = summary_dict
        # データがある場合は仮説生成へ
        state['next_action'] = {'action_type': 'generate_hypothesis', 'parameters': {}}
        # --- 既存のEDA処理ここまで ---

    logger.info(f"Next action set by EDA: {state['next_action']['action_type']}")
    return state

def create_dynamic_agent_graph(llm, smart_a2a_client, data_analyzer, max_iterations=50) -> StateGraph:
    workflow = StateGraph(DynamicAgentState)

    # EDAノードを追加
    workflow.add_node("eda", partial(eda_node, llm=llm))
    workflow.add_node("decision_maker", partial(decision_maker_node, llm=llm))
    workflow.add_node("action_executor", partial(action_executor_node, llm=llm, smart_a2a_client=smart_a2a_client, data_analyzer=data_analyzer))

    # エントリポイントをedaに変更
    workflow.set_entry_point("eda")
    # eda→decision_maker
    workflow.add_edge("eda", "decision_maker")

    workflow.add_conditional_edges(
        "decision_maker",
        route_action,
        {
            "action_executor": "action_executor",
            "__end__": END
        }
    )

    def route_after_execution(state: DynamicAgentState) -> Literal["decision_maker", "__end__"]:
        logger.info("--- Routing After Execution ---")
        if state.get('error_message'):
            logger.error(f"Error detected after execution: {state['error_message']}. Ending workflow.")
            return "__end__"
        if state.get('final_result') is not None:
             logger.info("Conclusion reached. Ending workflow.")
             return "__end__"
        if state.get('current_iteration', 0) >= state.get('max_iterations', 10):
             logger.warning(f"Max iterations reached after execution. Ending workflow.")
             return "__end__"

        log_msg = "Continuing to decision_maker."
        if state.get('currently_investigating_hypothesis_id'):
            log_msg += f" (Focus: {state.get('currently_investigating_hypothesis_id')})"
        logger.info(log_msg)
        return "decision_maker"

    workflow.add_conditional_edges(
        "action_executor",
        route_after_execution,
        {
            "decision_maker": "decision_maker",
            "__end__": END
        }
    )

    graph = workflow.compile()
    logger.info("Dynamic agent graph compiled successfully.")
    return graph

async def main_test():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('test_react.log', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    test_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    class DummyClient:
        async def find_and_send_task(self, skill_id, message):
            logger.info(f"[DummyClient] find_and_send_task called: skill={skill_id}, message={message}")
            await asyncio.sleep(0.5)
            return {"result": {"id": f"task_{skill_id}_{datetime.datetime.now().isoformat()}"}}
        async def get_task(self, task_id):
             logger.info(f"[DummyClient] get_task called: task_id={task_id}")
             await asyncio.sleep(1)
             return {"status": "COMPLETED", "result": f"Dummy result for {task_id}. Found 5 relevant entries."}
    test_smart_a2a_client = DummyClient()
    class DummyAnalyzer:
         async def analyze_collected_data(self, objective, history, hypotheses, data_summary):
             logger.info("[DummyAnalyzer] analyze_collected_data called")
             await asyncio.sleep(0.5)
             return {
                 "is_anomaly": False,
                 "is_data_sufficient": True,
                 "analysis": "Final analysis report (dummy).",
                 "summary_of_findings": [h for h in hypotheses if h['status'] in ['supported', 'rejected']],
                 "recommendations": "Continue monitoring."
             }
    test_data_analyzer = DummyAnalyzer()

    available_actions = [
        {'type': 'query_data_agent', 'description': 'Query a data agent for specific information related to hypotheses.'},
        {'type': 'generate_hypothesis', 'description': 'Generate new hypotheses based on current data and objective.'},
        {'type': 'evaluate_hypothesis', 'description': 'Evaluate a specific hypothesis using available data.'},
        {'type': 'conclude', 'description': 'Conclude the investigation and generate final report.'},
        {'type': 'error', 'description': 'Handle an unexpected error state.'},
        {'type': 'refine_hypothesis', 'description': '仮説を分割・再定義する。'},
        {'type': 'triangulate_data', 'description': '複数データソースでクロス照合する。'},
        {'type': 'escalate_to_expert', 'description': '専門家にエスカレーションする。'},
        {'type': 'sample_and_test', 'description': 'サンプリング検証を行う。'},
    ]
    available_data_agents_and_skills = [
        {'agent_name': 'PurchaseAgent', 'skill_id': 'analyze_order', 'description': '発注データを分析します。'},
        {'agent_name': 'ExpenseAgent', 'skill_id': 'query_expense', 'description': '経費データを検索します。'},
        {'agent_name': 'VendorAgent', 'skill_id': 'query_vendor', 'description': '取引先情報を検索します。'},
        {'agent_name': 'EmployeeAgent', 'skill_id': 'query_employee', 'description': '従業員情報を検索します。'},
        {'agent_name': 'PaymentAgent', 'skill_id': 'query_payment', 'description': '支払データを検索します。'}
    ]

    graph = create_dynamic_agent_graph(
        llm=test_llm,
        smart_a2a_client=test_smart_a2a_client,
        data_analyzer=test_data_analyzer,
        max_iterations=50
    )

    initial_state = DynamicAgentState(
        objective="特定の従業員と取引先の間で、単価が過去平均より著しく高い購買がないか検知する",
        history=[],
        current_hypotheses=[],
        collected_data_summary={},
        active_plan=None,
        next_action=None,
        final_result=None,
        available_actions=available_actions,
        available_data_agents_and_skills=available_data_agents_and_skills,
        error_message=None,
        max_iterations=50,
        current_iteration=0,
        data_points_collected=0,
        cycles_since_last_hypothesis=0,
        max_queries_without_hypothesis=2,
        consecutive_query_count=0,
        currently_investigating_hypothesis_id=None,
        eval_repeat_count=0
    )

    logger.info("Starting graph execution...")
    final_state = None
    config = {"recursion_limit": 100}
    async for output in graph.astream(initial_state, config=config):
        node_name = list(output.keys())[0]
        node_output = output[node_name]
        logger.info(f"--- Output from node: {node_name} ---")
        if END in output:
             final_state = output[END]
             logger.info("--- Graph ended ---")
             break
        else:
             final_state = node_output

    logger.info("--- Graph execution finished ---")

    if final_state:
        print("\n--- Final State ---")
        print(f"Objective: {final_state.get('objective')}")
        print(f"Iterations: {final_state.get('current_iteration')}")
        print(f"Error Message: {final_state.get('error_message')}")
        print(f"Final Result: {json.dumps(final_state.get('final_result'), ensure_ascii=False, indent=2)}")
        print(f"\nFinal Hypotheses: {json.dumps(final_state.get('current_hypotheses'), ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main_test())