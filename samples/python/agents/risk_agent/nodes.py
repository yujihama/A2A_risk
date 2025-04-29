from A2A_risk.samples.python.agents.risk_agent.prompts import (
    get_decision_maker_prompt,
    get_generate_hypothesis_prompt,
    get_evaluate_hypothesis_prompt,
    get_refine_hypothesis_prompt,
    get_initial_data_query_prompt,
    get_query_refinement_prompt,
    get_query_result_summary_prompt,
    get_eda_prompt,
    get_data_analysis_prompt,
    get_plan_verification_prompt,
    get_summarize_verification_prompt
)
from A2A_risk.samples.python.agents.risk_agent.utils import (
    _all_hypotheses_resolved,
    _unresolved_ratio,
    _count_inconclusive,
    _summarize_hypotheses,
)
from A2A_risk.samples.python.agents.risk_agent.config import *
from A2A_risk.samples.python.agents.risk_agent.state import *
from A2A_risk.samples.python.agents.risk_agent.agent import execute_step, PlanStep  # 追加: agent.pyの関数とモデルをimport

import logging
import json
import pandas as pd
import datetime
import asyncio

logger = logging.getLogger(__name__)

# --- ログ用ユーティリティ ---
def _log_hypotheses_status(hyps, prefix="[HYP_STATUS]"):
    statuses = [f"{h.get('id','?')}:{h.get('status','?')}" for h in hyps]
    logger.info(f"{prefix} 全hypステータス: {', '.join(statuses)}")

# eda_node, decision_maker_node, action_executor_node のスケルトン

async def eda_node(state: DynamicAgentState, llm):
    """
    EDAノード: collected_data_summaryから簡易統計を計算し、state['eda_summary']に格納。
    データがない場合は、LLMに依頼して基本的なデータ収集を指示する。
    """
    logger.info("[EDA] --- Node: EDA (探索的データ分析) ---")
    logger.info(f"[EDA] フォーカスhyp: {state.get('currently_investigating_hypothesis_id')}")
    summary_lines = []
    summary_dict = {}
    data_summary = state.get('collected_data_summary', {})
    objective = state.get('objective', 'No objective provided')
    available_agents = state.get('available_data_agents_and_skills', [])

    # --- 共通の状態カウンタ更新 ---
    current_iteration = state.get('current_iteration', 0) + 1
    state['current_iteration'] = current_iteration
    state['cycles_since_last_hypothesis'] = 0

    if not data_summary:
        summary_lines.append("EDA要約: データが存在しません。")
        summary_dict = {}
        state['eda_summary'] = "EDA要約: データが存在しません。"
        state['eda_stats'] = {}

        # prompts.py の get_initial_data_query_prompt を利用
        prompt_template = get_initial_data_query_prompt(objective, available_agents)
        logger.info("LLMに初期データ収集タスクの生成を依頼中...")
        # logger.info(f"LLMに次のプロンプトを送信中...: {prompt_template}")
        try:
            decision_data = await llm.ainvoke(prompt_template)
            logger.info(f"[EDA] LLM出力: {decision_data}")
            
            agent_skill_id = decision_data.get("agent_skill_id","None") # 要修正；実際のagent_skill_idが入るように
            query = decision_data.get("query","None") # 要修正；実際のqueryが入るように

            if not agent_skill_id or not query:
                raise ValueError("LLM response missing 'agent_skill_id' or 'query'")

            # available_agents には AgentCard オブジェクトまたは dict が格納される場合がある。
            # skill_id 抽出のユーティリティ
            valid_skill_ids: set[str] = set()
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
                    # 想定外の構造は無視してスキップ
                    continue

            logger.info(f"[EDA] アクション決定: query_data_agent, skill_id={agent_skill_id}, query={query}")
            state['next_action'] = {
                'action_type': 'query_data_agent',
                'parameters': {
                    'agent_skill_id': agent_skill_id,
                    'query': query
                }
            }
        except Exception as e:
            logger.error(f"[EDA] LLM初期データ収集失敗: {e}")
            state['next_action'] = {
                'action_type': 'error',
                'parameters': {'message': f'Failed to generate initial data query via LLM: {e}'}
            }
            state['error_message'] = f'Failed to generate initial data query via LLM: {e}'
    else:
        try:
            for key, value in data_summary.items():
                if isinstance(value, pd.DataFrame):
                    df = value
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
        logger.info(f"[EDA] データ要約生成完了。次アクション: generate_hypothesis")
        state['next_action'] = {'action_type': 'generate_hypothesis', 'parameters': {}}

    logger.info(f"[EDA] 次アクション: {state['next_action']['action_type']}")
    return state

async def query_data_agent_node(state: DynamicAgentState, smart_a2a_client, llm):
    """
    データエージェントにクエリを実行し、結果をstateに格納するノード。
    """
    logger.info("[QDA] --- Node: Query Data Agent ---")
    action_details = state.get('next_action')
    original_next_action = action_details  # 保存して後で比較
    if not action_details or action_details.get('action_type') != 'query_data_agent':
        logger.error("[QDA] Invalid action details for query_data_agent_node.")
        state['error_message'] = "Invalid action details for query_data_agent_node."
        state['next_action'] = None # エラーなのでアクションをクリア
        # エラー時の Observation を記録する方が親切かもしれない
        state['history'] = state.get('history', []) + [{
             'type': 'observation',
             'content': {'error': 'Invalid action details for query_data_agent_node.'},
             'timestamp': datetime.datetime.now().isoformat()
        }]
        return state

    parameters = action_details.get('parameters', {})
    agent_skill_id = parameters.get("agent_skill_id")
    query = parameters.get("query")
    observation = None
    error_occurred = False
    data_points_collected_this_run = 0

    # Action Executor で action は記録済みなので、ここでは実行ログのみ
    logger.info(f"[QDA] Executing: skill_id={agent_skill_id}, query={query}")

    if not agent_skill_id or not query:
        logger.error("[QDA] Missing 'agent_skill_id' or 'query' parameter")
        observation = {'error': "Missing 'agent_skill_id' or 'query' parameter", 'agent_skill_id': agent_skill_id, 'query': query}
        error_occurred = True
        state['error_message'] = "Missing 'agent_skill_id' or 'query' parameter"
    else:
        try:
            # データエージェントのスキルIDとクエリを使用して、LLMで細分化されたクエリを生成
            prompt_template = get_query_refinement_prompt(query)
            refined_query_data = await llm.ainvoke(prompt_template)
            logger.info(f"[QDA] query細分化LLM出力: {refined_query_data}")

            required_data_list = []
            for pattern in refined_query_data["answer"]:
                step_id = pattern.get("step_id", "")
                required_data = pattern.get("required_data", {}).get("new", {})
                if required_data != {}:
                    # 既存のrequired_dataと同じものがあればstep_idを追加、なければ新規追加
                    found = False
                    for item in required_data_list:
                        if item["required_data"] == required_data:
                            # 既存のstep_idをリスト化して追加
                            if isinstance(item["step_id"], list):
                                item["step_id"].append(step_id)
                            else:
                                item["step_id"] = [item["step_id"], step_id]
                            found = True
                            break
                    if not found:
                        required_data_list.append({"step_id": step_id, "required_data": required_data})

            result_text = []
            raw_data = []

            for required_data in required_data_list:
                # 細分化されたqueryを使ってデータエージェントにクエリを実行
                step = PlanStep(
                    id=f"step_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    description=f"データエージェントへのクエリ: {str(required_data['required_data'])}",
                    skill_id=agent_skill_id,
                    input_data={"input": str(required_data['required_data']), "return_type": "df"}, 
                    parameters=parameters
                )
                step_result = await execute_step(step)

                if step_result.is_completed:
                    text = None
                    data = None # dict[str, list[dict]] を期待
                    if hasattr(step_result, 'output_parts') and step_result.output_parts:
                        for part in step_result.output_parts:
                            if hasattr(part, 'type') and part.type == 'text':
                                text = getattr(part, 'text', None)
                            elif hasattr(part, 'type') and part.type == 'data':
                                data = getattr(part, 'data', None)
                    elif isinstance(step_result.output_data, dict):
                        # dict 形式の場合、text と data を想定
                        text = step_result.output_data.get("text", "")
                        data = step_result.output_data.get("data", None)
                    else:
                        # 文字列のみの場合
                        text = str(step_result.output_data)
                        data = None
                    
                    result_text.append(text)
                    raw_data.append({"step_id": required_data['step_id'], "data": data})
                
            # LLMでresult_textを要約し、発生したエラーなどを記録
            prompt_template = get_query_result_summary_prompt(result_text,required_data_list)
            logger.debug(f"[QDA] query結果要約LLMプロンプト: {prompt_template}")
            result_summary = await llm.ainvoke(prompt_template)
            logger.debug(f"[QDA] query結果要約LLM出力: {result_summary}")
            
            # main text fallback
            main_text = result_text[0] if result_text else None

            # 取得したデータを collected_data_summary に格納
            current_summary = state.get('collected_data_summary', {})
            current_summary[f"{agent_skill_id}_text"] = result_summary.get("summary", "")

            state['collected_data_summary'] = current_summary

            # Observation を生成
            observation = {
                'action_type': 'query_data_agent',
                'result': main_text,  # 主なテキスト結果
                'agent_skill_id': agent_skill_id,
                'query': query,
                'refined_query': refined_query_data,
                'agent': None,
                # 'data_summary_keys_added': added_data_keys, # 追加されたデータのキー
                'raw_data': raw_data 
            }

            # データポイント数と連続クエリ回数を更新
            data_points_collected_this_run = len(str(step_result.output_data).split()) // 10 # 簡易計算
            state['data_points_collected'] = state.get('data_points_collected', 0) + data_points_collected_this_run
            state['consecutive_query_count'] = state.get('consecutive_query_count', 0) + 1
            logger.info(f"[QDA] Data points collected approx: {data_points_collected_this_run}")


        except Exception as e:
            logger.error(f"[QDA] Error during query_data_agent execution: {e}", exc_info=True)
            observation = {'error': f"Failed to execute query_data_agent: {e}", 'agent_skill_id': agent_skill_id, 'query': query}
            error_occurred = True
            state['error_message'] = str(e)

    # Observation を履歴に追加
    state['history'] = state.get('history', []) + [{
        'type': 'observation',
        'content': observation,
        'timestamp': datetime.datetime.now().isoformat()
    }]

    # ノード名を履歴に追加
    state['history'] = state.get('history', []) + [{
        'type': 'node',
        'content': {'node_name': 'query_data_agent_node'},
        'timestamp': datetime.datetime.now().isoformat()
    }]

    # 次アクションが同じまま（= 新たにセットされなかった）か None の場合はクリア
    if state.get('next_action') is None or state['next_action'] == original_next_action:
        state['next_action'] = None

    return state

async def decision_maker_node(state: DynamicAgentState, llm):
    logger.info("[DM] --- Node: Decision Maker ---")
    logger.info(f"[DM] フォーカスhyp: {state.get('currently_investigating_hypothesis_id')}")
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

    # --- NEW: verification pending handling ---
    for h in current_hypotheses:
        if h.get('status') == 'supported':
            ver = h.get('verification', {})
            if ver.get('status') == 'pending':
                if currently_investigating_hypothesis_id != h['id']:
                    logger.info(f"[DM] Verification pending for supported hypothesis {h['id']}. Setting focus and planning verification.")
                    state['currently_investigating_hypothesis_id'] = h['id']
                if not state.get('next_action'):
                    state['next_action'] = {'action_type': 'plan_verification', 'parameters': {'hypothesis_id': h['id']}}
                return state

    # 直前のノードがquery_data_agent_nodeかつ仮説が一つもない場合はEDAへ遷移
    def was_last_node_query_data_agent(state):
        for entry in reversed(state.get('history', [])):
            if entry.get('type') == 'node':
                return entry.get('content', {}).get('node_name') == 'query_data_agent_node'
        return False

    if was_last_node_query_data_agent(state):
        if not current_hypotheses:
            logger.info("[DM] 直前がquery_data_agent_nodeかつ仮説ゼロなのでaction_executor_node EDAへ遷移 (action_type: 'eda')")
            state['next_action'] = {'action_type': 'eda', 'parameters': {}}
            return state
        else:
            logger.info("[DM] 直前がquery_data_agent_nodeかつ仮説があるので action_executor_node データ分析へ遷移 (action_type: 'data_analysis')")
            state['next_action'] = {'action_type': 'data_analysis', 'parameters': {}}
            return state

    predefined_action = state.get('next_action')
    if predefined_action:
        logger.info(f"[DM] Previous node already set the next action to: {predefined_action.get('action_type')}. DM skips its logic.")
        return state

    status_weight = {'needs_revision': 0, 'inconclusive': 1, 'new': 2}

    def risk_score(h):
        coverage = 0
        if h['id'] in eda_stats and isinstance(eda_stats[h['id']], dict):
            coverage = eda_stats[h['id']].get('coverage_pct', 0)
        elif h['id'] in collected_data_summary and isinstance(collected_data_summary[h['id']], dict):
            coverage = collected_data_summary[h['id']].get('coverage_pct', 0)
        return h.get('priority', 0) * (1 - coverage)

    def select_focus_candidate(hypos):
        return sorted(
            hypos,
            key=lambda h: (status_weight.get(h['status'], 99), -risk_score(h)),
        )

    unresolved_ratio = _unresolved_ratio(current_hypotheses)
    stagnation_score = (current_iteration / max_iterations) * unresolved_ratio + (cycles_since_last_hypothesis / SIM_THRESHOLD)
    if _all_hypotheses_resolved(current_hypotheses):
        logger.info("[DM][FORCE] 全仮説が解決済み。自動conclude遷移")
        _log_hypotheses_status(current_hypotheses)
        action = {'action_type': 'conclude', 'parameters': {}}
        logger.info(f"Next action: {action['action_type']}")
        state['next_action'] = action
        state['current_iteration'] = current_iteration + 1
        return state
    if stagnation_score >= STAGNATION_THRESHOLD:
        logger.info(f"[DM][FORCE] stagnation_score={stagnation_score:.2f} 閾値超過。hyp {currently_investigating_hypothesis_id} needs_revisionへ")
        _log_hypotheses_status(current_hypotheses)
        if currently_investigating_hypothesis_id:
            for h in current_hypotheses:
                if h['id'] == currently_investigating_hypothesis_id and h['status'] not in ['supported', 'rejected']:
                    h['status'] = 'needs_revision'
                    h['evaluation_reasoning'] = f"stagnation_scoreが閾値({STAGNATION_THRESHOLD})を超過したためneeds_revisionに設定。"
                    logger.info(f"仮説 {currently_investigating_hypothesis_id} をstagnationによりneeds_revisionに設定。")
                    break
            # フォーカスは維持したまま、強制的に evaluate_hypothesis を実行させる
            state['next_action'] = {
                'action_type': 'evaluate_hypothesis',
                'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}
            }
            thought = f"stagnation_score 超過により仮説 {currently_investigating_hypothesis_id} を needs_revision とし、再評価を強制。"
            state['history'] = state.get('history', []) + [{
                'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()
            }]
            state['current_iteration'] = current_iteration + 1
            return state
        # フォーカスしている仮説が無い場合は従来のロジックへフォールバック
        # 修正: needs_revisionも除外する
        unresolved = [h for h in current_hypotheses if h['status'] in ['inconclusive', 'new']]
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
        logger.info(f"[DM][FORCE] 最大イテレーション到達。強制終了")
        action = {'action_type': 'error', 'parameters': {'message': 'Max iterations reached'}}
        logger.info(f"Next action: {action['action_type']}")
        state['next_action'] = action
        state['error_message'] = 'Max iterations reached'
        return state

    focused_hypothesis = None

    focus_candidates = [h for h in current_hypotheses if h['status'] in ['inconclusive', 'new']]
    focus_candidates = select_focus_candidate(focus_candidates)
    if currently_investigating_hypothesis_id:
        focused_hypothesis = next((h for h in current_hypotheses if h['id'] == currently_investigating_hypothesis_id), None)
        if not focused_hypothesis or focused_hypothesis['status'] in ['supported', 'rejected']:
            if focused_hypothesis:
                 logger.info(f"[DM] Focus hypothesis {currently_investigating_hypothesis_id} is already {focused_hypothesis['status']}. Clearing focus.")
            else:
                 logger.warning(f"[DM] Focus hypothesis {currently_investigating_hypothesis_id} not found during check. Clearing focus.")
            currently_investigating_hypothesis_id = None
            state['currently_investigating_hypothesis_id'] = None
            focused_hypothesis = None
    if not focused_hypothesis and focus_candidates:
        logger.info(f"[DM] フォーカスhyp切替: {currently_investigating_hypothesis_id}")
        hypothesis_to_focus = focus_candidates[0]
        currently_investigating_hypothesis_id = hypothesis_to_focus['id']
        state['currently_investigating_hypothesis_id'] = currently_investigating_hypothesis_id
        focused_hypothesis = hypothesis_to_focus
        logger.info(f"No focus. Setting focus to: {currently_investigating_hypothesis_id} (Priority: {hypothesis_to_focus.get('priority')})")
        action = {
            "action_type": "evaluate_hypothesis",
            "parameters": {"hypothesis_id": currently_investigating_hypothesis_id}
        }
        logger.info(f"[DM] Next action: {action['action_type']}")
        thought = f"Setting focus on new or nearly-resolved hypothesis {currently_investigating_hypothesis_id} and initiating evaluation."
        state['next_action'] = action
        state['history'] = state.get('history', []) + [{
            'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()
        }]
        state['current_iteration'] = current_iteration + 1
        return state

    inconclusive_count = _count_inconclusive(current_hypotheses)
    force_refine_or_generate = False
    if cycles_since_last_hypothesis >= MAX_INCONCLUSIVE_BEFORE_GENERATE or inconclusive_count >= MAX_INCONCLUSIVE_BEFORE_GENERATE:
        if focused_hypothesis and focused_hypothesis.get('required_next_data'):
             logger.info(f"[DM] Inconclusive count ({inconclusive_count}) or cycles ({cycles_since_last_hypothesis}) reached threshold, but required_next_data exists for focus {focused_hypothesis['id']}. Proceeding to LLM decision.")
             pass
        else:
            force_refine_or_generate = True
            logger.warning(f"[DM] cycles_since_last_hypothesis={cycles_since_last_hypothesis}, inconclusive_count={inconclusive_count} で refine_hypothesis/generate_hypothesis を強制します。(No focus or no required_next_data)")

    if force_refine_or_generate:
        logger.info(f"[DM][FORCE] 強制アクション: {action['action_type']} (理由: inconclusive/needs_revision閾値超過)")
        action = None
        thought = None
        if focused_hypothesis:
            if focused_hypothesis['status'] in ['inconclusive', 'needs_revision'] and not focused_hypothesis.get('required_next_data'):
                 action = {'action_type': 'refine_hypothesis', 'parameters': {'hypothesis_id': focused_hypothesis['id']}}
                 logger.info(f"[DM] Next action: {action['action_type']} (Forced due to inconclusive/no required data)")
                 thought = f"Forcing 'refine_hypothesis' for {focused_hypothesis['id']} due to persistent inconclusive status without clear required data."
            else:
                 logger.info(f"Force refine condition met, but focused hypothesis {focused_hypothesis['id']} has required_next_data or is not inconclusive/needs_revision. Proceeding to LLM decision.")
                 pass
        else:
            action = {'action_type': 'generate_hypothesis', 'parameters': {}}
            logger.info(f"Next action: {action['action_type']} (Forced due to inconclusive count/cycles with no focus)")
            thought = f"Forcing 'generate_hypothesis' due to inconclusive/needs_revision cycles or count (no focus)."
        if action:
             state['next_action'] = action
             if thought:
                 state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
             state['current_iteration'] = current_iteration + 1
             return state

    if not focused_hypothesis:
        logger.info(f"[DM][FORCE] フォーカスなし。generate_hypothesis強制")
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
        logger.info(f"[DM] Next action: {action['action_type']}")
        thought = "No focus and no 'new' hypotheses. Attempting to generate new ones."
        state['next_action'] = action
        state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
        state['current_iteration'] = current_iteration + 1
        return state

    if focused_hypothesis:
        logger.info(f"[DM] フォーカス継続: {currently_investigating_hypothesis_id} (status: {focused_hypothesis['status']})")
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
            logger.info(f"[DM][FORCE] evaluate_hypothesisループガード発動。generate_hypothesis強制")
            action = {'action_type': 'generate_hypothesis', 'parameters': {}}
            logger.info(f"Next action: {action['action_type']}")
            state['currently_investigating_hypothesis_id'] = None
            state['eval_repeat_count'] = 0
            thought = f"Loop guard: Too many repeated evaluations for {currently_investigating_hypothesis_id}. Forcing generate_hypothesis."
            state['next_action'] = action
            state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
            state['current_iteration'] = current_iteration + 1
            return state
        if focused_hypothesis['status'] in ['inconclusive', 'needs_revision']:
            last_obs = None
            for entry in reversed(state.get('history', [])):
                if entry['type'] == 'observation':
                    last_obs = entry['content']
                    break
            action = None
            thought = None
            # if last_obs and isinstance(last_obs, dict):
            #     agent_skill = last_obs.get('agent_skill_id')
            #     obs_str = str(last_obs)
            if action:
                 logger.info(f"[DM] Next action (determined by observation analysis): {action['action_type']}")
                 state['next_action'] = action
                 if thought:
                      state['history'] = state.get('history', []) + [{'type': 'thought', 'content': thought, 'timestamp': datetime.datetime.now().isoformat()}]
                 state['current_iteration'] = current_iteration + 1
                 return state
            logger.info("[DM] No specific action determined by observation analysis for inconclusive/needs_revision status. Proceeding to LLM decision.")

        # --- LLMによるアクション決定 ---
        logger.info(f"[DM] LLMアクション決定依頼 (focus: {currently_investigating_hypothesis_id})")
        
        prompt_template = get_decision_maker_prompt(state, focused_hypothesis, currently_investigating_hypothesis_id, state.get('available_actions', []))
        logger.debug(f"[DM] LLMプロンプト: {prompt_template}")
        decision_data = await llm.ainvoke(prompt_template)
        
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
        logger.info(f"[DM] アクション決定: {action_type}, params={params}")
        logger.info(f"[DM] thought: {thought}")
    
        state['current_iteration'] = current_iteration + 1
        return state

    logger.error("Decision Maker reached an unexpected state (no focus and no action determined by logic).")
    action = {'action_type': 'error', 'parameters': {'message': 'Unexpected state in Decision Maker'}}
    logger.info(f"Next action: {action['action_type']}")
    state['next_action'] = action
    state['error_message'] = 'Unexpected state in Decision Maker'
    state['current_iteration'] = current_iteration + 1
    return state

async def action_executor_node(state: DynamicAgentState, llm, smart_a2a_client, data_analyzer):
    """
    決定されたアクションを実行し、関連するStateフィールドを更新するノード。
    仮説評価後にフォーカスを解除するロジックを追加。
    """
    logger.info("[AE] --- Node: Action Executor ---")
    logger.info(f"[AE] フォーカスhyp: {state.get('currently_investigating_hypothesis_id')}")
    action_details = state.get('next_action')
    original_next_action = action_details  # 保存して後で比較
    if not action_details:
        logger.error("[AE] Action Executor: No action defined in state.")
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
    try:
        if action_type == "generate_hypothesis":
            eda_summary = state.get('eda_summary', None)
            eda_stats = state.get('eda_stats', {})
            if eda_summary:
                eda_summary_lines = eda_summary.splitlines()
                eda_summary_head = "\n".join(eda_summary_lines[:5])
            else:
                eda_summary_head = None
            prompt_template = get_generate_hypothesis_prompt(state)
            logger.info("LLMに仮説生成を依頼中...")
            # logger.debug(f"[AE] LLMプロンプト: {prompt_template}") # Debugレベルに変更推奨
            new_hypotheses_list = await llm.ainvoke(prompt_template)
            current_hypotheses = state.get('current_hypotheses', [])
            existing_ids = {h['id'] for h in current_hypotheses}
            added_count = 0
            updated_hypotheses = []
            # 既存IDでユニーク化し、必須項目が揃っているものだけを追加
            hypotheses_dict = {h['id']: h for h in current_hypotheses if 'id' in h}
            for h_new in new_hypotheses_list['hypothesis']:
                if 'id' not in h_new or not isinstance(h_new.get('id'), str):
                    logger.warning(f"Skipping hypothesis due to missing or invalid ID: {h_new}")
                    continue
                if 'text' not in h_new or 'priority' not in h_new:
                    logger.warning(f"Skipping hypothesis due to missing required fields (text, priority): {h_new}")
                    continue
                h_new.setdefault('status', 'new')
                hypotheses_dict[h_new['id']] = h_new  # 上書きまたは追加
            state['current_hypotheses'] = list(hypotheses_dict.values())
            logger.info(f"仮説を新規生成しました: {len(state['current_hypotheses']) - len(current_hypotheses)}件")
            logger.info(f"生成された仮説: {[h for h in state['current_hypotheses'] if h not in current_hypotheses]}")
            observation = {'generated_hypotheses': [h for h in state['current_hypotheses'] if h not in current_hypotheses]} # observation の内容を整理

        elif action_type == "evaluate_hypothesis":
            hypothesis_id_to_evaluate = parameters.get("hypothesis_id")
            # logger.info(f"[AE] evaluate_hypothesis: hypothesis_id={hypothesis_id_to_evaluate}")
            if not hypothesis_id_to_evaluate:
                 # raise ValueError("Missing 'hypothesis_id' parameter for evaluate_hypothesis")
                 logger.error("[AE] Missing 'hypothesis_id' parameter for evaluate_hypothesis")
                 observation = {'error': "Missing 'hypothesis_id'", 'action_type': action_type}
                 error_occurred = True
            else:
                 current_hypotheses = state.get('current_hypotheses', [])
                 hypothesis_to_evaluate = next((h for h in current_hypotheses if h['id'] == hypothesis_id_to_evaluate), None)
                 if not hypothesis_to_evaluate:
                      # raise ValueError(f"Hypothesis with id '{hypothesis_id_to_evaluate}' not found.")
                      logger.error(f"[AE] Hypothesis with id '{hypothesis_id_to_evaluate}' not found.")
                      observation = {'error': f"Hypothesis '{hypothesis_id_to_evaluate}' not found", 'action_type': action_type}
                      error_occurred = True
                 else:
                    prompt_template = get_evaluate_hypothesis_prompt(state, hypothesis_to_evaluate, parameters)
                    logger.info(f"LLMに仮説評価依頼 (Hypothesis ID: {hypothesis_id_to_evaluate})...")
                    logger.debug(f"[AE] LLMプロンプト: {prompt_template}") 
                    evaluation_result = await llm.ainvoke(prompt_template)
                    
                    observation = {'evaluation_result': evaluation_result} # observation の内容を整理
                    evaluation_status = evaluation_result.get('evaluation_status')
                    reasoning = evaluation_result.get('reasoning')
                    required_data = evaluation_result.get('required_next_data')
                    evaluated_id = evaluation_result.get('hypothesis_id')
                    logger.info(f"[AE] 評価結果: id={evaluated_id}, status={evaluation_status}, reasoning={reasoning}, required_next_data={required_data}")
                    # if not evaluated_id or evaluated_id != hypothesis_id_to_evaluate:
                    #      logger.warning(f"LLM evaluation response ID mismatch. Expected {hypothesis_id_to_evaluate}, got {evaluated_id}. Using expected ID.")
                    #      evaluated_id = hypothesis_id_to_evaluate
                    if not evaluated_id:
                        logger.warning(f"LLM evaluation response missing 'hypothesis_id'. Using requested ID: {hypothesis_id_to_evaluate}.")
                        evaluated_id = hypothesis_id_to_evaluate
                    elif evaluated_id != hypothesis_id_to_evaluate:
                        logger.warning(f"LLM evaluation response ID mismatch. Expected {hypothesis_id_to_evaluate}, got {evaluated_id}. Using evaluated ID: {evaluated_id}")
                    
                    # --- FIX: avoid duplicate hypotheses & ensure unique update ---
                    # Build a dict keyed by hypothesis ID so that each ID appears **once**
                    hypotheses_dict = {h['id']: h.copy() for h in current_hypotheses if 'id' in h}

                    # Use the evaluated_id if provided, otherwise fall back to the originally requested ID
                    target_id = evaluated_id or hypothesis_id_to_evaluate
                    target = hypotheses_dict.get(target_id)
                    if target:
                        target.update({
                            'status': evaluation_status or target.get('status'),
                            'evaluation_reasoning': reasoning,
                            'required_next_data': required_data,
                        })
                    else:
                        logger.error(f"Evaluated hypothesis ID {target_id} not found in current list during update.")

                    # Overwrite state with the **deduplicated** list
                    state['current_hypotheses'] = list(hypotheses_dict.values())
                    logger.info(f"Hypothesis evaluation complete (ID: {target_id}), status={evaluation_status}")
                    # フォーカス解除ロジック
                    if evaluation_status == 'rejected':
                        # --- TEMP CLOSE LOGIC: 単一 rejected で即 conclude --------------------------
                        # 疎通確認用の一時的仕様。複合条件（他仮説状況など）は将来的に Decision Maker にて統合管理予定。
                        # logger.info("[AE][TEMP_STOP] 仮説が rejected になったため、conclude へ遷移します。")
                        # state['next_action'] = {'action_type': 'conclude', 'parameters': {}}
                        state['currently_investigating_hypothesis_id'] = None
                    elif evaluation_status == 'supported':
                        # --- ここも一時ロジック: supported は verification へ -------------------------
                        # DM での詳細制御が整備され次第、フォーカス解除や conclue 遷移の扱いを見直す想定。
                        ver_dict = target.get('verification', {}) if isinstance(target, dict) else {}
                        # 未設定の場合のみ pending をセット
                        if ver_dict.get('status') is None:
                            ver_dict['status'] = 'pending'
                        target['verification'] = ver_dict
                        # Decision Maker に判断させるため next_action は設定しない
                        state['currently_investigating_hypothesis_id'] = target_id
                    else:
                        if evaluation_status in ['inconclusive', 'needs_revision'] and (not required_data or required_data == {}):
                            logger.info("[AE] データ不足判定 (required_next_data なし) のため refine_hypothesis へ遷移します。")
                            state['next_action'] = {
                                'action_type': 'refine_hypothesis',
                                'parameters': {'hypothesis_id': target_id}
                            }
                        # required_next_data がある場合は Decision Maker に任せる
                    # skip old duplicate list-building logic

        elif action_type == "conclude":
            logger.info("[AE] 最終分析呼び出し (現在はダミー)")
            # await asyncio.sleep(1) # ダミー待機は不要かも
            final_hypotheses_summary = _summarize_hypotheses(state.get('current_hypotheses', []))
            dummy_report = {
                "is_anomaly": False, # これは分析結果に基づくべき
                "is_data_sufficient": True, # これも分析結果に基づくべき
                "analysis": f"調査が完了しました。\n{final_hypotheses_summary}",
                "summary_of_findings": [h for h in state.get('current_hypotheses', []) if h['status'] in ['supported', 'rejected']],
                "recommendations": "定期的なモニタリング継続を推奨します。" # これも分析結果に基づくべき
            }
            observation = {'final_report': dummy_report}
            state['final_result'] = dummy_report
            logger.info("最終分析完了 (ダミー)。")
            # state['next_action'] = None # conclude が最終アクションなのでクリアは不要かも -> 最後にまとめてクリア
        elif action_type == "error":
            error_msg = parameters.get('message', 'Undefined error occurred')
            logger.error(f"[AE] Error action executed: {error_msg}")
            observation = {'error': error_msg, 'forced_error': True} # 強制エラーであることを示す
            state['error_message'] = error_msg
            error_occurred = True # エラーフラグを立てる
            # state['next_action'] = None # 最後にまとめてクリア
        elif action_type == "refine_hypothesis":
            hypothesis_id = parameters.get("hypothesis_id")
            h = next((h for h in state['current_hypotheses'] if h['id'] == hypothesis_id), None)
            if not h:
                logger.error(f"refine_hypothesis: 指定ID {hypothesis_id} の仮説が見つかりません")
                observation = {'refined_hypotheses': [], 'based_on': hypothesis_id, 'error': 'not found'}
            else:
                prompt = get_refine_hypothesis_prompt(h)
                logger.info("refine_hypothesis: サブ仮説生成をLLMに依頼")
                refined = await llm.ainvoke(prompt)
                try:
                    if not isinstance(refined, list):
                        refined = []
                    for h in refined:
                        if 'status' not in h or h['status'] is None:
                            h['status'] = 'new'
                except Exception as e:
                    logger.warning(f"refine_hypothesis: LLM応答パース失敗: {e}")
                    refined = []
                # 既存IDでユニーク化して追加
                current_hypotheses = state.get('current_hypotheses', [])
                hypotheses_dict = {h['id']: h for h in current_hypotheses if 'id' in h}
                for h_new in refined:
                    if 'id' not in h_new or not isinstance(h_new.get('id'), str):
                        logger.warning(f"Skipping refined hypothesis due to missing or invalid ID: {h_new}")
                        continue
                    if 'text' not in h_new or 'priority' not in h_new:
                        logger.warning(f"Skipping refined hypothesis due to missing required fields (text, priority): {h_new}")
                        continue
                    h_new.setdefault('status', 'new')
                    hypotheses_dict[h_new['id']] = h_new
                state['current_hypotheses'] = list(hypotheses_dict.values())
                observation = {'refined_hypotheses': refined, 'based_on': hypothesis_id}
            state['cycles_since_last_hypothesis'] = state.get('cycles_since_last_hypothesis', 0) - 1
        elif action_type == "eda":
            logger.info("[AE] EDA 実行")
            # stateのhistoryから、最後の'query_data_agent'のobservationを取得
            last_query_data_agent_observation = next((
                o for o in reversed(state['history'])
                if o['type'] == 'observation'
                and isinstance(o.get('content'), dict)
                and o['content'].get('action_type') == 'query_data_agent'
            ), None)
            if last_query_data_agent_observation:
                # refined_queryの内容を取得
                refined_query = last_query_data_agent_observation['content']['refined_query']
                # dataの内容を取得
                data = last_query_data_agent_observation['content']['raw_data']

                # 将来的にはrefined_queryを使って処理した結果をデータとして渡す
                response = await llm.ainvoke(get_eda_prompt(refined_query, data))

                state['eda_summary'] = response['eda_result']
                state['collected_data_summary']['eda_result'] = response['eda_result']
                state['eda_stats'] = ""
                state['next_action'] = {'action_type': 'generate_hypothesis', 'parameters': {}}

                # historyにnodeを追加
                state['history'] = state.get('history', []) + [{
                    'type': 'node',
                    'content': {'node_name': 'eda'},
                    'timestamp': datetime.datetime.now().isoformat()
                }]
                # observationにeda_resultを追加
                observation = {'eda_result': response['eda_result']}
                logger.info(f"[AE] EDA 終了: {response['eda_result']}")
                
            else:
                logger.error("[AE] EDA 実行: 最後のquery_data_agentのobservationが見つかりません")
        elif action_type == "data_analysis":
            logger.info("[AE] データ分析 実行")
            # stateのhistoryから、最後の'query_data_agent'のobservationを取得
            last_query_data_agent_observation = next((
                o for o in reversed(state['history'])
                if o['type'] == 'observation'
                and isinstance(o.get('content'), dict)
                and o['content'].get('action_type') == 'query_data_agent'
            ), None)
            if last_query_data_agent_observation:
                query = last_query_data_agent_observation['content']['query']
                refined_query = last_query_data_agent_observation['content']['refined_query']
                data = last_query_data_agent_observation['content']['raw_data']

                # 将来的にはrefined_queryを使って処理した結果をデータとして渡す
                response = await llm.ainvoke(get_data_analysis_prompt(query, refined_query, data))

                # データ分析直後は最新データを用いて直ちに仮説を再評価する
                if currently_investigating_hypothesis_id:
                    state['next_action'] = {
                        'action_type': 'evaluate_hypothesis',
                        'parameters': {'hypothesis_id': currently_investigating_hypothesis_id}
                    }
                else:
                    # フォーカスがない場合は Decision Maker に任せる
                    state['next_action'] = None

                # historyにnodeを追加
                state['history'] = state.get('history', []) + [{
                    'type': 'node',
                    'content': {'node_name': 'data_analysis'},
                    'timestamp': datetime.datetime.now().isoformat()
                }]
                # observationにeda_resultを追加
                observation = {'data_analysis_result': response['data_analysis_result'], 'reasoning': response['reasoning']}
                logger.info(f"[AE] データ分析 終了: {str(response)}")
                
            else:
                logger.error("[AE] データ分析 実行: 最後のquery_data_agentのobservationが見つかりません")
    except Exception as e:
        # 各アクション処理中の予期せぬ例外
        logger.error(f"[AE] Action Executor: Unhandled error executing action {action_type}: {e}", exc_info=True)
        observation = {'error': f"Failed to execute action {action_type}: {e}"}
        error_occurred = True
        state['error_message'] = str(e)

    if observation: # observation が生成されていれば履歴に追加
        state['history'] = state.get('history', []) + [{
            'type': 'observation',
            'hypothesis_id': currently_investigating_hypothesis_id,
            'content': observation,
            'timestamp': datetime.datetime.now().isoformat()
        }]
        # collected_data_summaryをイミュータブルに更新
        collected_data_summary = dict(state.get('collected_data_summary', {}))
        if currently_investigating_hypothesis_id in collected_data_summary:
            existing = collected_data_summary[currently_investigating_hypothesis_id]
            if isinstance(existing, list):
                collected_data_summary[currently_investigating_hypothesis_id] = existing + [observation]
            else:
                collected_data_summary[currently_investigating_hypothesis_id] = [existing, observation]
        else:
            collected_data_summary[currently_investigating_hypothesis_id] = observation
        state['collected_data_summary'] = collected_data_summary
    elif not error_occurred:
         # observation が None だがエラーでもない場合 (アクションが何も生成しなかった場合など)
         logger.warning(f"[AE] No observation generated for action '{action_type}', and no error reported.")
         # ダミーの observation を追加しても良いかもしれない
         state['history'] = state.get('history', []) + [{
             'type': 'observation',
             'hypothesis_id': currently_investigating_hypothesis_id,
             'content': {'message': f"Action '{action_type}' completed without specific output."},
             'timestamp': datetime.datetime.now().isoformat()
         }]

    # 次アクションが同じまま（= 新たにセットされなかった）か None の場合はクリア
    if state.get('next_action') is None or state['next_action'] == original_next_action:
        state['next_action'] = None

    # 仮説評価でクローズした場合は全hypステータス出力
    if action_type == "evaluate_hypothesis":
        current_hypotheses = state.get('current_hypotheses', [])
        _log_hypotheses_status(current_hypotheses, prefix="[AE][HYP_STATUS_AFTER_EVAL]")

    return state 

# ------------------ Verification Nodes ------------------


async def plan_verification_node(state: DynamicAgentState, llm):
    """LLM に裏どりプランを作成させるノード"""
    logger = logging.getLogger(__name__)
    logger.info("[PLAN_VERIF] --- Node: Plan Verification ---")

    hyp_id = state.get('currently_investigating_hypothesis_id')
    if not hyp_id:
        logger.error("[PLAN_VERIF] No focused hypothesis for verification plan.")
        state['error_message'] = 'No hypothesis in focus for verification planning.'
        return state

    hyp = next((h for h in state.get('current_hypotheses', []) if h['id'] == hyp_id), None)
    if not hyp:
        logger.error(f"[PLAN_VERIF] Hypothesis {hyp_id} not found.")
        state['error_message'] = f'Hypothesis {hyp_id} not found for verification.'
        return state

    prompt_template = get_plan_verification_prompt(hyp, state['objective'], state.get('available_data_agents_and_skills', []))
    plan_data = await llm.ainvoke(prompt_template)
    verification_plan = plan_data.get('verification_plan', [])

    if not verification_plan:
        logger.warning("[PLAN_VERIF] LLM returned empty verification plan. Marking verification disproved.")
        hyp.setdefault('verification', {})
        hyp['verification'].update({'status': 'disproved', 'reasoning': 'LLM returned no viable verification steps.'})
        state['next_action'] = None
        return state

    state['current_verification_steps'] = verification_plan
    state['verification_repeat_count'] = 0
    # push first execute
    first_step = verification_plan[0]
    state['next_action'] = {
        'action_type': 'execute_verification_step',
        'parameters': first_step
    }

    logger.info(f"[PLAN_VERIF] Planned {len(verification_plan)} verification steps. Next: execute_verification_step {first_step.get('step_id')}")
    return state


async def execute_verification_step_node(state: DynamicAgentState, smart_a2a_client, llm):
    logger = logging.getLogger(__name__)
    logger.info("[EXEC_VERIF] --- Node: Execute Verification Step ---")

    params = state.get('next_action', {}).get('parameters', {})
    agent_skill_id = params.get('agent_skill_id')
    query = params.get('query')
    step_id = params.get('step_id')

    if not (agent_skill_id and query and step_id):
        logger.error("[EXEC_VERIF] Missing parameters for verification step execution.")
        state['error_message'] = 'Invalid verification step parameters.'
        return state

    # mimic query_data_agent logic but simpler
    try:
        prompt_template = get_query_refinement_prompt(query)
        refined_query_data = await llm.ainvoke(prompt_template)
        # choose first refined query for simplicity
        refined_query = refined_query_data['answer'][0]['required_data']['new'] if refined_query_data.get('answer') else query

        step = PlanStep(
            id=f"verif_{step_id}",
            description=f"Verification query {step_id}",
            skill_id=agent_skill_id,
            input_data={"input": str(refined_query), "return_type": "df"},
            parameters=params
        )
        step_result = await execute_step(step)

        text = str(step_result.output_data) if hasattr(step_result, 'output_data') else ''

        # store result
        verif_results = state.get('verification_results', {})
        verif_results[step_id] = {
            'refined_query': refined_query,
            'raw': text
        }
        state['verification_results'] = verif_results

        # add to evidence list
        hyp_id = state.get('currently_investigating_hypothesis_id')
        hyp = next((h for h in state.get('current_hypotheses', []) if h['id'] == hyp_id), None)
        if hyp:
            hyp.setdefault('verification', {})
            hyp['verification'].setdefault('evidence', []).append({'step_id': step_id, 'data': text})

    except Exception as e:
        logger.error(f"[EXEC_VERIF] Error executing verification step: {e}")
        state['error_message'] = str(e)
        return state

    # Decide next step
    remaining_steps = [s for s in state.get('current_verification_steps', []) if s['step_id'] not in state['verification_results']]
    if remaining_steps:
        next_step = remaining_steps[0]
        state['next_action'] = {
            'action_type': 'execute_verification_step',
            'parameters': next_step
        }
    else:
        state['next_action'] = {
            'action_type': 'summarize_verification',
            'parameters': {}
        }

    return state


async def summarize_verification_node(state: DynamicAgentState, llm):
    logger = logging.getLogger(__name__)
    logger.info("[SUM_VERIF] --- Node: Summarize Verification ---")
    hyp_id = state.get('currently_investigating_hypothesis_id')
    hyp = next((h for h in state.get('current_hypotheses', []) if h['id'] == hyp_id), None)
    if not hyp:
        logger.error("[SUM_VERIF] Hypothesis not found during summarization.")
        state['error_message'] = 'Hypothesis not found for verification summarization.'
        return state

    prompt_template = get_summarize_verification_prompt(hyp, state.get('verification_results', {}))
    summary = await llm.ainvoke(prompt_template)

    ver_status = summary.get('verification_status')
    risk_score = summary.get('risk_score')
    reasoning = summary.get('reasoning')

    hyp.setdefault('verification', {})
    hyp['verification'].update({'status': ver_status, 'risk_score': risk_score, 'reasoning': reasoning})

    logger.info(f"[SUM_VERIF] Verification status for {hyp_id}: {ver_status} (risk_score={risk_score})")

    # After summarization, decide follow-up
    if ver_status in ['verified', 'disproved']:
        # --- TEMP CLOSE LOGIC: verification 完了で即 conclude --------------------------
        logger.info("[SUM_VERIF] Verification completed (status: %s). Concluding analysis.", ver_status)
        state['next_action'] = {
            'action_type': 'conclude',
            'parameters': {}
        }
        # Reset focus since process is concluding
        state['currently_investigating_hypothesis_id'] = None
    else:
        state['next_action'] = {
            'action_type': 'decision_maker',
            'parameters': {}
        }

    return state 