import logging
from typing import Any, Dict
import datetime
from ..core.node_base import Node, NodeResult
from ..core.node_base import make_history_entry
from ..utils import _all_hypotheses_resolved
from ..prompts import get_decision_maker_prompt

logger = logging.getLogger(__name__)

# 定数（本来はconfigから取得すべき）
MAX_INCONCLUSIVE_BEFORE_GENERATE = 2
STAGNATION_THRESHOLD = 0.8
SIM_THRESHOLD = 3
EVAL_REPEAT_LIMIT = 10
LANGGRAPH_RECURSION_LIMIT = 25 # LangGraphのデフォルト再帰制限

class DecisionMakerNode(Node):
    id = "decision_maker"

    async def run(self, state: Dict[str, Any], toolbox):
        logger.info("--- Node: DecisionMaker ---")
        patch: Dict[str, Any] = {}
        events = []
        # 子グラフ用: target_hypothesis_idが指定されていればcurrent_hypothesesをフィルタ
        target_hypothesis_id = state.get('target_hypothesis_id')
        if target_hypothesis_id:
            filtered = [h for h in state.get('current_hypotheses', []) if h.get('id') == target_hypothesis_id]
            state = state.copy()
            state['current_hypotheses'] = filtered
        max_queries_without_hypothesis = state.get('max_queries_without_hypothesis', 2)
        cycles_since_last_hypothesis = state.get('cycles_since_last_hypothesis', 0)
        consecutive_query_count = state.get('consecutive_query_count', 0)
        current_hypotheses = state.get('current_hypotheses', [])
        currently_investigating_hypothesis_id = state.get('currently_investigating_hypothesis_id')
        next_action = state.get('next_action')

        # --- 1. stateのhistoryのうち、"node"の数が95を超えたらerror ---
        node_count = sum(1 for entry in state.get('history', []) if entry.get('type') == 'node')
        if node_count > 80:
            patch = {"next_action": {"action_type": "error", "parameters": {"message": "historyのnodeの数が95を超えました。"}}}
            return NodeResult(observation="history_error", patch=patch, events=events)


        # --- next_actionがセットされていれば、そのままpatchで返す ---
        if next_action and "action_type" in next_action and not(next_action['action_type'] == 'decision_maker'):
            logger.info(f"[DecisionMaker] next_action: {next_action}")
            patch = {"next_action": next_action}
            events.append(make_history_entry(
                "node",
                {"name": "decision_maker", "action": f"direct:{next_action['action_type']}"},
                state
            ))
            return NodeResult(observation=f"direct_{next_action['action_type']}", patch=patch, events=events)


        # --- 2.5 Supported仮説からの追加仮説生成 ---
        if currently_investigating_hypothesis_id:
            # 現在調査中の仮説が「supported」ステータス、または「supporting」仮説生成が必要な場合のみ追加仮説生成をトリガー
            focused_hypothesis = next((h for h in current_hypotheses if h.get('id') == currently_investigating_hypothesis_id), None)
            if focused_hypothesis and focused_hypothesis.get('status') in ("supported", "needs_revision"):
                # 紐づく追加仮説が存在するかチェック
                has_supporting_hypothesis = any(
                    h.get('parent_hypothesis_id') == currently_investigating_hypothesis_id for h in current_hypotheses
                )

                # IDの文字列にsubがいくつ含まれるか算出
                sub_count = focused_hypothesis.get('id').count('sub')

                if not has_supporting_hypothesis and sub_count < 2 and focused_hypothesis.get('status') == "supported":
                    logger.info(f"[DM] Hypothesis {currently_investigating_hypothesis_id} (supported) に基づき追加仮説生成をトリガーします。")
                    patch = {
                        "next_action": {
                            "action_type": "generate_hypothesis", # 既存ノードを利用
                            "parameters": {
                                "parent_hypothesis_id": currently_investigating_hypothesis_id
                            }
                        },
                        "last_supported_hypothesis_id": None  # フラグをクリア
                    }
                    events.append(make_history_entry(
                        "thought",
                        f"仮説 {currently_investigating_hypothesis_id} の支持に基づき、追加仮説を生成します。",
                        state
                    ))
                    return NodeResult(observation=f"trigger_support_gen_for_{currently_investigating_hypothesis_id}", patch=patch, events=events)

                if focused_hypothesis.get('status') == "needs_revision":
                    if not has_supporting_hypothesis:
                        logger.info(f"[DM] Hypothesis {currently_investigating_hypothesis_id} (needs_revision)のrefine判定")
                    if sub_count > 1 or has_supporting_hypothesis:
                        logger.info(f"[DM] Hypothesis {currently_investigating_hypothesis_id} (needs_revision)のフォーカスを解除")
                        currently_investigating_hypothesis_id = None

        # --- 2. 全仮説解決済みならconclude ---
        if _all_hypotheses_resolved(current_hypotheses):
            logger.info("[DM] 全仮説が解決済み。concludeへ遷移")
            patch = {
                "next_action": {"action_type": "conclude", "parameters": {}}
            }
            events.append(make_history_entry(
                "node",
                {"name": "decision_maker", "action": "conclude"},
                state
            ))
            return NodeResult(observation="conclude", patch=patch, events=events)



        # --- 5. フォーカス仮説の選定/解除 ---
        focus_candidates = [h for h in current_hypotheses if h['status'] in ['inconclusive', 'new']]
        focused_hypothesis = None
        if currently_investigating_hypothesis_id:
            focused_hypothesis = next((h for h in current_hypotheses if h['id'] == currently_investigating_hypothesis_id), None)
            if not focused_hypothesis or focused_hypothesis['status'] in ['supported', 'rejected']:
                logger.info(f"[DM] Focus hypothesis {currently_investigating_hypothesis_id} is resolved ({focused_hypothesis['status'] if focused_hypothesis else 'not found'}). Clearing focus.")
                logger.info(f"[DM] ★★all_hyp_status: {[hyp['id'] + ':' + hyp['status'] for hyp in current_hypotheses]}")
                currently_investigating_hypothesis_id = None
                focused_hypothesis = None
        if not focused_hypothesis and focus_candidates:
            hypothesis_to_focus = sorted(focus_candidates, key=lambda h: -h.get('priority', 0))[0]
            patch = {"currently_investigating_hypothesis_id": hypothesis_to_focus['id'],
                     "next_action": {"action_type": "evaluate_hypothesis", "parameters": {"hypothesis_id": hypothesis_to_focus['id']}},
                     "cycles_since_last_hypothesis": 0
                    }
            events.append(make_history_entry(
                "thought",
                f"focus切替: {hypothesis_to_focus['id']}にフォーカス, current_iterationリセット, cycles_since_last_hypothesisリセット",
                state
            ))
            return NodeResult(observation="focus_switch", patch=patch, events=events)

        # --- 6. 停滞スコアによる強制分岐 ---
        stagnation_score = cycles_since_last_hypothesis
        if stagnation_score >= SIM_THRESHOLD:
            logger.info(f"[DM] stagnation_score={stagnation_score} 閾値超過。needs_revision/evaluate_hypothesis強制")
            if currently_investigating_hypothesis_id:
                # フォーカス仮説をneeds_revisionに
                patch = {"next_action": {"action_type": "evaluate_hypothesis", "parameters": {"hypothesis_id": currently_investigating_hypothesis_id}}}
                events.append(make_history_entry(
                    "thought",
                    f"stagnation_score超過により仮説{currently_investigating_hypothesis_id}をneeds_revision・再評価",
                    state
                ))
                return NodeResult(observation="stagnation_force_eval", patch=patch, events=events)
            # フォーカスなし→未解決仮説にフォーカス
            unresolved = [h for h in current_hypotheses if h['status'] in ['inconclusive', 'new']]
            if unresolved:
                next_focus = sorted(unresolved, key=lambda h: -h.get('priority', 0))[0]
                patch = {"currently_investigating_hypothesis_id": next_focus['id'],
                         "next_action": {"action_type": "evaluate_hypothesis", "parameters": {"hypothesis_id": next_focus['id']}},
                         "cycles_since_last_hypothesis": 0
                        }
                events.append(make_history_entry(
                    "thought",
                    f"stagnation_score超過で仮説{next_focus['id']}にフォーカス",
                    state
                ))
                return NodeResult(observation="stagnation_force_focus", patch=patch, events=events)
            # それもなければconclude
            patch = {"next_action": {"action_type": "conclude", "parameters": {}}}
            return NodeResult(observation="stagnation_force_conclude", patch=patch, events=events)

        # --- 7. 連続クエリ回数による新規仮説生成 ---
        if not focused_hypothesis and consecutive_query_count >= max_queries_without_hypothesis:
            patch = {"next_action": {"action_type": "generate_hypothesis", "parameters": {}}}
            events.append(make_history_entry(
                "thought",
                "連続クエリでgenerate_hypothesis強制",
                state
            ))
            return NodeResult(observation="force_generate_hypothesis_by_query", patch=patch, events=events)

        # --- 8. LLMによるアクション決定 ---
        if toolbox.llm and focused_hypothesis:
            available_actions = ["query_data_agent", "evaluate_hypothesis", "generate_hypothesis", "refine_hypothesis", "conclude", "error"]
            prompt = get_decision_maker_prompt(state, focused_hypothesis, currently_investigating_hypothesis_id, available_actions)
            
            decision_data = await toolbox.llm.ainvoke(prompt)

            logger.info("---decision--------------------")
            logger.info(f"[DM] LLM response: {decision_data}")
            logger.info("-------------------------------")
            thought = decision_data.get("thought", "No thought provided.")
            action = decision_data.get("action")
            if not action or "action_type" not in action:
                patch = {"next_action": {"action_type": "error", "parameters": {"message": "LLM response missing action/action_type"}}}
                events.append(make_history_entry(
                    "thought",
                    "LLM応答不正",
                    state
                ))
                return NodeResult(observation="llm_error", patch=patch, events=events)
            patch = {"next_action": action, "currently_investigating_hypothesis_id": currently_investigating_hypothesis_id}
            events.append(make_history_entry(
                "thought",
                thought,
                state
            ))
            return NodeResult(observation="llm_decision", patch=patch, events=events)

        # --- 2.5 仮説が全く無い場合はgenerate_hypothesisを強制 ---
        if not current_hypotheses:
            patch = {
                "next_action": {"action_type": "generate_hypothesis", "parameters": {}}
            }
            events.append(make_history_entry(
                "thought",
                "仮説が無いためgenerate_hypothesisを強制",
                state
            ))
            return NodeResult(observation="force_generate_hypothesis_no_hyp", patch=patch, events=events)

        # --- 9. どの分岐にも該当しない場合はconclude ---
        patch = {"next_action": {"action_type": "conclude", "parameters": {}},
                 "error_message": "conclude hypothesis result"
                }
        events.append(make_history_entry(
            "node",
            {"name": "decision_maker", "action": "conclude"},
            state
        ))
        return NodeResult(observation="conclude_hypothesis_result", patch=patch, events=events) 