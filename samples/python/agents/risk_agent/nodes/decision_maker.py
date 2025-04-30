import logging
from typing import Any, Dict
import datetime
from ..core.node_base import Node, NodeResult
from ..utils import _all_hypotheses_resolved, _unresolved_ratio, _count_inconclusive
from ..prompts import get_decision_maker_prompt

logger = logging.getLogger(__name__)

# 定数（本来はconfigから取得すべき）
MAX_INCONCLUSIVE_BEFORE_GENERATE = 2
STAGNATION_THRESHOLD = 0.8
SIM_THRESHOLD = 3
EVAL_REPEAT_LIMIT = 10

class DecisionMakerNode(Node):
    id = "decision_maker"

    async def run(self, state: Dict[str, Any], toolbox):
        logger.info("--- Node: DecisionMaker ---")
        patch = {}
        events = []
        # 主要state取得
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
        next_action = state.get('next_action')

        # --- next_actionがセットされていれば、そのままpatchで返す ---
        if next_action and "action_type" in next_action:
            logger.info(f"[DecisionMaker] next_action: {next_action}")
            patch = {"next_action": next_action}
            events.append({"type": "node", "name": "decision_maker", "action": f"direct:{next_action['action_type']}"})
            return NodeResult(observation=f"direct_{next_action['action_type']}", patch=patch, events=events)

        # --- 2. 全仮説解決済みならconclude ---
        if _all_hypotheses_resolved(current_hypotheses):
            logger.info("[DM] 全仮説が解決済み。concludeへ遷移")
            patch = {
                "next_action": {"action_type": "conclude", "parameters": {}},
                "current_iteration": current_iteration + 1
            }
            events.append({"type": "node", "name": "decision_maker", "action": "conclude"})
            return NodeResult(observation="conclude", patch=patch, events=events)

        # --- 3. イテレーション上限 ---
        if current_iteration >= max_iterations:
            logger.warning(f"[DM] 最大イテレーション到達。強制終了")
            patch = {
                "next_action": {"action_type": "error", "parameters": {"message": "Max iterations reached"}},
                "error_message": "Max iterations reached"
            }
            events.append({"type": "node", "name": "decision_maker", "action": "error"})
            return NodeResult(observation="max_iterations", patch=patch, events=events)

        # --- 4. 停滞スコアによる強制分岐 ---
        unresolved_ratio = _unresolved_ratio(current_hypotheses)
        stagnation_score = (current_iteration / max_iterations) * unresolved_ratio + (cycles_since_last_hypothesis / SIM_THRESHOLD)
        if stagnation_score >= STAGNATION_THRESHOLD:
            logger.info(f"[DM] stagnation_score={stagnation_score:.2f} 閾値超過。needs_revision/evaluate_hypothesis強制")
            if currently_investigating_hypothesis_id:
                # フォーカス仮説をneeds_revisionに
                patch = {"next_action": {"action_type": "evaluate_hypothesis", "parameters": {"hypothesis_id": currently_investigating_hypothesis_id}},
                         "current_iteration": current_iteration + 1}
                events.append({"type": "thought", "content": f"stagnation_score超過により仮説{currently_investigating_hypothesis_id}をneeds_revision・再評価", "timestamp": datetime.datetime.now().isoformat()})
                return NodeResult(observation="stagnation_force_eval", patch=patch, events=events)
            # フォーカスなし→未解決仮説にフォーカス
            unresolved = [h for h in current_hypotheses if h['status'] in ['inconclusive', 'new']]
            if unresolved:
                next_focus = sorted(unresolved, key=lambda h: -h.get('priority', 0))[0]
                patch = {"currently_investigating_hypothesis_id": next_focus['id'],
                         "next_action": {"action_type": "evaluate_hypothesis", "parameters": {"hypothesis_id": next_focus['id']}},
                         "current_iteration": current_iteration + 1}
                events.append({"type": "thought", "content": f"stagnation_score超過で仮説{next_focus['id']}にフォーカス", "timestamp": datetime.datetime.now().isoformat()})
                return NodeResult(observation="stagnation_force_focus", patch=patch, events=events)
            # それもなければconclude
            patch = {"next_action": {"action_type": "conclude", "parameters": {}}, "current_iteration": current_iteration + 1}
            return NodeResult(observation="stagnation_force_conclude", patch=patch, events=events)

        # --- 5. 仮説未解決・停滞回数による強制分岐 ---
        inconclusive_count = _count_inconclusive(current_hypotheses)
        if cycles_since_last_hypothesis >= MAX_INCONCLUSIVE_BEFORE_GENERATE or inconclusive_count >= MAX_INCONCLUSIVE_BEFORE_GENERATE:
            # --- ▼▼▼ required_next_data を考慮した判定を追加 ▼▼▼ ---
            focused_hypothesis_tmp = None
            if currently_investigating_hypothesis_id:
                focused_hypothesis_tmp = next((h for h in current_hypotheses if h['id'] == currently_investigating_hypothesis_id), None)

            if focused_hypothesis_tmp and focused_hypothesis_tmp.get('required_next_data'):
                # 必要なデータが判明している場合は強制分岐せず、LLM 判断に委ねる
                logger.info(
                    f"[DM] Inconclusive count ({inconclusive_count}) or cycles ({cycles_since_last_hypothesis}) reached threshold, "
                    f"but required_next_data exists for focus {focused_hypothesis_tmp['id']}. Proceeding to LLM decision.")
                # 強制分岐しないので処理を継続（後続の LLM セクションへ）
            else:
                logger.warning(
                    f"[DM] cycles_since_last_hypothesis={cycles_since_last_hypothesis}, "
                    f"inconclusive_count={inconclusive_count} で refine_hypothesis/generate_hypothesis を強制します。(No focus or no required_next_data)")

                if currently_investigating_hypothesis_id:
                    patch = {
                        "next_action": {
                            "action_type": "refine_hypothesis",
                            "parameters": {"hypothesis_id": currently_investigating_hypothesis_id}
                        },
                        "current_iteration": current_iteration + 1
                    }
                    events.append({
                        "type": "thought",
                        "content": "inconclusive/cycles超過でrefine_hypothesis",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    return NodeResult(observation="force_refine_hypothesis", patch=patch, events=events)
                else:
                    patch = {
                        "next_action": {"action_type": "generate_hypothesis", "parameters": {}},
                        "current_iteration": current_iteration + 1
                    }
                    events.append({
                        "type": "thought",
                        "content": "inconclusive/cycles超過でgenerate_hypothesis",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    return NodeResult(observation="force_generate_hypothesis", patch=patch, events=events)

        # --- 6. フォーカス仮説の選定 ---
        focus_candidates = [h for h in current_hypotheses if h['status'] in ['inconclusive', 'new']]
        focused_hypothesis = None
        if currently_investigating_hypothesis_id:
            focused_hypothesis = next((h for h in current_hypotheses if h['id'] == currently_investigating_hypothesis_id), None)
            if not focused_hypothesis or focused_hypothesis['status'] in ['supported', 'rejected']:
                currently_investigating_hypothesis_id = None
                focused_hypothesis = None
        if not focused_hypothesis and focus_candidates:
            hypothesis_to_focus = sorted(focus_candidates, key=lambda h: -h.get('priority', 0))[0]
            patch = {"currently_investigating_hypothesis_id": hypothesis_to_focus['id'],
                     "next_action": {"action_type": "evaluate_hypothesis", "parameters": {"hypothesis_id": hypothesis_to_focus['id']}},
                     "current_iteration": current_iteration + 1}
            events.append({"type": "thought", "content": f"focus切替: {hypothesis_to_focus['id']}にフォーカス", "timestamp": datetime.datetime.now().isoformat()})
            return NodeResult(observation="focus_switch", patch=patch, events=events)

        # --- 7. 連続クエリ回数による新規仮説生成 ---
        if not focused_hypothesis and consecutive_query_count >= max_queries_without_hypothesis:
            patch = {"next_action": {"action_type": "generate_hypothesis", "parameters": {}},
                     "current_iteration": current_iteration + 1}
            events.append({"type": "thought", "content": "連続クエリでgenerate_hypothesis強制", "timestamp": datetime.datetime.now().isoformat()})
            return NodeResult(observation="force_generate_hypothesis_by_query", patch=patch, events=events)

        # --- 8. LLMによるアクション決定 ---
        if toolbox.llm and focused_hypothesis:
            available_actions = ["query_data_agent", "evaluate_hypothesis", "generate_hypothesis", "refine_hypothesis", "conclude", "error"]
            prompt = get_decision_maker_prompt(state, focused_hypothesis, currently_investigating_hypothesis_id, available_actions)
            
            decision_data = await toolbox.llm.ainvoke(prompt)

            print("-----------------------")
            logger.info(f"[DM] LLM response: {decision_data}")
            print("-----------------------")
            thought = decision_data.get("thought", "No thought provided.")
            action = decision_data.get("action")
            if not action or "action_type" not in action:
                patch = {"next_action": {"action_type": "error", "parameters": {"message": "LLM response missing action/action_type"}}}
                events.append({"type": "thought", "content": "LLM応答不正", "timestamp": datetime.datetime.now().isoformat()})
                return NodeResult(observation="llm_error", patch=patch, events=events)
            patch = {"next_action": action, "current_iteration": current_iteration + 1}
            events.append({"type": "thought", "content": thought, "timestamp": datetime.datetime.now().isoformat()})
            return NodeResult(observation="llm_decision", patch=patch, events=events)

        # --- 2.5 仮説が全く無い場合はgenerate_hypothesisを強制 ---
        if not current_hypotheses:
            patch = {
                "next_action": {"action_type": "generate_hypothesis", "parameters": {}},
                "current_iteration": current_iteration + 1
            }
            events.append({"type": "thought", "content": "仮説が無いためgenerate_hypothesisを強制", "timestamp": datetime.datetime.now().isoformat()})
            return NodeResult(observation="force_generate_hypothesis_no_hyp", patch=patch, events=events)

        # --- 9. どの分岐にも該当しない場合はエラー ---
        patch = {"next_action": {"action_type": "error", "parameters": {"message": "Unexpected state in Decision Maker"}},
                 "error_message": "Unexpected state in Decision Maker"}
        events.append({"type": "node", "name": "decision_maker", "action": "error"})
        return NodeResult(observation="unexpected_state", patch=patch, events=events) 