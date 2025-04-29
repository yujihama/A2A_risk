# -*- coding: utf-8 -*-
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, RootModel, validator
from langchain.output_parsers import PydanticOutputParser
from typing import List, Dict, Any, TypedDict, Optional
import json
import os
import re
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from A2A_risk.samples.python.common.types import Message, DataPart
import logging

# ファイル冒頭付近に追加
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# agent.pyから必要なクラスをインポート
from samples.python.agents.smart_kakaku_signal.agent import ExecutionPlan, PlanStep, execute_step

class OutputModel(BaseModel):
    """
    兆候検知のための計画
    """
    id: str = Field(..., description='step1,2,3...必ずstepで始まる')
    skill_id: str = Field(..., description='利用可能なスキルidのいずれか')
    description: str = Field(..., description='このステップで実施したいこと、目的について説明')
    query: str = Field(..., description='スキルに渡す指示を明確かつ具体的に（例「AについてのXXXを回答してください」"')
    expected_output: str = Field(..., description='スキルからの回答に含めてほしい情報を明確かつ具体的に（例「ID、NAME、PRICEのカラムがあるXXXの一覧...」）')

    @validator('id')
    def id_must_start_with_step(cls, v):
        if not re.match(r'^step\d+$', v):
            raise ValueError('idは必ず"step"で始まり、その後に数字が続く形式にしてください (例: step1, step2)')
        return v

class PlanListWrapper(BaseModel):
    """
    計画リスト全体を内包するオブジェクト
    """
    plans: List[OutputModel] = Field(
        ..., description="OutputModel のリスト"
    )
  
class ClarityFeedbackItem(BaseModel):
    """明確性に関する個別の指摘事項"""
    step_id: Optional[str] = Field(None, description="指摘対象のステップID (例: 'step1', 'step2')。計画全体への指摘の場合はnull。")
    comment: str = Field(..., description="具体的で明確な指摘内容。")

class ClarityFeedbackResponse(BaseModel):
    """Clarity Checker Agentの応答を構造化するモデル"""
    feedback_items: List[ClarityFeedbackItem] = Field(description="明確性に関する指摘事項のリスト。問題がなければ空リスト。")

# --- 設定 (Settings) ---
if not os.environ.get("OPENAI_API_KEY"):
    print("警告: 環境変数 'OPENAI_API_KEY' が設定されていません。実行前に設定してください。")

# LLMモデル初期化 (Initialize LLM model)
try:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
except Exception as e:
    print(f"エラー: OpenAIモデルの初期化に失敗しました。APIキーが有効か確認してください。エラー詳細: {e}")
    llm = None

# 利用可能なスキル定義 (Available Skills Definition)
AVAILABLE_SKILLS = [
    {
        "id": "analyze_order",
        "name": "発注情報分析 (Order Information Analysis)",
        "description": "発注に関する情報を自然言語で分析します。発注ID,発注日,担当者ID,担当者名,取引先ID,取引先名,品目名,単価,数量,発注金額,稟議IDの情報を保持しています。\n以下の操作を組 み合わせた分析が可能です。\n  \"filter\",\"select\",\"sum\",\"mean\",\"count\",\"max\",\"min\",\"sort\",\"head\",\"tail\",\"group_by\"\n",
        "inputModes": ["text"],
        "outputModes": ["text"],
        "examples": [
            "2023年の総発注額はいくらですか？",
            "取引先Aに対する発注件数を教えてください",
            "発注量が最も多い取引先を教えてください"
        ]
    },
    {
        "id": "analyze",
        "name": "分析 (Analysis)",
        "description": "与えられた情報をもとに比較や分析を行います",
        "inputModes": ["text"],
        "outputModes": ["text"],
        "examples": [
            "Step1の結果「XXX」とStep2の結果「YYY」をもとに、AAAに該当するデータがあるかを判断してください",
            "「XXX」の結果「YYY」となりましたが、この結果は妥当ですか",
        ]
    }
]

# ルール定義 (Rules Definition)
RULES = """
重要な注意 (Important Notes):
1. 必ず「利用可能なスキル」に含まれるスキルID のみを使用してください。
2. スキルへの指示 (`query`) は、具体的かつ明確な指示で記載してください。
3. 「analyze」以外のスキルを選択する場合、各ステップは独立している前提で、queryの情報のみでスキルが結果を出力できるような指示にしてください。他のstepの結果を参照することはできません。
4. 「analyze」を選択する場合はどのstepの結果を使うのか明記してください。
5. 最終ステップはリスクシナリオの結論（異常の有無など）を出力するものにしてください。
"""

MAX_REVISIONS = 3 # 最大修正回数 (Maximum number of revisions)

MODEL = ChatOpenAI(model="gpt-4.1-mini")

@tool
def show_skill_list() -> str:
    "利用可能なスキルと仕様の一覧を回答します。"
    res = str(AVAILABLE_SKILLS)
    print("-----call show_skill_list-----")
    return res

@tool
def test_skill(skill:str,test_query:str) -> str:
    "特定のskillに対して質問をすることができます。計画策定にあたりskillでできることを具体的に理解するために使用できます。"
    res = ""
    if skill == "show_skill_list":
      res = "I can do anything,"
    elif skill == "analyze_order":
      res ="I can do anything,"
    return res
  
# --- 状態定義 (State Definition) ---
class AuditPlanState(TypedDict):
    """監査計画生成プロセスの状態を管理するクラス"""
    objective: str             # 監査目的（リスクシナリオ）- Audit objective (risk scenario)
    skills: List[Dict]         # 利用可能なスキルリスト - List of available skills
    rules: str                 # ルール定義 - Rules definition
    plan: Optional[List[Dict]] # 現在の監査計画JSON (辞書型リスト) - Current audit plan (list of dictionaries)
    rule_feedback: List[str]   # Rule Checkerからのフィードバック - Feedback from Rule Checker
    clarity_feedback: List[str]# Clarity Checkerからのフィードバック - Feedback from Clarity Checker
    revision_count: int        # 修正回数 - Revision count
    # --- 追加 ---
    phase: str                 # 現在のフェーズ（'initial_planning', 'hypothesis_generation', 'refinement_planning' など）
    exploration_data: Optional[Any] # 初期データ収集結果
    hypotheses: Optional[Any]       # 仮説リスト

# --- プロンプト生成関数 ---
def make_prompt(state: AuditPlanState, risk_scenario: str):
    phase = state.get('phase', 'initial_planning')
    if phase == 'initial_planning':
        logging.info("[プロンプト種別] データ探索用プロンプトを使用します。")
        prompt_for_exploration = f"""
あなたは内部監査人として、リスクシナリオの検証に先立ち、関連データの全体像や特徴を把握するための「データ探索計画」を作成するAIアシスタントです。

#task1:
以下のリスクシナリオに関連するデータについて、まずは全体像や傾向、異常値の有無などを把握するために有効と思われるスキルを選定してください。

## リスクシナリオ:
{risk_scenario}

#task2:
選定したスキルを使い、以下のような観点でデータ探索を行うためのステップを計画してください。
- 主要な統計量（合計、平均、最大・最小値、件数など）の把握
- 代表的なサンプルデータの抽出
- 異常値や外れ値の有無の確認
- データ分布や傾向の把握
- その他、仮説構築に役立つ情報の取得

各ステップのqueryは、誰が見ても明確で具体的な指示となるように記載してください。
expected_outputには、希望するカラムや出力形式を具体的に指定してください。

#利用可能なスキル
show_skill_listツールで確認してください。

{RULES}
"""
        return prompt_for_exploration
    elif state['revision_count'] > 0:
        logging.info("[プロンプト種別] 修正用プロンプトを使用します。")
        rule_fb = "\n".join(f"- {fb}" for fb in state['rule_feedback']) if state.get('rule_feedback') else "なし"
        clarity_fb = "\n".join(f"- {fb}" for fb in state['clarity_feedback']) if state.get('clarity_feedback') else "なし"
        prompt_for_fix_plan = f"""
あなたは内部監査人としてリスクシナリオを検証するための実行計画を作成するAIアシスタントです。
以下のtaskを実施してください。

#task1:
前回の担当者が以下のリスクシナリオを検知するために作成した作業計画をよく読んでください。
## リスクシナリオ:
{risk_scenario}

## 作業計画（前回の担当者が作成）
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

#task2:
作業計画について、レビュアから以下の指摘が来ています。
これらの指摘を**全て**反映して、計画したJSONの全体を回答してください。

## ルール指摘 (Rule Violations):\n{rule_fb}\n"
## 明確性指摘 (Clarity Issues):\n{clarity_fb}\n\n"

#{RULES}

#使用可能なツール
show_skill_listツールで確認してください。

"""
        return prompt_for_fix_plan
    else:
        logging.info("[プロンプト種別] 仮説検証・初回計画用プロンプトを使用します。")
        prompt_for_plan = f"""
あなたは内部監査人としてリスクシナリオを検証するための実行計画を作成するAIアシスタントです。
以下のtaskを実施してください。

#task1:
以下のリスクシナリオの検証に必要と思われるスキルを選定してください。

## リスクシナリオ:
{risk_scenario}

#task2:
スキルを組み合わせて、リスクシナリオの検証に必要なデータの取得から分析、結果生成までの一連のステップを計画してください。
スキルへの指示（query）は、必ず明確かつ具体的に、経緯を知らない人にもわかるように丁寧に記載してください。
またスキルから返してほしい情報（expected_output）にテーブルデータを含める場合は、必ず希望するカラムなどの構成を指定してください。

#{RULES}

#利用可能ななツール
show_skill_listツールで確認してください。

"""
        return prompt_for_plan

# --- エージェント関数 (Agent Functions) ---

def planner_agent(state: AuditPlanState) -> AuditPlanState:
    """計画を作成または修正するエージェント (Agent to create or revise the plan)"""
    logging.info("\n--- Planner Agent ---")
    # フェーズによって計画種別を明示
    phase = state.get('phase', 'initial_planning')
    if phase == 'initial_planning':
        logging.info("[計画種別] データ探索用計画の策定")
    elif phase == 'refinement_planning':
        logging.info("[計画種別] 仮説検証用計画の策定")
    else:
        logging.info(f"[計画種別] その他フェーズ: {phase}")
    if llm is None:
        logging.error("エラー: LLMが初期化されていないため、計画を作成できません。")
        return {**state, "plan": [{"error": "LLM not initialized"}], "rule_feedback": ["LLM初期化エラー"], "clarity_feedback": []}

    current_revision = state['revision_count']
    objective = state['objective']
    # スキル定義を整形してプロンプトに含める
    skills_str = json.dumps(state['skills'], ensure_ascii=False, indent=2)
    rules = state['rules']

    # LLMへの指示プロンプトを組み立てる
    prompt_for_plan = make_prompt(state, objective)

    # logging.debug(f"\nPlanner Prompt:\n{prompt_for_plan}\n") # デバッグ用にプロンプトを表示

    # LLMに計画生成を依頼
    TOOLS: list[BaseTool] = [show_skill_list,test_skill]
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "日本語で回答してください。OutputModelのリストを持つPlanListWrapperの形式でJSONを出力してください。"),
            ("placeholder", "{messages}"),
        ]
    )
    agent = create_react_agent(MODEL, TOOLS, prompt=prompt_template,response_format=PlanListWrapper) # debug=True, 
    result: PlanListWrapper = agent.invoke({"messages": [("user", prompt_for_plan)]})
    plans = result["structured_response"].plans
    dicts = [plan.model_dump() for plan in plans]

    logging.info("--------------------------------")
    for d in dicts:
        logging.info(d)
    logging.info("--------------------------------")

    logging.info("計画生成/修正 完了 (Plan generation/revision complete)")
    # 状態を更新して返す (フィードバックはリセット)
    return {
        **state,
        "plan": dicts,
        "rule_feedback": [],
        "clarity_feedback": [],
    }

def rule_checker_agent(state: AuditPlanState) -> AuditPlanState:
    """計画がルールに準拠しているかチェックするエージェント (Agent to check plan against rules)"""
    logging.info("--- Rule Checker Agent ---")
    if llm is None:
        logging.error("エラー: LLMが初期化されていないため、ルールチェックを実行できません。")
        return {**state, "rule_feedback": ["LLM初期化エラー"]}
    if not isinstance(state.get('plan'), list) or not state['plan'] or 'error' in state['plan'][0]:
         logging.info("計画が存在しないかエラーのためスキップ (Skipping rule check due to missing/error plan)")
         if state.get('plan') and 'error' in state['plan'][0]:
              return state
         else:
              return {**state, "rule_feedback": ["チェック対象の計画が存在しません。"]}

    plan_str = json.dumps(state['plan'], ensure_ascii=False, indent=2)
    skills = [s['description'] for s in state['skills']]
    rules = state['rules'] if state['rules'] else ""

    # LLMへの指示プロンプト
    prompt = (
        "あなたは監査計画のルールチェックボットです。\n"
        "以下の計画JSONが、指定されたルールに準拠しているか確認してください。\n\n"
        f"計画 (Plan):\n```json\n{plan_str}\n```\n\n"
        f"ルール (Rules):\n{rules}\n\n"
        f"利用可能なスキル (Available Skill): {skills} ※analyzeも利用可能\n\n"
        "ルール違反のある箇所についてのみ、ステップIDと具体的な違反内容を箇条書きでリストアップしてください。修正案も記載してください。\n"
        "違反していない箇所については回答不要です。"
        "また一つも違反がなければ「違反なし」とだけ回答してください。"
    )
    logging.debug(f"\nRule Checker Prompt:\n{prompt}\n") # デバッグ用

    try:
        # LLMにルールチェックを依頼
        response = llm.invoke(prompt).content
        # logging.debug(f"\nRule Checker Raw Response:\n{response}\n") # デバッグ用
        # logging.debug(f"ルールチェック結果 (Rule Check Result):\n{response}")
        # 応答からフィードバックリストを抽出
        if "違反なし" in response or response.strip() == "":
            feedback = []
            logging.info("違反なし")
        else:
            # 簡単なパース処理（箇条書き "-" で始まる行を抽出）
            feedback = [line.strip('- ').strip() for line in response.split('\n') if line.strip().startswith('-')]
            if not feedback and response.strip() != "": # パース失敗時など、空でない応答はそのままフィードバックとする
                 feedback = [response.strip()]
            logging.info(f"rule_checker_agent結果: {feedback}")
        return {**state, "rule_feedback": feedback}
    except Exception as e:
        logging.error(f"エラー: Rule Checker Agent実行中にエラー発生: {e}")
        return {**state, "rule_feedback": [f"ルールチェック中にエラーが発生しました: {e}"]}

def clarity_checker_agent(state: AuditPlanState) -> AuditPlanState:
    """計画の明確性・具体性をチェックするエージェント (Agent to check plan clarity and specificity)"""
    logging.info("--- Clarity Checker Agent ---")
    if llm is None: return {**state, "clarity_feedback": ["LLM初期化エラー"]}
    if not isinstance(state.get('plan'), list) or not state['plan'] or 'error' in state['plan'][0]:
         logging.info("計画が存在しないかエラーのためスキップ")
         return {**state, "clarity_feedback": state.get('clarity_feedback', [])}

    plan_str = json.dumps(state['plan'], ensure_ascii=False, indent=2)
    available_skills_str = json.dumps(state['skills'], ensure_ascii=False, indent=2)

    # ★ Pydanticモデルで構造化出力するLLMを準備
    try:
        structured_llm = llm.with_structured_output(ClarityFeedbackResponse)
    except Exception as e:
        logging.error(f"エラー: 構造化出力LLMの準備に失敗: {e}")
        return {**state, "clarity_feedback": [f"構造化出力LLM準備エラー: {e}"]}

    # LLMへの指示プロンプト
    prompt = (
        "あなたは監査計画のレビュー担当です（計画作成の経緯は知りません）。\n"
        "以下の計画JSONの各ステップについて、記述が明確で具体的か、ステップ間のつながりが論理的か懐疑的な視点で評価してください。\n\n"
        f"計画 (Plan):\n```json\n{plan_str}\n```\n\n"
        "チェック観点 (Checkpoints):\n"
        "- query: スキルへの指示は曖昧さがなく、第三者が見ても何をすべきか明確か？誰が見ても解釈が一致する指示になっているか？「analyze」以外のスキルの場合、各ステップのみで完結する記載になっていること。\n"
        "- expected_output: 期待される結果の形式（例：カラム名）は具体的に示されているか？\n"
        "- query/expected_output: 以下の各スキルで回答可能な範囲の指示になっているか？\n"
        f"-- スキル一覧:\n{available_skills_str}\n"
        "ステップIDと共に指摘内容のみを具体的かつ簡潔に指摘内容を構造化して回答してください。記載の修正、ステップの統合・細分化などの修正案も記載してください。\n"
        "問題がなければ、feedback_itemsを空リストにして回答してください。"
    )
    logging.debug(f"\nClarity Checker Prompt:\n{prompt}\n") # デバッグ用

    try:
        # ★ 構造化LLMを呼び出し
        response_obj: ClarityFeedbackResponse = structured_llm.invoke(prompt)
        logging.info(f"clarity_checker_agent結果: {response_obj.model_dump()}") # Pydanticオブジェクトを辞書で表示

        # ★ Pydanticオブジェクトからフィードバックリスト(文字列)を生成してstateに格納
        feedback_list = []
        if response_obj.feedback_items:
            for item in response_obj.feedback_items:
                step_prefix = f"step{item.step_id}: " if item.step_id is not None else "全体: "
                feedback_list.append(f"{step_prefix}{item.comment}")

        count = state["revision_count"] + 1
        return {**state, "clarity_feedback": feedback_list, "revision_count":count}
    except Exception as e:
        # .invoke() が失敗した場合や、LLMがスキーマに従わない応答をした場合など
        logging.error(f"エラー: Clarity Checker Agent (構造化出力) 実行中にエラー発生: {e}")
        # エラー時はフォールバックとして空リストまたはエラーメッセージを設定可能
        return {**state, "clarity_feedback": [f"明確性チェック(構造化)中にエラーが発生しました: {e}"]}

# --- 条件分岐 (Conditional Edge Logic) ---
def should_continue(state: AuditPlanState) -> str:
    """修正が必要か、終了するかを判断する (Decide whether to continue revision or end)"""
    rule_feedback = state.get('rule_feedback', [])
    clarity_feedback = state.get('clarity_feedback', [])
    revision_count = state.get('revision_count', 0)
    logging.info(f"should_continue: rule_feedback={rule_feedback}, clarity_feedback={clarity_feedback}, revision_count={revision_count}")

    # 計画生成/パース自体でエラーが発生した場合は終了
    if isinstance(state.get('plan'), list) and state['plan'] and 'error' in state['plan'][0]:
         error_msg = state['plan'][0].get('error', 'Unknown error')
         logging.info(f"計画生成/パースエラーのため終了します。エラー: {error_msg}")
         return END

    # フィードバックがあり、かつ最大修正回数未満の場合
    if (rule_feedback or clarity_feedback) and revision_count < MAX_REVISIONS:
        logging.info(f"フィードバックあり (Feedback found: Rules={len(rule_feedback)}, Clarity={len(clarity_feedback)}), revision_count={revision_count}. 修正のためPlannerに戻ります。")
        return "planner"
    # 最大修正回数に到達した場合
    elif revision_count >= MAX_REVISIONS:
         logging.info(f"最大修正回数({MAX_REVISIONS})に到達しました。終了します。")
         return END
    # フィードバックがない場合
    else:
        logging.info("フィードバックなし。計画は承認されました。終了します。")
        return END

# --- DynamicPlanGenerator クラス ---
class DynamicPlanGenerator:
    def __init__(self, llm_client, registry):
        self.llm_client = llm_client
        self.registry = registry
        self.workflow = self._build_workflow()
        try:
            self.app = self.workflow.compile()
            logging.info("\nワークフローのコンパイル完了 (Workflow compiled successfully)")
        except Exception as e:
            logging.error(f"エラー: ワークフローのコンパイルに失敗しました: {e}")
            self.app = None

    def _build_workflow(self):
        """LangGraphワークフローを構築する（フェーズ制御付き新構造）"""
        workflow = StateGraph(AuditPlanState)

        # --- ノード追加 ---
        workflow.add_node("planner", planner_agent)
        workflow.add_node("rule_checker", rule_checker_agent)
        workflow.add_node("clarity_checker", clarity_checker_agent)
        # exploration_executorはasync/registry渡し対応
        async def exploration_executor_with_registry(state):
            return await exploration_executor(state, registry=self.registry)
        workflow.add_node("exploration_executor", exploration_executor_with_registry)
        workflow.add_node("hypothesis_generator", hypothesis_generator)
        workflow.add_node("hypothesis_evaluator", hypothesis_evaluator)

        # --- エントリーポイント ---
        workflow.set_entry_point("planner")

        # --- フェーズ遷移制御 ---
        # planner → exploration_executor（初期計画後、初期データ収集へ）
        def phase_router_after_planner(state: AuditPlanState) -> str:
            phase = state.get("phase", "initial_planning")
            if phase == "initial_planning":
                return "exploration_executor"
            elif phase == "refinement_planning":
                return "rule_checker"
            else:
                return "exploration_executor"  # デフォルト

        workflow.add_conditional_edges(
            "planner",
            phase_router_after_planner,
            {
                "exploration_executor": "exploration_executor",
                "rule_checker": "rule_checker"
            }
        )

        # exploration_executor → hypothesis_generator
        workflow.add_edge("exploration_executor", "hypothesis_generator")
        # hypothesis_generator → hypothesis_evaluator (一時的に削除)
        # workflow.add_edge("hypothesis_generator", "hypothesis_evaluator")
        # hypothesis_generator → END (追加)
        workflow.add_edge("hypothesis_generator", END)

        # planner（詳細計画）→ rule_checker → clarity_checker（従来ループ）
        workflow.add_edge("rule_checker", "clarity_checker")
        workflow.add_conditional_edges(
            "clarity_checker",
            should_continue,
            {
                "planner": "planner",
                END: END
            }
        )
        return workflow

    def _convert_workflow_result_to_execution_plan(
        self,
        final_state: AuditPlanState,
        parameters: Dict[str, Any],
        scenario_text: str
    ) -> ExecutionPlan:
        """LangGraphの最終状態をExecutionPlanに変換する"""
        if not final_state or not isinstance(final_state.get('plan'), list) or not final_state['plan'] or 'error' in final_state['plan'][0]:
            logging.error("エラー: 最終状態に有効な計画が含まれていません。")
            # エラーを含むExecutionPlanを返すか、例外を送出するかは要件による
            error_msg = final_state['plan'][0].get('error', 'Plan generation failed') if final_state.get('plan') else 'Plan not found in final state'
            return ExecutionPlan(
                 plan_id=str(uuid4()),
                 # parametersから取得できる情報を設定
                 product_id=parameters.get("product_id", "unknown"),
                 threshold=float(parameters.get("threshold", 120.0)), # 例: デフォルト120%
                 steps=[],
                 is_completed=False,
                 is_executed=False,
                 is_anomaly_detected=None,
                 anomaly_details=f"計画生成エラー: {error_msg}",
                 available_skills=AVAILABLE_SKILLS # 利用可能スキル情報を保持
            )

        steps = []
        for plan_item in final_state['plan']:
            # OutputModelからPlanStepへの変換
            step = PlanStep(
                id=plan_item["id"],
                description=plan_item["description"],
                skill_id=plan_item["skill_id"],
                # queryをinput_dataの"input"キーに入れる
                input_data={"input": plan_item["query"]},
                # expected_outputをリスト形式でoutput_dataに入れる
                # 将来的に複数の期待値に対応する場合はリストが適切
                output_data=[plan_item["expected_output"]],
                # parametersは現状OutputModelにないので空辞書
                parameters={},
                # ワークフロー完了時は is_completed=False (実行前のため)
                is_completed=False,
                error=None,
                selected_agent=None,
                start_time=None,
                # transitionsはplan_to_langgraph.pyで解釈されるためここでは設定不要
                transitions={}
            )
            steps.append(step)

        # ExecutionPlanオブジェクトを作成して返す
        return ExecutionPlan(
            plan_id=str(uuid4()),
            # parametersから取得できる情報を設定
            product_id=parameters.get("product_id", "unknown"), # キーが存在しない場合のデフォルト値
            threshold=float(parameters.get("threshold", 120.0)), # 例: デフォルト120%
            steps=steps,
            current_step_index=0,
            is_completed=False, # 計画生成直後は未完了
            is_executed=False, # 計画生成直後は未実行
            is_anomaly_detected=None, # 実行前は未検出
            anomaly_details=None,
            # created_atはExecutionPlanのデフォルトファクトリで設定される
            available_skills=AVAILABLE_SKILLS # 利用可能スキル情報を保持
            # scenario_text は ExecutionPlan にフィールドがないため設定しない
        )

    async def generate_execution_plan(
        self,
        scenario_analysis: Dict[str, Any],
        parameters: Dict[str, Any],
        scenario_text: str
    ) -> ExecutionPlan:
        """LangGraphワークフローを実行し、ExecutionPlanを生成する"""
        if self.app is None:
            logging.error("\nエラー: ワークフローがコンパイルされていないため、実行できません。")
            # エラーを示すExecutionPlanを返す
            return ExecutionPlan(
                 plan_id=str(uuid4()),
                 product_id=parameters.get("product_id", "unknown"),
                 threshold=float(parameters.get("threshold", 120.0)),
                 steps=[],
                 is_completed=False, is_executed=False, is_anomaly_detected=None,
                 anomaly_details="Workflow compilation failed",
                 available_skills=AVAILABLE_SKILLS
            )
        if llm is None:
             logging.error("\nエラー: LLMが初期化されていないため、実行できません。")
             return ExecutionPlan(
                 plan_id=str(uuid4()),
                 product_id=parameters.get("product_id", "unknown"),
                 threshold=float(parameters.get("threshold", 120.0)),
                 steps=[],
                 is_completed=False, is_executed=False, is_anomaly_detected=None,
                 anomaly_details="LLM initialization failed",
                 available_skills=AVAILABLE_SKILLS
             )

        # 初期状態を設定 (Set initial state)
        initial_state = AuditPlanState(
            objective=scenario_text, # シナリオテキストを目的として設定
            skills=AVAILABLE_SKILLS, # グローバル定数を参照
            rules=RULES,             # グローバル定数を参照
            plan=None,               # 初回は計画なし
            rule_feedback=[],
            clarity_feedback=[],
            revision_count=0,
            # --- 追加 ---
            phase='initial_planning',
            exploration_data=None,
            hypotheses=None
        )

        logging.info("\n--- 監査計画生成プロセス開始 (Audit Plan Generation Process Start) ---")
        # ワークフローを実行 (Invoke the workflow)
        # stream() を使うと中間状態も取得できるが、ここでは最終結果のみ取得
        # 非同期実行には ainvoke を使用
        try:
             final_state = await self.app.ainvoke(initial_state) # invoke -> ainvoke
             logging.info("\n--- プロセス終了 (Process End) ---")
             logging.info(f"Final State: {final_state}")
        except Exception as e:
             logging.error(f"エラー: LangGraphワークフローの実行中にエラーが発生しました: {e}")
             # エラー時の最終状態（部分的な可能性あり）またはNoneを設定
             final_state = {"error": str(e), "plan": [{"error": f"Workflow execution failed: {e}"}]} # エラー情報を追加

        # ★★★ ここで final_state を確認 ★★★
        logging.info("\n--- ワークフロー完了後の最終状態 (Final State after Workflow Completion) ---")
        logging.info(f"{final_state}")
        # ★★★ 確認ポイント終了 ★★★
        raise Exception("★★★ 意図的に処理を停止 ★★★") # ここで例外を投げて停止
 
        logging.info("\n--- 最終的な監査計画 (Converting Final State to Execution Plan) ---")
        # 最終状態からExecutionPlanに変換
        execution_plan = self._convert_workflow_result_to_execution_plan(
            final_state,
            parameters,
            scenario_text # scenario_textを渡す
        )

        # 最終的な修正回数を表示 (デバッグ用)
        final_revision_count = final_state.get('revision_count', 'N/A') if final_state else 'N/A'
        logging.info(f"最終修正回数 (Final Revision Count): {final_revision_count}")
        logging.info(f"生成されたExecutionPlan: {execution_plan.json(indent=2,ensure_ascii=False)}")

        return execution_plan

# --- 新ノード: exploration_executor ---
import asyncio
async def exploration_executor(state: 'AuditPlanState', registry=None) -> 'AuditPlanState':
    """
    初期データ収集ノード（本実装）。
    execute_step関数を流用し、data_agentから初期データを収集し、exploration_dataに格納。
    phaseを'hypothesis_generation'に進める。
    """
    logging.info("\n--- Exploration Executor Node ---")
    plan = state.get('plan', [])
    exploration_results = {}
    # データ探索用step（例: skill_idがanalyze_order等）を抽出
    exploration_steps = [step for step in plan if step.get('skill_id') == 'analyze_order']
    for step in exploration_steps:
        try:
            plan_step = PlanStep(
                id=step["id"],
                description=step["description"],
                skill_id=step["skill_id"],
                input_data={"input": step["query"]},
                parameters={},
                is_completed=False,
                output_data=None,
                error=None,
                selected_agent=None,
                start_time=None,
                transitions={}
            )
            logging.info(f"[探索実行] skill_id={plan_step.skill_id} query={plan_step.input_data['input']}")
            result_step = await execute_step(plan_step)
            logging.info(f"[探索レスポンス詳細] step_id={plan_step.id} output_data={result_step.output_data} error={result_step.error}")
            exploration_results[plan_step.id] = {
                "description": plan_step.description,
                "output_data": result_step.output_data,
                "error": result_step.error
            }
        except Exception as e:
            logging.error(f"[探索エラー] step_id={step['id']} error={e}")
            exploration_results[step['id']] = {
                "description": step["description"],
                "error": str(e)
            }
    return {
        **state,
        "exploration_data": exploration_results,
        "phase": "hypothesis_generation"
    }

# --- 新ノード: hypothesis_generator ---
def hypothesis_generator(state: AuditPlanState) -> AuditPlanState:
    """
    仮説生成ノード（ダミー実装）。
    exploration_dataとobjectiveをもとに、仮説リスト（ダミー）をhypothesesに格納。
    phaseを'hypothesis_evaluation'に進める。
    """
    logging.info("\n--- Hypothesis Generator Node ---")
    logging.info(f"[仮説生成] exploration_data: {state.get('exploration_data')}")
    logging.info(f"[仮説生成] objective: {state.get('objective')}")
    # TODO: 実際のLLMによる仮説生成ロジックを後で実装
    dummy_hypotheses = [
        {"hypothesis": "リスクAの兆候があるかもしれない", "priority": 1},
        {"hypothesis": "データ不足の可能性", "priority": 2}
    ]
    logging.info(f"[仮説生成] 生成された仮説: {dummy_hypotheses}")
    # 仮説の妥当性チェック用ログ
    logging.info("[仮説妥当性チェック] 以下の情報で仮説の妥当性を確認してください：")
    logging.info(f"  - exploration_data: {state.get('exploration_data')}")
    logging.info(f"  - objective: {state.get('objective')}")
    logging.info(f"  - hypotheses: {dummy_hypotheses}")
    logging.info("[仮説妥当性チェック] ※将来的にLLM判定を組み込む想定。現状はダミーログのみ。")
    return {
        **state,
        "hypotheses": dummy_hypotheses,
        "phase": "hypothesis_evaluation"
    }

# --- 新ノード: hypothesis_evaluator ---
def hypothesis_evaluator(state: AuditPlanState) -> AuditPlanState:
    """
    仮説評価ノード（ダミー実装）。
    hypothesesが空またはリスク兆候なしならphaseを'final_analysis'に、仮説があれば'refinement_planning'に進める。
    """
    logging.info("\n--- Hypothesis Evaluator Node ---")
    hypotheses = state.get("hypotheses", [])
    logging.info(f"[仮説評価] hypotheses: {hypotheses}")
    # ダミーロジック: 仮説が1つ以上かつ"リスク"を含む場合のみ詳細計画へ
    if hypotheses and any("リスク" in h.get("hypothesis", "") for h in hypotheses):
        next_phase = "refinement_planning"
        logging.info("仮説あり: refinement_planningへ")
    else:
        next_phase = "final_analysis"
        logging.info("仮説なしまたはリスク兆候なし: final_analysisへ")
    return {
        **state,
        "phase": next_phase
    }

