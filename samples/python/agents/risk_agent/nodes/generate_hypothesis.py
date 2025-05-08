import logging
import uuid
from typing import Any, Dict, List
import asyncio

from ..core.node_base import Node, NodeResult, make_history_entry
from ..prompts import get_generate_hypothesis_prompt, get_supporting_hypothesis_prompt

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from ..agent import execute_step, PlanStep
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Hypothesis(BaseModel):
    id: str = Field(description="仮説のID")
    text: str = Field(description="仮説の内容")
    priority: float = Field(description="仮説の優先度")
    status: str = Field(description="仮説のステータス")
    time_window: str = Field(description="仮説の時間枠")
    supporting_evidence_keys: List[str] = Field(description="仮説の支持根拠のキー")
    next_validation_step_suggestion: str = Field(description="仮説の次の検証ステップの提案")
    metric_definition: str = Field(description="仮説のメトリック定義")

class HypothesisList(BaseModel):
    """
    生成した仮説リスト
    """
    hypotheses: List[Hypothesis] = Field(
        ..., description="生成した仮説のリスト"
    )
  

# ----------------------------------------------------------------------------
# get_reference_info の並列重複実行を防ぐためのキャッシュとロック
# ----------------------------------------------------------------------------

# Key: (raw_query, risk_scenario) に対するツール結果を保持
_REFERENCE_CACHE: Dict[tuple[str, str], str] = {}

# 実行中ステータスを共有するロック。複数同時呼び出し時は待機させる
_REFERENCE_LOCK: "asyncio.Lock | None" = None  # 遅延初期化

# ----------------------------------------------------------------------------

async def get_reference_info(raw_query: str, risk_scenario: str) -> str:
    """不正を検知するための兆候のリストを生成するツール
    ※仮説生成の際は必ず使用すること
    
    Args:
        raw_query: 検知したい不正の内容
        risk_scenario: リスクシナリオ
    Returns:
        str: 過去の不正事例を参考に生成されたリスクシナリオ
    """

    global _REFERENCE_LOCK  # noqa: PLW0603
    if _REFERENCE_LOCK is None:
        _REFERENCE_LOCK = asyncio.Lock()

    cache_key = (raw_query, risk_scenario)

    # 既に取得済みであればすぐ返す
    if cache_key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[cache_key]

    # 未取得の場合はロックを取得して実行（N+1 重複を防止）
    async with _REFERENCE_LOCK:
        # 先に待機している間に他コルーチンが取得した可能性を再確認
        if cache_key in _REFERENCE_CACHE:
            return _REFERENCE_CACHE[cache_key]

        query = (
            "RAGを使って、以下の不正を検知するためのリスクシナリオに関する兆候を複数生成してください。具体的な事例にフォーカスした兆候ではなく、汎用的な兆候を生成してください。\n\n不正:\n"
            f"{raw_query} \n\nリスクシナリオ:\n{risk_scenario} \n\n 出力形式: [<兆候1>(ref:<参考事例>), 兆候2(ref:<参考事例>), 兆候3(ref:<参考事例>),...] "
        )

        # A2Aエージェントのうち、「analyze_fraud_case」スキルを呼び出す
        step = PlanStep(
            id="1",
            description=f"データエージェントへのクエリ: {query}",
            skill_id="analyze_fraud_case",
            input_data={"input": query},
        )

        step_result = await execute_step(step)

        result_text = step_result.output_data.get("text", "") if step_result.output_data else ""

        # キャッシュに保存
        _REFERENCE_CACHE[cache_key] = result_text

        # ログ出力
        logger.info("--------------------------------")
        logger.info(f"tool_result: {result_text}")
        logger.info("--------------------------------")

        return result_text

class GenerateHypothesisNode(Node):
    """仮説生成ノード

    初期仮説、または支持された仮説に基づく追加仮説を生成する。
    LLM が使えない場合はルールベースでダミー仮説を作る。
    """

    id = "generate_hypothesis"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.info("--- Node: GenerateHypothesis ---")

        objective: str = state.get("objective", "No objective provided")
        existing_hypotheses: List[Dict] = state.get("current_hypotheses", [])
        # decision_makerから渡されるパラメータを取得
        params = state.get("next_action", {}).get("parameters", {})
        parent_hypothesis_id = params.get("parent_hypothesis_id")

        newly_generated_hypotheses: List[Dict[str, Any]] = []
        prompt = ""
        parent_hypothesis = None

        if parent_hypothesis_id:
            # --- 追加仮説生成モード ---
            logger.info(f"Generating supporting hypotheses for parent: {parent_hypothesis_id}")
            parent_hypothesis = next((h for h in existing_hypotheses if h.get('id') == parent_hypothesis_id), None)
            if not parent_hypothesis:
                logger.error(f"Parent hypothesis {parent_hypothesis_id} not found in state.")
                # エラー処理またはフォールバックが必要
                # ここでは空リストを返すか、エラーを示すNodeResultを返す
                return NodeResult(
                    observation="Parent hypothesis not found",
                    patch={"next_action": None, "error_message": f"Parent hypothesis {parent_hypothesis_id} not found"},
                    events=[{"type": "error", "message": "Parent hypothesis not found"}]
                )
            prompt = get_supporting_hypothesis_prompt(state, parent_hypothesis)
            # logger.info(f"supporting prompt:{prompt}")
            generation_mode = "supporting"
        else:
            # --- 初期仮説生成モード ---
            logger.info("Generating initial hypotheses.")
            prompt = get_generate_hypothesis_prompt(state)
            generation_mode = "initial"

        if prompt:
            try:
                react_llm = ChatOpenAI(
                    model="o3-mini"
                )
                react_agent = create_react_agent(
                    model=react_llm, 
                    tools=[get_reference_info], 
                    response_format=HypothesisList, 
                )
                # agent_executor を使用して呼び出す
                resp_data = await react_agent.ainvoke({"messages": [("human", prompt)]})

                raw_hypotheses = dict(resp_data["structured_response"]).get("hypotheses", [])
                # IDの付与と parent_hypothesis_id の設定
                for i, hypo in enumerate(raw_hypotheses):
                    hypo = dict(hypo)

                    if parent_hypothesis_id:
                        # 親仮説がある場合は sub+親仮説ID+(i+1) 形式
                        hypo["id"] = f"sub_{parent_hypothesis_id}_{i+1}"
                    if not hypo.get("id"):
                        prefix = "hyp"
                        hypo["id"] = f"{prefix}_{str(uuid.uuid4())[:8]}_{i}"
                    if parent_hypothesis_id:
                        hypo["parent_hypothesis_id"] = parent_hypothesis_id
                    else:
                         # 初期生成時は None またはキー自体を削除
                         hypo.pop("parent_hypothesis_id", None)
                    # status がない場合や不正な場合は 'new' に設定
                    if hypo.get("status") != "new":
                         hypo["status"] = "new"
                    newly_generated_hypotheses.append(hypo)

            except Exception as e:  # noqa: BLE001
                logger.warning(f"LLM failed during {generation_mode} hypothesis generation, fallback: {e}")
                # フォールバックはモードに応じて調整可能

        if not newly_generated_hypotheses:
            # フォールバック: モードに応じたダミー仮説生成
            fallback_text = f"{objective} に関する初期仮説" # デフォルト
            fallback_prefix = "fb_hyp"
            fallback_parent_id = None
            if parent_hypothesis:
                fallback_text = f"{parent_hypothesis['text']} を深掘りする追加仮説"
                fallback_prefix = "fb_sup"
                fallback_parent_id = parent_hypothesis_id

            fallback_hypo = {
                "id": f"{fallback_prefix}_{str(uuid.uuid4())[:8]}",
                "text": fallback_text,
                "priority": 0.5, # デフォルト優先度
                "status": "new",
                "parent_hypothesis_id": fallback_parent_id,
                "supporting_evidence_keys": [],
                "next_validation_step_suggestion": "",
                "metric_definition": ""
            }
            newly_generated_hypotheses.append(fallback_hypo)

        # 既存の仮説リストと新しく生成された仮説を結合
        updated_hypotheses = existing_hypotheses + newly_generated_hypotheses

        patch = {
            # 既存リストに新しい仮説を追加する
            "current_hypotheses": updated_hypotheses,
            "next_action": None, # このノードはアクションを決定しない
            # cycles_since_last_hypothesis のリセットなどは decision_maker 側で行う想定
        }

        # console 出力
        logger.info(f"Generated {len(newly_generated_hypotheses)} new hypotheses ({generation_mode} mode). Total hypotheses: {len(updated_hypotheses)}")
        if newly_generated_hypotheses:
             logger.info("Newly generated: %s", newly_generated_hypotheses)

        events = [make_history_entry(
            "node",
            {
                "name": "generate_hypothesis",
                "mode": generation_mode,
                "parent_id": parent_hypothesis_id,
                "count": len(newly_generated_hypotheses),
                "generated_hypotheses": [hypo["id"] for hypo in newly_generated_hypotheses],
                "process":[step.content for step in resp_data["messages"]]
            },
            state
        )]

        # 観察結果としては新しく生成されたもののみを返す
        return NodeResult(observation=newly_generated_hypotheses, patch=patch, events=events) 