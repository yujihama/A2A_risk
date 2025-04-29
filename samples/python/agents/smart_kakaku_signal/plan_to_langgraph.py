from langgraph.graph import StateGraph, START, END
from typing import Any, Dict
from samples.python.agents.smart_kakaku_signal.agent import ExecutionPlan, PlanStep

# 追加: Message, DataPartのimport
from A2A_risk.samples.python.common.types import Message, DataPart
import asyncio
import json
import time
import logging # Add import for logging
import pprint, traceback

# PlanStepをLangGraphノードに変換する関数
def make_step_node(step: PlanStep, agent_executor, llm_client=None, default_next=None):
    async def node_fn(state: dict) -> dict:
        print(f"[DEBUG] make_step_node: step={step}, llm_client={llm_client}")
        print(f"[START] ノード: {step.id}")
        print(f"  スキルID: {step.skill_id}")
        
        try:
            if step.skill_id == "analyze":
                if llm_client is None:
                    print("[ERROR] llm_client is None in make_step_node!")
                    raise ValueError("llm_clientが設定されていません")
                prev_results = ""
                for k, v in state.items():
                    prev_results += f"ステップ {k}: {v}\n"
                instruction = step.input_data.get("input") if isinstance(step.input_data, dict) else str(step.input_data)
                prompt = f"""
あなたは分析AIです。以下はこれまでのステップの結果です:
{prev_results}

次の指示に従って分析してください:
{instruction}
"""
                response = await llm_client.ainvoke([
                    {"role": "system", "content": "あなたは優秀な分析AIです。"},
                    {"role": "user", "content": prompt}
                ])
                result = response.content if hasattr(response, "content") else str(response)
                print("analyzeノードの分析結果:", result)
                # analyzeノードは従来通りOK固定
                condition = "OK"
            else:
                # --- input_dataとoutput_dataを結合して単一のDataPartを作成 ---
                input_str = str(step.input_data.get("input"))
                output_str = str(step.output_data)

                # エージェントに渡す情報を結合 (キーは既存のログに合わせて 'input' とする)
                combined_input = f"指示: {input_str}\n期待される出力: {output_str}"
                input_part_data = {"input": combined_input}
                print(f"input_part_data: {input_part_data}")

                input_message = Message(
                    role="user",
                    parts=[DataPart(data=input_part_data)] 
                )
                response = await agent_executor.find_and_send_task(step.skill_id, input_message)
                # レスポンス内容を詳細にログ出力
                logging.info(f"Agent initial response for step {step.id}: {response}")
                pprint.pprint(response)
                
                task_id = getattr(response.result, "id", None) if response and hasattr(response, "result") else None
                if not task_id:
                    raise ValueError("エージェントからタスクIDを取得できませんでした")
                print(f"  タスクが送信されました。タスクID: {task_id}")
                max_retries = 10
                poll_interval = 2
                result_text = None
                for attempt in range(max_retries):
                    print(f"  結果をポーリング中... 試行 {attempt + 1}/{max_retries}")
                    task_status = await agent_executor.get_task(task_id)
                    status_obj = getattr(task_status.result, "status", None) if task_status and hasattr(task_status, "result") else None
                    state_val = getattr(status_obj, "state", None) if status_obj else None
                    if state_val == "completed":
                        message = getattr(status_obj, "message", None)
                        if message and getattr(message, "parts", None):
                            part = message.parts[0]
                            if getattr(part, "type", None) == "text":
                                result_text = getattr(part, "text", None)
                                print("  タスク結果を取得しました")
                                break
                    elif state_val in ["failed", "canceled"]:
                        raise ValueError("タスクが失敗またはキャンセルされました")
                    await asyncio.sleep(poll_interval)
                print(f"[END]   ノード: {step.id}")
                print(f"  スキルID: {step.skill_id}")
                print(f"  返ってきた結果: {result_text}")
                result = result_text
                # --- ここから条件付きエッジ判定 ---
                condition = "OK"
                if llm_client is not None and step.output_data:
                    print(f"[DEBUG] step: {step}")
                    # 期待値チェック用プロンプト
                    prompt = f"""
以下の確認を行うため担当者へ確認を行いました。
{step.description}

 - 担当者への問合せ内容：
{step.input_data}
 - 問合せによって期待される情報：
{step.output_data}

以下は担当者から実際に回答された内容です。期待される情報を満たしているかを判定してください。
具体的に、担当者の回答をもとに次のタスクを実行できるかどうかを論理的に考えて判定してください。
 - 担当者の回答:
{result_text}

以下のJSON形式で回答してください。
判定結果：OK/NG/ERRORのいずれかで答えてください。
理由：簡潔に述べてください。
再度問合せする内容：期待する結果を得るために再度問合せする場合は、問合せ内容を記載してください。
"""
                    print(f"[DEBUG] prompt: {prompt}")
                    llm_response = await llm_client.ainvoke([
                        {"role": "system", "content": "あなたは内部監査の支援を行うAIアシスタントです。"},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_object"
                    }
                    )
                    raw_content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                    content = json.loads(raw_content)
                    print(f"[LLM判定] 結果: {content['判定結果']}")
                    print(f"[LLM判定] 理由: {content['理由']}")
                    print(f"[LLM判定] 再度問合せする内容: {content['再度問合せする内容']}")

                    # OK/NG/ERRORのみを扱う。YES/NOもOK/NGにマッピング
                    if content["判定結果"] == "OK":
                        condition = "OK"
                    elif content["判定結果"] == "NG":
                        condition = "NG"
                    else:
                        condition = "ERROR"
                # --- ここまで条件付きエッジ判定 ---
        except Exception as e:
            print(f"  スキル実行エラー: {e}")
            import traceback
            traceback.print_exc()
            result = None
            condition = "ERROR"
        state = dict(state)
        state[step.id] = result
        state[f"{step.id}_condition"] = condition
        print(f"  判定結果: {condition}")
        return state
    return node_fn

# 条件分岐を行う関数を定義
def route_based_on_condition(state, plan_steps):
    # 最後に実行されたステップのIDと条件を取得
    last_step_id = None
    last_step_number = -1
    for key in state:
        if key.startswith("step") and "_condition" not in key:
            try:
                step_number = int(key[4:])
                if step_number > last_step_number:
                    last_step_number = step_number
                    last_step_id = key
            except ValueError:
                continue # "step"で始まるが数字でないキーは無視

    if last_step_id is None:
        print("[ROUTE_DEBUG] Last step ID not found in state.")
        return None # 最初のステップ、またはステップが見つからない場合

    condition = state.get(f"{last_step_id}_condition")
    print(f"[ROUTE_DEBUG] Routing from step: {last_step_id}, condition: {condition}")

    if condition is None:
        print(f"[ROUTE_DEBUG] Condition not found for step: {last_step_id}")
        return "analyze_summary" # 条件がない場合はエラー扱い

    # 対応するPlanStepオブジェクトを検索
    current_step_obj = None
    for step in plan_steps:
        if step.id == last_step_id:
            current_step_obj = step
            break

    if current_step_obj is None:
        print(f"[ROUTE_DEBUG] PlanStep object not found for ID: {last_step_id}")
        return "analyze_summary" # PlanStepが見つからない場合もエラー扱い

    transitions = getattr(current_step_obj, "transitions", {}) or {}
    next_node = transitions.get(condition)

    print(f"[ROUTE_DEBUG] Transitions for {last_step_id}: {transitions}")
    print(f"[ROUTE_DEBUG] Calculated next_node based on condition '{condition}': {next_node}")

    if next_node:
        return next_node
    elif condition in ["NG", "ERROR"]:
        print(f"[ROUTE_DEBUG] Condition is '{condition}', routing to analyze_summary.")
        return "analyze_summary"
    else:
        # OKで遷移先がない場合は次のステップ (IDが+1) or analyze_summary
        next_step_id_num = last_step_number + 1
        next_step_id = f"step{next_step_id_num}"
        found_next = any(s.id == next_step_id for s in plan_steps)
        if found_next:
             print(f"[ROUTE_DEBUG] Condition is '{condition}' with no explicit transition. Routing to next sequential step: {next_step_id}")
             return next_step_id
        else:
             print(f"[ROUTE_DEBUG] Condition is '{condition}' with no explicit transition and no next sequential step. Routing to analyze_summary.")
             return "analyze_summary"

# 全体分析ノード
def make_summary_node(scenario_text, scenario_analysis, parameters, data_analyzer):
    async def summary_node_fn(state: dict) -> dict:
        print(f"[DEBUG] make_summary_node: data_analyzer={data_analyzer}")
        print("[START] 全体分析ノード (analyze_summary)")
        try:
            if data_analyzer is None:
                print("[ERROR] data_analyzer is None in make_summary_node!")
                raise ValueError("data_analyzerが設定されていません")
            # step_resultsを整形（step_id→{'description':..., 'result':...}のdictに変換）
            step_results = {}
            for k, v in state.items():
                if k.startswith("step"):
                    step_results[k] = {"description": f"ステップ {k}", "result": v}
            result = await data_analyzer.analyze_collected_data(
                scenario_text,
                scenario_analysis,
                step_results,
                parameters
            )
            print("全体分析ノードの結果:", result)
        except Exception as e:
            print(f"全体分析ノードでエラー: {e}")
            result = {"error": str(e)}
        state = dict(state)
        state["summary"] = result
        return state
    return summary_node_fn

def make_dynamic_node(node_name):
    async def dynamic_node_fn(state: dict) -> dict:
        print(f"[DYNAMIC NODE] {node_name} に遷移しました。state: {state}")
        # ここで必要に応じてユーザー問い合わせやエラー通知などの処理を実装可能
        # 今は何もせずanalyze_summaryに遷移
        return {"__next__": "analyze_summary", **state}
    return dynamic_node_fn

# ExecutionPlanからLangGraphグラフを構築
def build_langgraph_from_plan(plan: ExecutionPlan, agent_executor, llm_client=None, scenario_text=None, scenario_analysis=None, parameters=None, data_analyzer=None):
    graph = StateGraph(state_schema=dict)
    node_names = set()
    # まず全PlanStepノードを追加
    for step in plan.steps:
        node_name = step.id
        node_fn = make_step_node(step, agent_executor, llm_client=llm_client)
        graph.add_node(node_name, node_fn)
        node_names.add(node_name)
    # 全体分析ノードを追加
    summary_node_name = "analyze_summary"
    summary_node_fn = make_summary_node(scenario_text, scenario_analysis, parameters, data_analyzer)
    graph.add_node(summary_node_name, summary_node_fn)
    node_names.add(summary_node_name)
    # START→最初のノードのみ
    if plan.steps:
        graph.add_edge(START, plan.steps[0].id)

    # 条件付きエッジを追加
    # route_based_on_condition関数に必要な情報を渡すため、lambdaを使用
    conditional_router = lambda state: route_based_on_condition(state, plan.steps)

    all_possible_destinations = set(node_names) | {END} # ENDも遷移先として追加

    for step in plan.steps:
        # 各ステップから条件付きエッジを設定
        # 遷移先候補を動的に取得 + analyze_summary を常に含める
        possible_targets = set(getattr(step, 'transitions', {}).values()) | {summary_node_name}
        # 次のシーケンシャルステップも候補に追加 (存在すれば)
        current_step_num = int(step.id[4:])
        next_seq_step_id = f"step{current_step_num + 1}"
        if next_seq_step_id in node_names:
            possible_targets.add(next_seq_step_id)

        # マッピングを作成
        target_map = {target: target for target in possible_targets if target in node_names}
        # 候補にない遷移先(ENDなど)への遷移はここでは定義しない
        # (analyze_summaryからのENDは別途定義)

        print(f"[BUILD_DEBUG] Adding conditional edge from {step.id} with targets: {target_map}")
        graph.add_conditional_edges(
            step.id,
            conditional_router,
            target_map # 計算された遷移先マップを使用
        )

    # analyze_summary→END (これは変更なし)
    graph.add_edge(summary_node_name, END)

    # デバッグ: グラフ構造を出力
    print("[DEBUG] LangGraphノード一覧:")
    for node in graph.nodes:
        print(f"  ノード: {node}")
    # エッジ情報は LangGraph 内部で管理されるため、ここでの表示は不完全になる可能性あり
    # print("[DEBUG] LangGraphエッジ一覧 (add_edgeで明示的に追加されたもの):")
    # for src, dst in graph.edges:
    #    print(f"  {src} → {dst}")
    # print("[DEBUG] 条件付きエッジが設定されました。")

    return graph.compile()

# サンプルのagent_executor（ダミー）
class DummyAgentExecutor:
    def execute_skill(self, skill_id, input_data):
        return {"skill_id": skill_id, "input_data": input_data, "result": "dummy_result"}

# サンプルmain
if __name__ == "__main__":
    # ダミーPlanStep/ExecutionPlanを作成
    step1 = PlanStep(id="step1", skill_id="query_order", description="desc1", input_data={"a":1}, transitions={"OK": "step2"})
    step2 = PlanStep(id="step2", skill_id="analyze", description="desc2", input_data={"b":2}, transitions={"OK": "analyze_summary"})
    plan = ExecutionPlan(plan_id="p1", product_id="pid", threshold=1.0, steps=[step1, step2])
    agent_executor = DummyAgentExecutor()
    # llm_client, data_analyzerを有効なインスタンスで渡す
    from langchain_openai import ChatOpenAI
    class SimpleDataAnalyzer:
        async def analyze_collected_data(self, scenario_text, scenario_analysis, step_results, parameters):
            return {"is_anomaly": False, "is_data_sufficient": True, "analysis": "ダミー分析結果"}
    llm_client = ChatOpenAI(model="gpt-4o", temperature=0)
    data_analyzer = SimpleDataAnalyzer()
    graph = build_langgraph_from_plan(plan, agent_executor, llm_client=llm_client, scenario_text=None, scenario_analysis=None, parameters=None, data_analyzer=data_analyzer)
    # グラフをinvoke
    result = graph.invoke({})
    # is_anomalyだけ出力
    if "summary" in result and isinstance(result["summary"], dict):
        is_anomaly = result["summary"].get("is_anomaly", None)
        if is_anomaly is None:
            print("is_anomalyが取得できませんでした")
        elif is_anomaly is True:
            print("異常検出: あり")
        elif is_anomaly is False:
            print("異常検出: なし")
        else:
            print("異常検出: 判断保留")
    else:
        print("summaryが取得できませんでした") 