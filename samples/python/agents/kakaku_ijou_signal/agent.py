import os
import time
import asyncio
from typing import TypedDict, Annotated, List, Union
from uuid import uuid4
import operator
import re

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

# A2Aレジストリとスマートクライアントをインポート
from samples.python.common.registry.agent_registry import AgentRegistry
from samples.python.common.client.smart_client import SmartA2AClient
from samples.python.common.types import Task, Message, TextPart, TaskState, Part

# LLM連携用 (評価ノードで使用)
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from langchain_openai import OpenAI

# .envファイルから環境変数を読み込む (任意)
load_dotenv()

# --- LLM設定 (評価用) ---
# TODO: プロトタイプ計画に基づき、Google Gemini等に変更可能にする
llm = ChatOpenAI(model="gpt-4o", temperature=0) # modelは適宜変更

# --- エージェントレジストリとスマートクライアントの設定 ---
registry = AgentRegistry()
smart_a2a_client = None  # 初期化は非同期で行うため、別途初期化関数を用意

async def initialize_registry():
    """エージェントレジストリを初期化する"""
    global smart_a2a_client
    
    print("エージェントレジストリを初期化中...")
    
    # エージェントを発見して登録
    await registry.discover_agents([
        "http://localhost:8001",  # PurchasingDataAgent
        "http://localhost:8002"   # InventoryDataAgent
    ])
    
    # 登録されたエージェントを確認
    agents = registry.list_agents()
    print(f"登録されたエージェント: {', '.join([agent.name for agent in agents])}")
    
    # LLMベースのスキル選択器を作成
    from samples.python.common.registry.skill_selector import SkillSelector
    skill_selector = SkillSelector(model_name="gpt-3.5-turbo", temperature=0)
    print("LLMベースのスキル選択器を初期化しました。")
    
    # スマートクライアントを初期化
    smart_a2a_client = SmartA2AClient(registry, skill_selector=skill_selector)
    print("スマートA2Aクライアントを初期化しました。")

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """LangGraphのState定義"""
    task_id: str | None # 発行されたタスクID
    product_id_to_query: str # 問い合わせる商品ID (初期入力)
    retrieved_data: str | None # エージェントから取得したデータ
    is_price_anomaly: bool | None # 価格異常判定結果
    anomaly_reason: str | None # 異常判定の理由
    error_message: str | None # 処理中に発生したエラー
    last_agent_url: str | None # 最後にアクセスしたエージェントのURL（ポーリング用）

# --- LangGraph Nodes ---
async def request_data(state: AgentState) -> AgentState:
    """適切なエージェントを選択してデータ取得タスクを依頼するノード"""
    print("--- Node: request_data ---")
    product_id = state['product_id_to_query']
    print(f"製品ID {product_id} の情報をリクエスト中...")
    state['error_message'] = None # エラーをリセット
    
    try:
        # スマートクライアントが初期化されていることを確認
        if not smart_a2a_client:
            raise ValueError("スマートA2Aクライアントが初期化されていません。")
        
        # メッセージを作成
        input_message = Message(
            role="user", 
            parts=[TextPart(text=f"製品ID {product_id} に関する情報を検索してください。")]
        )
        
        # LLMベースのスキル選択のみを使用
        message_text = f"製品ID {product_id} に関する情報を検索してください。"
        skill_result = await smart_a2a_client.skill_selector.select_skill(message_text)
        
        # 選択されたスキルを使用
        skill_id = skill_result.skill_id
        print(f"LLMが選択したスキル: {skill_id} (確信度: {skill_result.confidence})")
        print(f"選択理由: {skill_result.reasoning}")
        
        # 選択されたスキルでタスクを送信
        task_response = await smart_a2a_client.find_and_send_task(
            skill_id=skill_id,
            message=input_message
        )
        
        # レスポンス情報をデバッグ出力
        print(f"タスクレスポンスの種類: {type(task_response)}")
        
        # result フィールドに Task オブジェクトが格納されている
        if task_response.result and task_response.result.id:
            server_task_id = task_response.result.id
            print(f"タスクが正常に送信されました。サーバータスクID: {server_task_id}")
            state['task_id'] = server_task_id
            state['retrieved_data'] = None # 取得データをリセット
            
            # URLからエンドポイント(/a2a)を除去してベースURLを保存
            current_url = smart_a2a_client.client.url
            base_url = current_url.replace('/a2a', '') if current_url.endswith('/a2a') else current_url
            state['last_agent_url'] = base_url
        else:
            print("エラー: レスポンスからタスクIDを取得できませんでした。")
            if hasattr(task_response, 'error') and task_response.error:
                error_msg = f"サーバーエラー: {task_response.error}"
            else:
                error_msg = "レスポンスからタスクIDを取得できませんでした。"
            state['error_message'] = error_msg
    except Exception as e:
        print(f"タスク送信中にエラーが発生しました: {e}")
        state['error_message'] = f"タスク送信中にエラーが発生しました: {e}"
        state['task_id'] = None

    return state

async def wait_for_result(state: AgentState) -> AgentState:
    """エージェントのタスク結果をポーリングで待機するノード"""
    print("--- Node: wait_for_result ---")
    task_id = state['task_id']
    if not task_id:
        print("エラー: ポーリングするタスクIDがありません。")
        state['error_message'] = "ポーリングするタスクIDがありません。"
        return state

    print(f"タスクID {task_id} の結果をポーリング中...")
    max_retries = 10 # 最大試行回数
    poll_interval = 2 # ポーリング間隔 (秒)

    for attempt in range(max_retries):
        try:
            # get_task メソッドを使用して結果をポーリング
            task_response = await smart_a2a_client.get_task(
                task_id=task_id, 
                agent_url=state['last_agent_url']
            )
            
            # デバッグ情報を表示
            print(f"試行 {attempt + 1}/{max_retries}: レスポンスの種類: {type(task_response)}")
            
            # エラーチェック
            if hasattr(task_response, 'error') and task_response.error:
                print(f"レスポンスでエラー: {task_response.error}")
                if attempt == max_retries - 1:  # 最後の試行で失敗した場合
                    state['error_message'] = f"API エラー: {task_response.error}"
                    return state
                await asyncio.sleep(poll_interval)
                continue
            
            # Task オブジェクトは result フィールドに格納されている
            if not task_response.result:
                print(f"試行 {attempt + 1}/{max_retries}: レスポンスに結果がありません")
                if attempt == max_retries - 1:  # 最後の試行で失敗した場合
                    print("ポーリングがタイムアウトしました。")
                    state['error_message'] = "ポーリングがタイムアウトしました。結果データを受信できませんでした。"
                    return state
                await asyncio.sleep(poll_interval)
                continue
                
            task = task_response.result
            print(f"試行 {attempt + 1}/{max_retries}: タスク状態 = {task.status.state}")

            if task.status.state == TaskState.COMPLETED:
                print("タスクが完了しました。")
                # 結果メッセージからテキストを抽出
                result_message = task.status.message
                if result_message and result_message.parts and isinstance(result_message.parts[0], TextPart):
                    retrieved_text = result_message.parts[0].text
                    print(f"取得したデータ: {retrieved_text}")
                    state['retrieved_data'] = retrieved_text
                    state['error_message'] = None
                    return state
                else:
                    print("警告: 完了したタスクに有効なテキスト結果がありません。")
                    state['retrieved_data'] = "完了したタスクに有効なテキスト結果がありません。"
                    return state # データなしでも次に進む

            elif task.status.state in [TaskState.FAILED, TaskState.CANCELED]:
                error_text = "タスクが失敗したか、キャンセルされました。"
                if task.status.message and task.status.message.parts and isinstance(task.status.message.parts[0], TextPart):
                    error_text = task.status.message.parts[0].text
                print(f"エラー: {error_text}")
                state['error_message'] = error_text
                state['retrieved_data'] = None
                return state

            # 処理中の場合は待機してリトライ
            await asyncio.sleep(poll_interval)

        except Exception as e:
            print(f"タスク結果のポーリング中にエラーが発生しました: {e}")
            state['error_message'] = f"タスク結果のポーリング中にエラーが発生しました: {e}"
            state['retrieved_data'] = None
            return state

    print("ポーリングがタイムアウトしました。")
    state['error_message'] = "タスク結果のポーリングがタイムアウトしました。"
    state['retrieved_data'] = None
    return state

# --- 評価用LLMプロンプトとデータ構造 ---
class EvaluationResult(BaseModel):
    is_anomaly: bool = Field(description="Whether the price is considered anomalous based on the retrieved data.")
    reason: str = Field(description="The reason for the anomaly judgment.")

async def evaluate_price(state: AgentState) -> AgentState:
    """取得したデータから価格が異常かどうかを判断するノード"""
    print("--- Node: evaluate_price ---")
    retrieved_data = state.get('retrieved_data')
    if not retrieved_data:
        print("エラー: 評価するデータがありません。")
        state['error_message'] = "評価するデータがありません。"
        state['is_price_anomaly'] = False
        state['anomaly_reason'] = "データ不足のため評価できません。"
        return state

    print(f"データを評価中: {retrieved_data}")

    # LLMを使用して価格が異常かどうかを判断
    llm = OpenAI()
    prompt = f"""
    以下の製品情報を分析して、その価格が市場の一般的な価格と比較して異常に高いか低いかを判断してください。
    
    製品情報: {retrieved_data}
    
    判断基準:
    1. 同種の製品の一般的な市場価格と比較して著しく高い場合、価格異常と判断します。
    2. 同種の製品の一般的な市場価格と比較して著しく低い場合、価格異常と判断します。
    3. 一般的な価格範囲内であれば、価格正常と判断します。
    
    以下の形式で回答してください：
    異常の有無: [True/False]
    理由: [あなたの詳細な分析理由]
    """

    try:
        result = llm.invoke(prompt)
        print(f"LLMレスポンス: {result}")
        
        # 応答から異常かどうかを抽出
        if "異常の有無: True" in result or "異常の有無:True" in result:
            state['is_price_anomaly'] = True
        else:
            state['is_price_anomaly'] = False
        
        # 応答から理由を抽出
        reason_match = re.search(r'理由:\s*(.+)', result, re.DOTALL)
        if reason_match:
            state['anomaly_reason'] = reason_match.group(1).strip()
        else:
            state['anomaly_reason'] = "理由は提供されませんでした。"
            
        print(f"評価結果: 異常={state['is_price_anomaly']}, 理由={state['anomaly_reason']}")
    except Exception as e:
        print(f"価格評価中にエラーが発生しました: {e}")
        state['error_message'] = f"価格評価中にエラーが発生しました: {e}"
        state['is_price_anomaly'] = False
        state['anomaly_reason'] = "評価中にエラーが発生しました。"
    
    return state

# --- LangGraph conditions ---
def should_evaluate(state: AgentState) -> str:
    """データ取得結果に基づき、評価フェーズに進むかエラー終了かを決定する"""
    print("--- Condition: should_evaluate ---")
    if state['error_message']:
        print(f"エラーのためエラー終了へルーティング: {state['error_message']}")
        return "end_error"
    elif state['retrieved_data']:
        print("評価フェーズへルーティング。")
        return "evaluate_price"
    else:
        print("データが取得されなかったためENDへルーティング。")
        return END

def check_initial_request(state: AgentState) -> str:
    """リクエスト結果に基づき、ポーリングフェーズに進むかエラー終了かを決定する"""
    print("--- Condition: check_initial_request ---")
    if state['error_message']:
        print(f"リクエストエラーのためエラー終了へルーティング: {state['error_message']}")
        return "end_error"
    elif state['task_id']:
        print("ポーリングフェーズへルーティング。")
        return "wait_for_result"
    else:
        print("タスクIDが取得できなかったためエラー終了へルーティング。")
        return "end_error"

# --- LangGraph Definition ---
workflow = StateGraph(AgentState)

# ノードの追加
workflow.add_node("request_data", request_data)
workflow.add_node("wait_for_result", wait_for_result)
workflow.add_node("evaluate_price", evaluate_price)
workflow.add_node("end_error", lambda state: print("--- Node: end_error ---") or state) # エラー終了用ノード

# エッジの追加（遷移定義）
# START ノードからのエントリーポイントを追加
workflow.add_edge(START, "request_data")
workflow.add_conditional_edges(
    "request_data",
    check_initial_request,
    {
        "wait_for_result": "wait_for_result",
        "end_error": "end_error" # タスク送信失敗時は終了
    }
)

workflow.add_conditional_edges(
    "wait_for_result",
    should_evaluate,
    {
        "evaluate_price": "evaluate_price",
        "end_error": "end_error" # ポーリング失敗/タイムアウト/エラー時は終了
    }
)
# 評価後は常に終了 (成功・失敗問わず)
workflow.add_edge("evaluate_price", END)
workflow.add_edge("end_error", END)

# グラフのコンパイル (LangGraphの追加機能を有効化)
graph = workflow.compile()

# --- エントリーポイント関数 ---
async def run_graph(product_id: str):
    """グラフを実行して結果を返す"""
    # レジストリとスマートクライアントを初期化
    print("グラフ実行前にエージェントレジストリを初期化します...")
    await initialize_registry()
    
    # 初期状態の設定
    initial_state = {
        "task_id": None,
        "product_id_to_query": product_id,
        "retrieved_data": None,
        "is_price_anomaly": None,
        "anomaly_reason": None,
        "error_message": None,
        "last_agent_url": None
    }
    
    # グラフの実行
    print(f"製品ID {product_id} でグラフを実行します...")
    result = await graph.ainvoke(initial_state)
    
    # 結果の表示
    print("\n=== 実行結果 ===")
    if result.get('error_message'):
        print(f"エラー: {result['error_message']}")
        return result
        
    print(f"製品ID: {result['product_id_to_query']}")
    print(f"取得データ: {result['retrieved_data']}")
    
    if result.get('is_price_anomaly') is not None:
        print(f"価格異常: {'あり' if result['is_price_anomaly'] else 'なし'}")
        print(f"理由: {result['anomaly_reason']}")
    
    return result

# --- テスト用コード ---
async def _test_graph():
    # サーバーが起動している必要がある
    print("\n=== LLMベースのスキル選択のテスト ===")
    
    # 製品情報のテスト
    print("\n製品情報のテスト（ID: P001）:")
    await run_graph("P001")
    
    # 在庫情報のテスト
    print("\n在庫情報のテスト（ID: I001）:")
    await run_graph("I001")
    
    # 存在しないIDのテスト
    print("\n存在しないIDのテスト（ID: X999）:")
    await run_graph("X999")

    # LLMベースの自然言語クエリのテスト
    print("\n=== LLMベースの自然言語クエリのテスト ===")
    await test_llm_query("製品P001の価格を教えてください")
    await test_llm_query("I001の在庫状況を知りたい")
    await test_llm_query("在庫が最も多い製品を教えてください")
    
    # 複数エージェント呼び出しのテスト
    print("\n=== 複数エージェント呼び出しのテスト ===")
    await test_multiple_agents()

async def test_multiple_agents():
    """
    複数のエージェントを呼び出してテストする
    """
    print("複数エージェント呼び出しテスト実行中...")
    
    # エージェントレジストリが初期化されていることを確認
    if not smart_a2a_client:
        print("初期化中...")
        await initialize_registry()
    
    # テストメッセージ
    test_message = Message(
        role="user", 
        parts=[TextPart(text="製品と在庫の両方の情報が必要です")]
    )
    
    try:
        # product_info スキルを持つすべてのエージェントにタスクを送信
        print("製品情報スキルを持つすべてのエージェントにタスクを送信中...")
        product_responses = await smart_a2a_client.find_and_send_tasks_to_all(
            skill_id="product_info",
            message=test_message
        )
        
        print(f"製品情報エージェントからの応答: {len(product_responses)}件")
        for i, resp in enumerate(product_responses, 1):
            if "error" in resp:
                print(f"  {i}. エージェント '{resp['agent'].name}' からエラー: {resp['error']}")
            else:
                print(f"  {i}. エージェント '{resp['agent'].name}' からの応答: {resp['response']}")
        
        # inventory_info スキルを持つすべてのエージェントにタスクを送信
        print("\n在庫情報スキルを持つすべてのエージェントにタスクを送信中...")
        inventory_responses = await smart_a2a_client.find_and_send_tasks_to_all(
            skill_id="inventory_info",
            message=test_message
        )
        
        print(f"在庫情報エージェントからの応答: {len(inventory_responses)}件")
        for i, resp in enumerate(inventory_responses, 1):
            if "error" in resp:
                print(f"  {i}. エージェント '{resp['agent'].name}' からエラー: {resp['error']}")
            else:
                print(f"  {i}. エージェント '{resp['agent'].name}' からの応答: {resp['response']}")
                
    except Exception as e:
        print(f"複数エージェントテスト中にエラーが発生しました: {e}")

async def test_llm_query(natural_query: str):
    """
    自然言語クエリを使ってLLMベースのスキル選択をテストする
    
    Args:
        natural_query: 自然言語のクエリテキスト
    """
    print(f"\n自然言語クエリのテスト: '{natural_query}'")
    
    # エージェントレジストリが初期化されていることを確認
    if not smart_a2a_client:
        print("初期化中...")
        await initialize_registry()
    
    # テキストからLLMでスキルを選択
    try:
        skill_result = await smart_a2a_client.skill_selector.select_skill(natural_query)
        
        # 選択結果を表示
        print(f"【LLM分析結果】")
        print(f"選択されたスキル: {skill_result.skill_id}")
        print(f"確信度: {skill_result.confidence}")
        print(f"選択理由: {skill_result.reasoning}")
        
        # 選択されたスキルでどのエージェントが利用可能かを確認
        agents = await registry.find_agents_by_skill(skill_result.skill_id)
        if agents:
            agent_names = [agent.name for agent in agents]
            print(f"利用可能なエージェント: {', '.join(agent_names)}")
        else:
            print(f"スキル '{skill_result.skill_id}' を持つエージェントは見つかりませんでした。")
            
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    # 直接実行された場合はテストを実行
    print("テストグラフを実行中...")
    asyncio.run(_test_graph())
