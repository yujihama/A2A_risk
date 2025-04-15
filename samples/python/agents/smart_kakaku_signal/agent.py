import os
import time
import asyncio
import re
import json
import logging
from typing import TypedDict, Annotated, List, Union, Dict, Any, Optional, Tuple
from uuid import uuid4
from pprint import pprint
from datetime import datetime
import yaml

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# A2Aレジストリとスマートクライアントをインポート
from samples.python.common.registry.agent_registry import AgentRegistry
from samples.python.common.client.smart_client import SmartA2AClient
from samples.python.common.registry.skill_selector import SkillSelector
from samples.python.common.types import Task, Message, TextPart, TaskState, Part, AgentCard, DataPart

# エージェント選択機能をインポート
from .agent_selector import AgentSelector

# .envファイルから環境変数を読み込む
load_dotenv()

# --- グローバル変数 ---
registry = AgentRegistry()
smart_a2a_client = None  # 初期化は非同期で行うため、別途初期化関数を用意
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # 計画立案用LLM
agent_selector = None  # エージェント選択器

# ロガーを設定
logger = logging.getLogger(__name__)

# --- データモデル ---
class PlanStep(BaseModel):
    """プラン内の単一ステップを表すモデル"""
    id: str = Field(description="ステップの一意なID (例: 'step1')")
    description: str = Field(description="ステップの説明")
    skill_id: str = Field(description="使用するスキルID (例: 'search_product_data')")
    input_data: Dict[str, Any] = Field(description="ステップへの入力データ")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="ステップのパラメータ")
    is_completed: bool = Field(default=False, description="ステップが完了したかどうか")
    output_data: Optional[Union[Dict[str, Any], List[str]]] = Field(default=None, description="ステップの出力データ（実行結果辞書またはLLM生成の期待値リスト）")
    error: Optional[str] = Field(default=None, description="エラーが発生した場合のメッセージ")
    selected_agent: Optional[Dict[str, Any]] = Field(default=None, description="選択されたエージェントの情報")
    start_time: Optional[datetime] = Field(default=None, description="ステップの開始時刻")

class ExecutionPlan(BaseModel):
    """実行計画全体を表すモデル"""
    plan_id: str = Field(description="計画の一意なID")
    product_id: str = Field(description="対象製品ID")
    threshold: float = Field(description="異常判定の閾値（％）")
    steps: List[PlanStep] = Field(description="実行ステップのリスト")
    current_step_index: int = Field(default=0, description="現在実行中のステップのインデックス")
    is_completed: bool = Field(default=False, description="計画全体が完了したかどうか")
    is_executed: bool = Field(default=False, description="計画が実行されたかどうか")
    is_anomaly_detected: Optional[bool] = Field(default=None, description="異常が検出されたかどうか")
    anomaly_details: Optional[str] = Field(default=None, description="検出された異常の詳細")
    created_at: float = Field(default_factory=time.time, description="計画作成時刻")
    available_skills: List[Dict[str, Any]] = Field(default_factory=list, description="利用可能なスキルのリスト")

# --- 初期化関数 ---
async def initialize_registry():
    """エージェントレジストリを初期化する"""
    global registry, llm
    global smart_a2a_client, agent_selector
    
    print("エージェントレジストリを初期化中...")
    
    # 設定ファイルからエージェント情報を読み込む
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"設定ファイルを読み込みました: {config_path}")
            
        # 設定ファイルからエージェントURLを取得
        agent_urls = []
        if 'agents' in config:
            for agent_id, agent_info in config['agents'].items():
                if 'url' in agent_info:
                    agent_urls.append(agent_info['url'])
                    print(f"エージェント '{agent_info.get('name', agent_id)}' の URL: {agent_info['url']}")
        
        # URLが見つからない場合はデフォルト値を使用
        if not agent_urls:
            print("警告: 設定ファイルからエージェントURLが見つかりませんでした。デフォルト値を使用します。")
            agent_urls = [
                "http://localhost:8001",  # PurchasingDataAgent
                "http://localhost:8002",  # InventoryDataAgent
                "http://localhost:8003"   # MarketPriceAgent
            ]
    except Exception as e:
        print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
        print("デフォルトのエージェントURLを使用します。")
        agent_urls = [
            "http://localhost:8001",  # PurchasingDataAgent
            "http://localhost:8002",  # InventoryDataAgent
            "http://localhost:8003"   # MarketPriceAgent
        ]
    
    # エージェントを発見して登録
    await registry.discover_agents(agent_urls)
    
    # 登録されたエージェントを確認
    agents = registry.list_agents()
    print(f"登録されたエージェント: {', '.join([agent.name for agent in agents])}")
    
    # LLMベースのスキル選択器を作成
    skill_selector = SkillSelector(model_name="gpt-4o", temperature=0)
    print("LLMベースのスキル選択器を初期化しました。")
    
    # エージェント選択器を初期化
    agent_selector = AgentSelector(llm=llm)
    print("エージェント選択器を初期化しました。")
    
    # スマートクライアントを初期化
    smart_a2a_client = SmartA2AClient(registry, skill_selector=skill_selector)
    print("スマートA2Aクライアントを初期化しました。")

# --- ステップ実行関数 ---
async def execute_step(step: PlanStep) -> PlanStep:
    """
    計画の1ステップを実行する
    
    Args:
        step: 実行するステップ
        
    Returns:
        PlanStep: 実行結果で更新されたステップ
    """
    logger.info(f"ステップ {step.id} ({step.description}) を開始")
    logger.info(f"  スキルID: {step.skill_id}")
    logger.info(f"  入力データ: {json.dumps(step.input_data, ensure_ascii=False)}")
    step.start_time = datetime.now()
    
    try:
        # スキルIDが指定されている場合、対応するエージェントを検索して実行
        if step.skill_id:
            print(f"スキルID '{step.skill_id}' のエージェントを検索中...")
            matching_agents = await registry.find_agents_by_skill(step.skill_id)
            
            if matching_agents:
                print(f"スキル '{step.skill_id}' を持つエージェントが {len(matching_agents)} 件見つかりました")
                # 見つかったエージェントの詳細情報を表示
                for i, agent in enumerate(matching_agents):
                    print(f"  エージェント{i+1}: 名前={agent.name}, URL={agent.url}")
                    # スキルの詳細も表示
                    for skill in agent.skills:
                        if skill.id == step.skill_id:
                            print(f"    スキル詳細: 名前={skill.name}, 説明={skill.description or '説明なし'}")
                
                # 最適なエージェントを選択 (スコアや優先度などのロジックを実装可能)
                selected_agent = matching_agents[0]
                print(f"エージェント '{selected_agent.name}' を選択しました")
                
                try:
                    # SmartA2AClientを使用してタスクを送信
                    input_message = Message(
                        role="user",
                        parts=[DataPart(data=step.input_data)]
                    )
                    
                    # ベースURLを設定
                    base_url = selected_agent.url
                    smart_a2a_client.client.url = f"{base_url}/a2a"
                    
                    print(f"エージェント {selected_agent.name} にタスクを送信中...")
                    task_response = await smart_a2a_client.client.send_task(
                        payload={
                            "id": str(uuid4()),
                            "message": input_message
                        }
                    )
                    
                    # レスポンスからタスクIDを取得
                    if not task_response.result or not task_response.result.id:
                        raise ValueError("エージェントからタスクIDを取得できませんでした")
                    
                    task_id = task_response.result.id
                    print(f"タスクが送信されました。タスクID: {task_id}")
                    
                    # 結果をポーリングで待機
                    max_retries = 10
                    poll_interval = 2
                    result_text = None
                    
                    for attempt in range(max_retries):
                        print(f"結果をポーリング中... 試行 {attempt + 1}/{max_retries}")
                        task_response = await smart_a2a_client.client.get_task(
                            payload={
                                "id": task_id
                            }
                        )
                        
                        # 完了チェック
                        if task_response.result and task_response.result.status.state == TaskState.COMPLETED:
                            result_message = task_response.result.status.message
                            if result_message and result_message.parts and isinstance(result_message.parts[0], TextPart):
                                result_text = result_message.parts[0].text
                                print(f"タスク結果を取得しました")
                                break
                        
                        # エラーチェック
                        if task_response.result and task_response.result.status.state in [TaskState.FAILED, TaskState.CANCELED]:
                            error_msg = "タスクが失敗またはキャンセルされました"
                            if task_response.result.status.message:
                                for part in task_response.result.status.message.parts:
                                    if isinstance(part, TextPart):
                                        error_msg = part.text
                            raise ValueError(f"エージェントタスクエラー: {error_msg}")
                        
                        # まだ処理中の場合は待機
                        await asyncio.sleep(poll_interval)
                    
                    # 結果を設定
                    if result_text:
                        step.is_completed = True
                        step.output_data = {
                            "result": result_text,
                            "agent": selected_agent.name
                        }
                    else:
                        step.is_completed = False
                        step.error = "タスク結果の取得がタイムアウトしました"

                except Exception as e:
                    print(f"エージェント呼び出し中にエラーが発生しました: {e}")
                    import traceback
                    traceback.print_exc()
                    step.is_completed = False
                    step.error = f"エージェント呼び出しエラー: {str(e)}"
            else:
                print(f"スキル '{step.skill_id}' を持つエージェントが見つかりませんでした")
                
                # 自己完結型処理の追加: エージェントが見つからない場合の処理
                if step.skill_id == "self_processing" or step.skill_id == "calculate_and_compare" or "calculate" in step.skill_id or "compare" in step.skill_id:
                    print(f"スキル '{step.skill_id}' は内部処理として実行します")
                    
                    # 前のステップから販売価格と市場価格のデータを取得
                    sales_price = None
                    market_price = None
                    threshold = 5.0  # デフォルト値
                    
                    # 入力データから情報を取得
                    for key, value in step.input_data.items():
                        if isinstance(value, str):
                            # 販売価格を探す（「販売価格: 」または「価格: 」のパターンに対応）
                            sales_match = re.search(r'(?:販売)?価格[=は:：]?\s*(\d+(?:,\d+)*)', value)
                            if sales_match and not sales_price:
                                sales_price_str = sales_match.group(1).replace(',', '')
                                try:
                                    sales_price = int(sales_price_str)
                                    print(f"販売価格を抽出しました: {sales_price}円")
                                except ValueError:
                                    pass
                                    
                            # 市場価格を探す
                            market_match = re.search(r'市場価格[=は:：]?\s*(\d+(?:,\d+)*)', value)
                            if market_match and not market_price:
                                market_price_str = market_match.group(1).replace(',', '')
                                try:
                                    market_price = int(market_price_str)
                                    print(f"市場価格を抽出しました: {market_price}円")
                                except ValueError:
                                    pass
                    
                    # パラメータからしきい値を取得
                    if 'threshold' in step.parameters:
                        threshold = float(step.parameters['threshold'])
                    elif 'threshold' in step.input_data:
                        threshold = float(step.input_data['threshold'])
                        
                    # 十分なデータがあるか確認
                    if sales_price is None or market_price is None:
                        step.is_completed = False
                        step.error = "必要なデータ（販売価格または市場価格）が前のステップから取得できませんでした"
                        return step
                    
                    # 乖離率を計算
                    deviation = abs(sales_price - market_price) / market_price * 100
                    is_anomaly = deviation >= threshold
                    
                    result_text = (
                        f"乖離率計算結果:\n"
                        f"販売価格: {sales_price}円\n"
                        f"市場価格: {market_price}円\n"
                        f"乖離率: {deviation:.2f}%\n"
                        f"しきい値: {threshold}%\n"
                        f"結論: 価格は{'異常' if is_anomaly else '正常'}です。乖離率は{deviation:.2f}%{'で、しきい値を超えています' if is_anomaly else 'で、正常範囲内です'}。"
                    )
                    
                    print(f"内部処理による計算が完了しました。乖離率: {deviation:.2f}%")
                    step.is_completed = True
                    step.output_data = {
                        "agent": "SmartKakakuSignalAgent (内部処理)",
                        "result": result_text
                    }
                    return step
                else:
                    step.is_completed = False
                    step.error = f"スキル '{step.skill_id}' に対応するエージェントが見つかりません"
        else:
            print(f"警告: スキルIDが指定されていません。手動実行ステップとして処理します。")
            # 手動実行ステップの処理などをここに追加可能
            step.is_completed = True
            step.output_data = {"result": "手動実行ステップが完了しました"}
        
        return step
        
    except Exception as e:
        print(f"ステップ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        step.is_completed = False
        step.error = str(e)
        return step

if __name__ == "__main__":
    # 直接実行された場合はテストモジュール
    print("独立実行のためのテストモードを開始します")
    print("シナリオベース異常検知エージェントのモジュールファイルです") 