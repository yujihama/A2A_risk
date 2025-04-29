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
from A2A_risk.samples.python.common.registry.agent_registry import AgentRegistry
from A2A_risk.samples.python.common.client.smart_client import SmartA2AClient
from A2A_risk.samples.python.common.registry.skill_selector import SkillSelector
from A2A_risk.samples.python.common.types import Task, Message, TextPart, TaskState, Part, AgentCard, DataPart

# エージェント選択機能をインポート
from .agent_selector import AgentSelector

# .envファイルから環境変数を読み込む
load_dotenv()

# --- グローバル変数 ---
registry = AgentRegistry()
smart_a2a_client = None  # 初期化は非同期で行うため、別途初期化関数を用意
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)  # 計画立案用LLM
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
    transitions: Dict[str, str] = Field(default_factory=dict, description="判定条件ごとの遷移先ノードID")

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
async def initialize_registry(config: Dict[str, Any]):
    """エージェントレジストリを初期化する

    Args:
        config: 設定ファイルの内容
    """
    global registry, llm
    global smart_a2a_client, agent_selector

    print("エージェントレジストリを初期化中...")

    try:
        # 設定ファイルからエージェントURLを取得
        agent_urls = []
        if 'agents' in config:
            for agent_id, agent_info in config['agents'].items():
                if 'url' in agent_info:
                    agent_urls.append(agent_info['url'])
                    print(f"エージェント '{agent_info.get('name', agent_id)}' の URL: {agent_info['url']}")

    except Exception as e:
        print(f"設定の解析中にエラーが発生しました: {e}")
        
    # エージェントを発見して登録
    await registry.discover_agents(agent_urls)
    
    # 登録されたエージェントを確認
    agents = registry.list_agents()
    print(f"登録されたエージェント: {', '.join([agent.name for agent in agents])}")
    
    # LLMベースのスキル選択器を作成
    skill_selector = SkillSelector(model_name="gpt-4.1-mini", temperature=0)
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
    step.start_time = datetime.now()
    
    try:
        # スキルIDが指定されている場合、対応するエージェントを検索して実行
        if step.skill_id:
            print(f"スキルID '{step.skill_id}' のエージェントを検索中...")
            matching_agents = await registry.find_agents_by_skill(step.skill_id)
            
            if matching_agents:
                print(f"スキル '{step.skill_id}' を持つエージェントが {len(matching_agents)} 件見つかりました")

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
                    
                    logger.info(f"エージェント {selected_agent.name} にタスクを送信中...")
                    logger.info(f"input_message: {input_message}")
                    task_response = await smart_a2a_client.client.send_task(
                        payload={
                            "id": str(uuid4()),
                            "message": input_message
                        }
                    )
                    logger.debug(f"task_response: {task_response}")
                    # レスポンスからタスクIDを取得
                    if not task_response.result or not task_response.result.id:
                        raise ValueError("エージェントからタスクIDを取得できませんでした")
                    
                    task_id = task_response.result.id
                    logger.info(f"タスクが送信されました。タスクID: {task_id}")
                    
                    # 結果をポーリングで待機
                    max_retries = 30
                    poll_interval = 4
                    result_message = None
                    
                    for attempt in range(max_retries):
                        logger.debug(f"結果をポーリング中... 試行 {attempt + 1}/{max_retries}")
                        task_response = await smart_a2a_client.client.get_task(
                            payload={
                                "id": task_id
                            }
                        )
                    
                        # 完了チェック
                        if task_response.result and task_response.result.status.state == TaskState.COMPLETED:
                            logger.debug(f"task_response: {task_response}")
                            result_message = task_response.result.status.message
                            if result_message and result_message.parts:
                                text = None
                                data = None
                                for part in result_message.parts:
                                    if isinstance(part, TextPart):
                                        text = part.text
                                    elif isinstance(part, DataPart):
                                        data = part.data
                                logger.info(f"タスク結果を取得しました")
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
                    if result_message and result_message.parts:
                        text = None
                        data = None
                        for part in result_message.parts:
                            if isinstance(part, TextPart):
                                text = part.text
                            elif isinstance(part, DataPart):
                                data = part.data
                        step.is_completed = True
                        step.output_data = {
                            "text": text,
                            "data": data,
                            "agent": selected_agent.name
                        }
                    else:
                        step.is_completed = False
                        step.error = "タスク結果の取得がタイムアウトしました"

                except Exception as e:
                    logger.error(f"エージェント呼び出し中にエラーが発生しました: {e}")
                    import traceback
                    traceback.print_exc()
                    step.is_completed = False
                    step.error = f"エージェント呼び出しエラー: {str(e)}"
            else:
                print(f"スキル '{step.skill_id}' を持つエージェントが見つかりませんでした")
                
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