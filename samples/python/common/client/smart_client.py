import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from uuid import uuid4

from A2A_risk.samples.python.common.client.client import A2AClient
from A2A_risk.samples.python.common.types import Message
from A2A_risk.samples.python.common.registry.agent_registry import AgentRegistry
from A2A_risk.samples.python.common.registry.skill_selector import SkillSelector

logger = logging.getLogger(__name__)

class SmartA2AClient:
    """
    A2Aクライアントの拡張版。エージェントレジストリを用いて
    動的なエージェント選択を行う。
    """
    
    def __init__(self, registry: AgentRegistry, skill_selector: Optional[SkillSelector] = None):
        """
        エージェントレジストリを指定してスマートA2Aクライアントを初期化する。
        
        Args:
            registry: エージェントレジストリ
            skill_selector: スキル選択器（指定されていない場合は新しく作成）
        """
        self.registry = registry
        # 基本クライアントにはデフォルトURLを指定して初期化
        self.client = A2AClient(url="http://localhost:8001/a2a")  # デフォルトURL（あとで上書きされる）
        self.skill_selector = skill_selector or SkillSelector()
        logger.info("SmartA2AClient initialized with registry and skill selector.")
    
    async def find_and_send_task(
        self, 
        skill_id: str, 
        message: Message, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        指定されたスキルを持つエージェントを検索し、タスクを送信する。
        
        Args:
            skill_id: 必要なスキルのID
            message: 送信するメッセージ
            **kwargs: 追加のペイロードパラメータ
            
        Returns:
            タスクレスポンス
            
        Raises:
            ValueError: 適切なエージェントが見つからない場合
        """
        # 適切なエージェントを検索
        matching_agents = await self.registry.find_agents_by_skill(skill_id)
        
        if not matching_agents:
            error_msg = f"スキルID '{skill_id}' を持つエージェントが見つかりません。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 最適なエージェントを選択（単純な例では最初のものを使用）
        # TODO: 評価指標を使った洗練された選択アルゴリズムに拡張可能
        selected_agent = matching_agents[0]
        logger.info(f"Selected agent '{selected_agent.name}' with URL: {selected_agent.url}")
        
        # 選択したエージェントにタスクを送信
        self.client.url = f"{selected_agent.url}/a2a"  # エンドポイント(/a2a)を追加
        
        # ペイロードを準備
        payload = {
            "id": str(uuid4()),
            "message": message,
            **kwargs
        }
        
        logger.info(f"Sending task to {selected_agent.name} with skill {skill_id}")
        return await self.client.send_task(payload=payload)
    
    async def auto_select_and_send_task(
        self,
        message: Message,
        **kwargs
    ) -> Dict[str, Any]:
        """
        メッセージの内容を分析し、最適なスキルを自動選択してタスクを送信する
        
        Args:
            message: 送信するメッセージ
            **kwargs: 追加のペイロードパラメータ
            
        Returns:
            タスクレスポンス
            
        Raises:
            ValueError: 適切なスキルまたはエージェントが見つからない場合
        """
        # 利用可能なすべてのスキルを取得
        all_skills = await self.registry.get_all_skills()
        
        if not all_skills:
            error_msg = "登録されているスキルがありません。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # LLMを使用してスキルを選択
        selected_skill_id = await self.skill_selector.select_skill(message, all_skills)
        
        if not selected_skill_id:
            error_msg = "メッセージに適したスキルが見つかりませんでした。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Automatically selected skill: {selected_skill_id}")
        
        # 選択したスキルを使用してタスクを送信
        return await self.find_and_send_task(selected_skill_id, message, **kwargs)
    
    async def get_task(self, task_id: str, agent_url: Optional[str] = None) -> Dict[str, Any]:
        """
        指定されたタスクIDのタスク状態を取得する。
        オプションでエージェントURLを指定できる。
        
        Args:
            task_id: タスクID
            agent_url: エージェントのURL（指定されていない場合は直前に使用したURLを使用）
            
        Returns:
            タスク状態のレスポンス
        """
        if agent_url:
            # エンドポイントを追加
            self.client.url = f"{agent_url}/a2a" if not agent_url.endswith("/a2a") else agent_url
            
        logger.info(f"Getting task {task_id} from {self.client.url}")
        return await self.client.get_task(payload={"id": task_id})
    
    async def cancel_task(self, task_id: str, agent_url: Optional[str] = None) -> Dict[str, Any]:
        """
        指定されたタスクIDのタスクをキャンセルする。
        オプションでエージェントURLを指定できる。
        
        Args:
            task_id: タスクID
            agent_url: エージェントのURL（指定されていない場合は直前に使用したURLを使用）
            
        Returns:
            キャンセル結果のレスポンス
        """
        if agent_url:
            # エンドポイントを追加
            self.client.url = f"{agent_url}/a2a" if not agent_url.endswith("/a2a") else agent_url
            
        logger.info(f"Canceling task {task_id} from {self.client.url}")
        return await self.client.cancel_task(payload={"id": task_id})
    
    async def find_and_send_tasks_to_all(
        self, 
        skill_id: str, 
        message: Message
    ) -> List[Dict[str, Any]]:
        """
        指定されたスキルIDを持つすべてのエージェントにタスクを送信する
        
        Args:
            skill_id: 検索するスキルID
            message: 送信するメッセージ
            
        Returns:
            各エージェントからのレスポンスのリスト
        """
        agents = await self.registry.find_agents_by_skill(skill_id)
        
        if not agents:
            logger.warning(f"スキル '{skill_id}' を持つエージェントが見つかりませんでした")
            return []
            
        responses = []
        for agent in agents:
            logger.info(f"エージェント '{agent.name}' にタスクを送信します")
            self.client.url = agent.url
            
            try:
                # ペイロードを準備
                payload = {
                    "message": message.model_dump() if hasattr(message, "model_dump") else message
                }
                
                # タスクを送信
                response = await self.client.send_task(payload=payload)
                
                # レスポンスを保存
                responses.append({
                    "agent": agent,
                    "response": response
                })
                
                logger.info(f"エージェント '{agent.name}' からのレスポンス: {response}")
            except Exception as e:
                logger.error(f"エージェント '{agent.name}' へのタスク送信中にエラーが発生しました: {e}")
                responses.append({
                    "agent": agent,
                    "error": str(e)
                })
                
        return responses 