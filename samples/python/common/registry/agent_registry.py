import logging
import httpx
from typing import List, Dict, Optional, Set

from samples.python.common.types import AgentCard, AgentSkill

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    A2Aエコシステム内のエージェントを管理するレジストリクラス。
    Agent Cardの保存、取得、検索機能を提供する。
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentCard] = {}
        logger.info("AgentRegistry initialized.")
    
    async def register_agent(self, agent_card: AgentCard) -> None:
        """
        エージェントを登録する。既存のエージェントは上書きされる。
        """
        self.agents[agent_card.name] = agent_card
        logger.info(f"Agent '{agent_card.name}' registered with URL: {agent_card.url}")
    
    async def discover_agent(self, url: str) -> Optional[AgentCard]:
        """
        指定されたURLからAgent Cardを取得して登録する。
        """
        try:
            well_known_url = f"{url.rstrip('/')}/.well-known/agent.json"
            logger.info(f"Discovering agent at: {well_known_url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(well_known_url, timeout=10.0)
                response.raise_for_status()
                
                # レスポンスをAgentCardにパース
                agent_card_data = response.json()
                agent_card = AgentCard.model_validate(agent_card_data)
                
                # レジストリに登録
                await self.register_agent(agent_card)
                return agent_card
                
        except httpx.RequestError as e:
            logger.error(f"Agent discovery failed - Request error: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Agent discovery failed - HTTP error {e.response.status_code}: {e}")
        except Exception as e:
            logger.error(f"Agent discovery failed - Unexpected error: {e}")
            
        return None
    
    async def discover_agents(self, urls: List[str]) -> List[AgentCard]:
        """
        複数のURLからエージェントを発見して登録する。
        """
        discovered_agents = []
        
        for url in urls:
            agent_card = await self.discover_agent(url)
            if agent_card:
                discovered_agents.append(agent_card)
        
        logger.info(f"Discovered {len(discovered_agents)} agents from {len(urls)} URLs.")
        return discovered_agents
    
    async def find_agents_by_skill(self, skill_id: str) -> List[AgentCard]:
        """
        指定されたスキルIDを持つエージェントを検索する。
        """
        matching_agents = []
        
        for agent_name, card in self.agents.items():
            for skill in card.skills:
                if skill.id == skill_id:
                    matching_agents.append(card)
                    break  # 同じエージェントで複数のスキルがマッチする場合は一度だけ追加
        
        logger.info(f"Found {len(matching_agents)} agents with skill '{skill_id}'.")
        return matching_agents
    
    async def get_all_skills(self) -> List[AgentSkill]:
        """
        登録されているすべてのエージェントから利用可能なスキルを取得する。
        重複するスキルIDは除外される。
        
        Returns:
            すべてのユニークなスキルのリスト
        """
        all_skills = []
        unique_skill_ids: Set[str] = set()
        
        for agent_card in self.agents.values():
            for skill in agent_card.skills:
                if skill.id not in unique_skill_ids:
                    all_skills.append(skill)
                    unique_skill_ids.add(skill.id)
        
        logger.info(f"Collected {len(all_skills)} unique skills from all agents.")
        return all_skills
    
    def get_agent(self, agent_name: str) -> Optional[AgentCard]:
        """
        名前でエージェントを取得する。
        """
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[AgentCard]:
        """
        登録されているすべてのエージェントを一覧表示する。
        """
        return list(self.agents.values()) 