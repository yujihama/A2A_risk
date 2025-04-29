"""
エージェントカードの情報を活用してLLMベースで最適なエージェントを選択するモジュール
"""

import json
from typing import List, Tuple, Optional, Dict, Any
from pydantic.v1 import BaseModel, Field

from langchain_openai import ChatOpenAI
from A2A_risk.samples.python.common.types import AgentCard, AgentSkill

class AgentRanking(BaseModel):
    """エージェントのランキング結果を表すモデル"""
    agent_name: str = Field(description="エージェント名")
    score: float = Field(description="適合スコア（0.0～1.0）")
    reasoning: str = Field(description="選択理由")

class AgentSelector:
    """
    エージェントカードの情報を分析して、最適なエージェントを選択するクラス
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        初期化関数
        
        Args:
            llm: 使用するLLMモデル（指定されない場合は新規作成）
        """
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    
    async def select_agent(
        self, 
        agents: List[AgentCard], 
        skill_id: str,
        query: str
    ) -> Tuple[AgentCard, str]:
        """
        複数のエージェントの中から、指定されたスキルと問い合わせに最適なエージェントを選択する
        
        Args:
            agents: 選択候補となるエージェントのリスト
            skill_id: 必要なスキルID
            query: 実行したい問い合わせ内容
            
        Returns:
            選択されたエージェントとその選択理由のタプル
        """
        if not agents:
            raise ValueError("選択するエージェントがありません")
        
        if len(agents) == 1:
            # 候補が1つしかない場合は自動的に選択
            return agents[0], "利用可能なエージェントが1つしかないため自動選択"
        
        # 各エージェントのカード情報から選択に有用な情報を抽出
        agent_info = []
        for agent in agents:
            # スキル情報を抽出
            matching_skill = None
            for skill in agent.skills:
                if skill.id == skill_id:
                    matching_skill = skill
                    break
            
            if not matching_skill:
                continue
                
            # エージェント情報を構造化
            agent_info.append({
                "name": agent.name,
                "description": agent.description or "説明なし",
                "url": agent.url,
                "version": agent.version,
                "provider": agent.provider.organization if agent.provider else "不明",
                "skill": {
                    "id": matching_skill.id,
                    "name": matching_skill.name,
                    "description": matching_skill.description or "説明なし",
                    "tags": matching_skill.tags or [],
                    "examples": matching_skill.examples or []
                }
            })
        
        # LLMへのプロンプト作成
        prompt = f"""
あなたはエージェント選択の専門家です。以下の条件に基づいて、与えられたクエリに最適なエージェントを選択してください。

クエリ: {query}
必要なスキル: {skill_id}

利用可能なエージェント:
{json.dumps(agent_info, ensure_ascii=False, indent=2)}

各エージェントについて、クエリと必要なスキルへの適合度を0.0～1.0のスコアで評価し、選択理由を説明してください。
以下のJSON形式で回答してください:

```json
[
  {{
    "agent_name": "エージェント名",
    "score": 0.95,
    "reasoning": "このエージェントを選択する理由"
  }},
  ...
]
```

最も適合度の高いエージェントを最初に配置してください。
"""
        
        try:
            # LLMに分析させる
            response = await self.llm.ainvoke([
                {"role": "system", "content": "あなたはエージェント選択の専門家です。情報を分析して最適なエージェントを選択します。"},
                {"role": "user", "content": prompt}
            ])
            
            # JSONを抽出
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[\s*{.*}\s*\]', response.content, re.DOTALL)
            
            if not json_match:
                # JSONが見つからない場合はデフォルト
                return agents[0], "LLMからの応答を解析できなかったため、最初のエージェントを選択"
            
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            rankings = json.loads(json_str)
            
            if not rankings:
                return agents[0], "ランキング結果が空のため、最初のエージェントを選択"
            
            # 最高スコアのエージェントを取得
            top_ranking = rankings[0]
            agent_name = top_ranking.get("agent_name")
            reasoning = top_ranking.get("reasoning", "理由なし")
            
            # 名前で対応するエージェントを検索
            selected_agent = None
            for agent in agents:
                if agent.name == agent_name:
                    selected_agent = agent
                    break
            
            # 見つからない場合は最初のエージェントを選択
            if not selected_agent:
                return agents[0], f"指定された名前 '{agent_name}' のエージェントが見つからないため、最初のエージェントを選択"
            
            return selected_agent, reasoning
            
        except Exception as e:
            print(f"エージェント選択中にエラーが発生しました: {e}")
            # エラー時は最初のエージェントを返す
            return agents[0], f"エラーが発生したため、最初のエージェントを選択: {e}"
    
    def _extract_skill_info(self, agent: AgentCard, skill_id: str) -> Dict[str, Any]:
        """
        エージェントから指定されたスキルの情報を抽出する
        
        Args:
            agent: エージェントカード
            skill_id: 取得したいスキルID
            
        Returns:
            スキル情報を含む辞書
        """
        for skill in agent.skills:
            if skill.id == skill_id:
                return {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description or "説明なし",
                    "tags": skill.tags or [],
                    "examples": skill.examples or []
                }
        
        return {
            "id": skill_id,
            "name": "不明",
            "description": "エージェントに該当するスキルが見つかりません",
            "tags": [],
            "examples": []
        } 