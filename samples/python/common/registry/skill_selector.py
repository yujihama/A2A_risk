import logging
import os
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from A2A_risk.samples.python.common.types import AgentCard, AgentSkill, Message

logger = logging.getLogger(__name__)

class SkillMatchResult(BaseModel):
    """スキル選択の結果を表すモデル"""
    skill_id: str = Field(description="選択されたスキルのID")
    confidence: float = Field(description="選択の確信度（0.0～1.0）")
    reasoning: str = Field(description="選択の理由")

class SkillSelector:
    """メッセージの内容を分析して適切なスキルを選択するクラス"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        """
        初期化関数
        
        Args:
            model_name: 使用するLLMモデル名
            temperature: モデルの温度パラメータ
        """
        # 環境変数からAPIキーが設定されていることを確認
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
            
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # 明確なJSON形式を指示するシステムプロンプト
        self.system_prompt = """あなたはメッセージを分析して、最適なスキルを選択するAIアシスタントです。
利用可能なスキルの中から、メッセージの内容に最も適したスキルを選んでください。

利用可能なスキル：
- search_product_data: 製品情報（価格、仕様など）を検索するためのスキル
- search_inventory_data: 在庫情報（在庫数、入荷予定など）を検索するためのスキル

あなたは以下の厳密なJSON形式で回答してください。他の形式は受け付けられません：
{
  "skill_id": "検索したスキルのID (search_product_dataまたはsearch_inventory_data)",
  "confidence": 信頼度 (0.0〜1.0の数値),
  "reasoning": "選択した理由を説明する短いテキスト"
}

例：
{
  "skill_id": "search_product_data",
  "confidence": 0.9,
  "reasoning": "メッセージには製品の価格情報の要求が含まれているため"
}

必ず有効なJSONを返してください。追加の説明やマークダウンは不要です。"""
        
    async def select_skill(self, message_text: str) -> SkillMatchResult:
        """
        メッセージの内容から最適なスキルを選択する
        
        Args:
            message_text: 分析するメッセージのテキスト
            
        Returns:
            SkillMatchResult: 選択されたスキルの情報
        """
        try:
            # LLMからの回答を取得
            response = await self.llm.ainvoke([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message_text}
            ])
            
            # レスポンスを表示してデバッグ
            print(f"LLMレスポンス: {response.content}")
            
            # 手動でJSONをパースして辞書に変換してから、SkillMatchResultに変換
            import json
            import re
            
            # JSONを抽出するための正規表現
            json_pattern = r'```json\s*(.*?)\s*```|{.*}'
            match = re.search(json_pattern, response.content, re.DOTALL)
            
            if match:
                json_str = match.group(1) or match.group(0)
                # 余分な文字を取り除く
                json_str = json_str.strip().replace('```', '')
                
                try:
                    # JSONをパース
                    data = json.loads(json_str)
                    
                    # 辞書からSkillMatchResultオブジェクトを作成
                    return SkillMatchResult(
                        skill_id=data.get('skill_id', 'search_product_data'),
                        confidence=data.get('confidence', 0.8),
                        reasoning=data.get('reasoning', 'LLMから取得した理由')
                    )
                except json.JSONDecodeError as e:
                    print(f"JSONパースエラー: {e}, 内容: {json_str}")
            
            # フォールバック: テキスト解析で簡易的に判断
            if "inventory" in response.content.lower() or "在庫" in response.content.lower():
                return SkillMatchResult(
                    skill_id="search_inventory_data",
                    confidence=0.7,
                    reasoning="テキスト解析による在庫情報の検出"
                )
            else:
                return SkillMatchResult(
                    skill_id="search_product_data",
                    confidence=0.7,
                    reasoning="デフォルトの製品情報スキル"
                )
            
        except Exception as e:
            print(f"スキル選択中にエラーが発生しました: {e}")
            return SkillMatchResult(
                skill_id="search_product_data",
                confidence=0.5,
                reasoning="エラーが発生したため、デフォルトのスキルを使用します。"
            )

    async def select_skill_from_message(
        self, 
        message: Message, 
        available_skills: List[AgentSkill]
    ) -> Optional[str]:
        """
        ユーザーメッセージから適切なスキルIDを選択する
        
        Args:
            message: ユーザーのメッセージ
            available_skills: 利用可能なスキルのリスト
            
        Returns:
            選択されたスキルID、またはNone（適切なスキルが見つからない場合）
        """
        if not available_skills:
            logger.warning("No skills available to select from")
            return None
        
        # メッセージからテキスト部分を抽出
        message_text = ""
        for part in message.parts:
            if hasattr(part, "text"):
                message_text += part.text + "\n"
        
        if not message_text.strip():
            logger.warning("No text content in message")
            return None
        
        # スキル情報をLLMに理解しやすい形式に変換
        skills_info = []
        for i, skill in enumerate(available_skills):
            skills_info.append(
                f"スキル {i+1}:\n"
                f"ID: {skill.id}\n"
                f"名前: {skill.name}\n"
                f"説明: {skill.description}\n"
            )
        
        skills_text = "\n".join(skills_info)
        
        # プロンプトの作成
        prompt = f"""
あなたは以下のスキルリストから、ユーザーのメッセージに最も適したスキルを選択する助手です。

利用可能なスキル:
{skills_text}

ユーザーのメッセージ:
{message_text}

上記のメッセージを分析し、最も適切なスキルのIDを選択してください。
複数のスキルが該当する場合は、最も関連性の高いものを1つだけ選んでください。
該当するスキルがない場合は「該当なし」と回答してください。

回答は選択したスキルのIDのみを出力してください。説明は不要です。
"""

        try:
            # LLMからの回答を取得
            response = await self.llm.ainvoke(prompt)
            skill_id = response.content.strip()
            
            # 「該当なし」の場合はNoneを返す
            if skill_id in ["該当なし", "None", "なし"]:
                logger.info("LLM determined no matching skill")
                return None
                
            # 回答が有効なスキルIDかチェック
            valid_skill_ids = [skill.id for skill in available_skills]
            if skill_id in valid_skill_ids:
                logger.info(f"Selected skill: {skill_id}")
                return skill_id
            
            # スキルIDが正確に一致しない場合は類似度で検索
            for valid_id in valid_skill_ids:
                if valid_id.lower() in skill_id.lower() or skill_id.lower() in valid_id.lower():
                    logger.info(f"Matched approximate skill ID: {valid_id}")
                    return valid_id
            
            logger.warning(f"LLM returned invalid skill ID: {skill_id}")
            return None
        
        except Exception as e:
            logger.error(f"Error selecting skill: {e}")
            return None 