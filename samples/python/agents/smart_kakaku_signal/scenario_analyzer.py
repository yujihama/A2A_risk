import re
import json
import logging
from typing import Dict, Any, List, Optional

# ロガーの設定
logger = logging.getLogger(__name__)

class ScenarioAnalyzer:
    """
    自然言語で書かれたシナリオを解析し、必要なデータと判定ロジックを抽出するクラス
    """
    
    def __init__(self, llm_client):
        """
        初期化
        
        Args:
            llm_client: LLMクライアント（ChatOpenAIなど）
        """
        self.llm_client = llm_client
        
    async def analyze_scenario(self, scenario_text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        シナリオ文から必要な情報を抽出する
        
        Args:
            scenario_text: 自然言語で書かれたシナリオテキスト
            parameters: パラメータ

        Returns:
            Dict: 抽出された情報（必要なデータと判定ロジック）
        """
        logger.info(f"シナリオを分析中: {scenario_text[:100]}...")
        
        prompt = f"""
あなたは与えられたリスクシナリオから不正の兆候を検知するエキスパートです。
以下のリスクシナリオを分析して、不正の兆候を検知するための判定条件を特定してください。

シナリオ:
{scenario_text}

パラメータ:
{parameters}

以下の形式でJSONを返してください:
```json
{{
  "decision_logic": {{
    "description": "判定ロジックの説明",
    "operations": [
      {{
        "type": "タスクタイプ（例: comparison, average, ratio）",
        "description": "何をするためのタスクか",
        "input_values": ["タスクに使用する情報のリスト"],
        "output_value": ["タスクによって取得したい情報"]
      }},
      ...
    ],
    "anomaly_condition": {{
      "description": "不正と判定する条件の説明"
    }}
  }}
}}
```
"""
        
        try:
            # LLMにシナリオを分析させる
            response = await self.llm_client.ainvoke([
                {"role": "system", "content": "あなたは優秀な内部監査人です。リスクシナリオから判定ロジックを抽出します。"},
                {"role": "user", "content": prompt}
            ])
            
            # JSONを抽出
            scenario_analysis = self._extract_json_from_response(response.content)
            
            return scenario_analysis
            
        except Exception as e:
            logger.error(f"シナリオ分析中にエラーが発生しました: {e}")
            raise
    
    def _extract_json_from_response(self, response_content: str) -> Dict[str, Any]:
        """
        LLMの応答からJSONを抽出する
        
        Args:
            response_content: LLMの応答テキスト
            
        Returns:
            Dict: 抽出されたJSON
            
        Raises:
            ValueError: JSONが抽出できない場合
        """
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'{.*}', response_content, re.DOTALL)
            if not json_match:
                raise ValueError("LLMの応答からJSONを抽出できませんでした")
        
        json_str = json_match.group(1) if "```json" in response_content else json_match.group(0)
        return json.loads(json_str.strip().replace('```', '')) 