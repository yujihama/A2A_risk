import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

# ロガーの設定
logger = logging.getLogger(__name__)

class DynamicDataAnalyzer:
    """
    収集されたデータをシナリオに基づいて分析するクラス
    """
    
    def __init__(self, llm_client):
        """
        初期化
        
        Args:
            llm_client: LLMクライアント（ChatOpenAIなど）
        """
        self.llm_client = llm_client
    
    async def analyze_collected_data(
        self, 
        scenario_text: str, 
        scenario_analysis: Dict[str, Any], 
        step_results: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        収集されたデータを分析してシナリオの条件に基づき判定する
        
        Args:
            scenario_text: 元のシナリオテキスト
            scenario_analysis: シナリオ分析結果
            step_results: 各ステップの実行結果
            parameters: 入力パラメータ
            
        Returns:
            Dict: 分析結果
        """
        logger.info("収集データを分析中...")
        
        # 各ステップの結果をテキスト形式で整形
        collected_data_text = ""
        for step_id, result_info in step_results.items():
            collected_data_text += f"\n--- ステップ {step_id}: {result_info['description']} ---\n"
            collected_data_text += f"{result_info['result']}\n"
        
        # 決定ロジックを抽出
        decision_logic = scenario_analysis.get('decision_logic', {})
        
        # LLMによる分析プロンプト
        prompt = f"""
あなたはデータアナリストAIとして、以下のシナリオに基づいてデータを分析し、異常があるかどうかを判定します。

シナリオ:
{scenario_text}

収集されたデータ:
{collected_data_text}

判定ロジック:
{json.dumps(decision_logic, ensure_ascii=False, indent=2)}

パラメータ:
{json.dumps(parameters, ensure_ascii=False, indent=2)}

以下のタスクを実行してください:
1. 収集されたデータから必要な値を抽出する
2. 判定ロジックに従って計算や比較を行う
3. 異常の有無を判定する
4. 判定の根拠と詳細を説明する

以下のJSON形式で回答してください:
```json
{{
  "is_anomaly": true または false または null,
  "is_data_sufficient": true または false,
  "extracted_values": {{
    "値の名前": 抽出した値,
    ...
  }},
  "calculations": [
    {{
      "description": "計算の説明",
      "formula": "計算式",
      "result": 計算結果
    }},
    ...
  ],
  "decision": {{
    "description": "判定の説明",
    "condition": "判定条件",
    "result": true または false
  }},
  "analysis": "詳細な分析結果と説明"
}}
```

重要: データが不十分な場合は判断を保留し、is_anomalyをnullとし、is_data_sufficientをfalseとしてください。
"""
        
        try:
            # LLMにデータを分析させる
            response = await self.llm_client.ainvoke([
                {"role": "system", "content": "あなたはデータ分析と異常検知のエキスパートです。シナリオに基づいてデータを分析し、異常を検出します。"},
                {"role": "user", "content": prompt}
            ])
            
            # JSONを抽出
            analysis_result = self._extract_json_from_response(response.content)
            
            logger.info(f"データ分析完了: 異常検出={analysis_result.get('is_anomaly')}, データ十分性={analysis_result.get('is_data_sufficient')}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"データ分析中にエラーが発生しました: {e}")
            # エラー時は判断保留の結果を返す
            return {
                "is_anomaly": None,
                "is_data_sufficient": False,
                "extracted_values": {},
                "calculations": [],
                "decision": {
                    "description": "分析中にエラーが発生しました",
                    "condition": "エラー",
                    "result": None
                },
                "analysis": f"分析中にエラーが発生しました: {str(e)}"
            }
    
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