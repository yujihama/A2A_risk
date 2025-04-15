import re
import json
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4
import os

from pydantic.v1 import BaseModel

# 現在のエージェントからの型をインポート
from samples.python.agents.smart_kakaku_signal.agent import ExecutionPlan, PlanStep

# ロガーの設定
logger = logging.getLogger(__name__)

class DynamicPlanGenerator:
    """
    シナリオ分析結果から実行計画を動的に生成するクラス
    """
    
    def __init__(self, llm_client, registry):
        """
        初期化
        
        Args:
            llm_client: LLMクライアント（ChatOpenAIなど）
            registry: エージェントレジストリ
        """
        self.llm_client = llm_client
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        
    async def generate_execution_plan(
        self, 
        scenario_analysis: Dict[str, Any], 
        parameters: Dict[str, Any],
        scenario_text: str
    ) -> ExecutionPlan:
        """
        シナリオ分析結果から実行計画を生成する
        
        Args:
            scenario_analysis: シナリオ分析結果
            parameters: 入力パラメータ
            scenario_text: 元のシナリオテキスト
            
        Returns:
            ExecutionPlan: 生成された実行計画
        """
        self.logger.info("実行計画を生成中...")
        
        # 利用可能なスキルを取得
        all_skills = await self.registry.get_all_skills()
        skill_info = []
        
        for skill in all_skills:
            skill_info.append({
                "id": skill.id,
                "name": skill.name,
                "description": skill.description or "",
                "inputModes": skill.inputModes or [],
                "outputModes": skill.outputModes or [],
                # "tags": skill.tags or [],
                "examples": skill.examples or []
            })
        
        # 利用可能なスキル一覧を詳細にログ出力
        # logger.info(f"利用可能なスキル数: {len(skill_info)}")
        # skill_ids = [skill['name'] for skill in skill_info]
        # logger.info(f"利用可能なスキルID一覧: {', '.join(skill_ids)}")
        
        # シナリオテキストをログに出力
        self.logger.info(f"シナリオテキスト: {scenario_text}")
        
        # シナリオ分析結果をログに出力（簡潔に）
        required_data_summary = [f"{data.get('data_type')}({data.get('product_id')})" for data in scenario_analysis.get('required_data', [])]
        self.logger.info(f"必要なデータ: {', '.join(required_data_summary)}")
        
        # 必要なデータを抽出
        required_data = scenario_analysis.get('required_data', [])
        
        # 決定ロジックを抽出
        decision_logic = scenario_analysis.get('decision_logic', {})
        
        # 利用可能なスキルの簡易ログ
        self.logger.info(f"利用可能なスキルID一覧: {', '.join([skill.id for skill in all_skills])}")
        
        # シナリオテキストのログ
        self.logger.info(f"シナリオテキスト: {scenario_text}")
        
        # 必要なデータの概要
        required_data = scenario_analysis.get('required_data', [])
        required_data_summary = "\n".join([f"  - データタイプ: {data.get('type', 'unknown')}, 製品ID: {data.get('product_id', 'unknown')}" for data in required_data])
        self.logger.info(f"必要なデータ概要:\n{required_data_summary}")
        
        # 決定ロジック
        decision_logic = scenario_analysis.get('decision_logic', '')
        
        # プロンプト全文とLLMレスポンスのログ出力（コメントアウトを解除）
        prompt = f"""
あなたは内部監査人としてリスクシナリオを検証するための実行計画を作成するAIアシスタントです。
利用可能なスキルを組み合わせて、検証に必要なデータの取得から分析、結果生成までの一連のステップを計画してください。
スキルへの指示（input_data）は、必ず明確かつ具体的に、経緯を知らない人にもわかるように丁寧に記載してください。
必要なスキルがない場合は、「analyze」スキルを使用してください。

## リスクシナリオ:
{scenario_text}

## 利用可能なスキル:
{json.dumps(skill_info, ensure_ascii=False, indent=2)}
{{
    "id": "analyze",
    "name": "分析",
    "description": "与えられた情報をもとに比較や分析を行います",
    "inputModes": [
      "text"
    ],
    "outputModes": [
      "text"
    ],
    "examples": [
      "Step1の結果「XXX」とStep2の結果「YYY」をもとに、AAAに該当するデータがあるかを判断してください",
      "「XXX」の結果「YYY」となりましたが、この結果は妥当ですか",
    ]
}}

以下のJSON形式でステップ実行計画を作成してください:
"""
# 必要なデータ: {json.dumps(required_data, ensure_ascii=False, indent=2)}

        # JSONテンプレートは別の文字列として定義（f-stringの外で）
        json_template = """```json
{
  "steps": [
    {
      "id": "step1",
      "skill_id": "利用可能なスキルidのいずれか",
      "description": "スキルで実施したいことの説明",
      "input_data": ["XXXを回答してください。(スキルに渡す指示を明確かつ具体的に)"],
      "output_data": ["XXXについての...(スキルからの回答に含めてほしい情報を明確かつ具体的に)"],
    },
    {
      "id": "step2",
      "skill_id": "利用可能なスキルidのいずれか",
      "description": "スキルで実施したいことの説明",
      "input_data": ["AAAを回答してください。(スキルに渡す指示を明確かつ具体的に)"],
      "output_data": ["AAAについての...(スキルからの回答に含めてほしい情報を明確かつ具体的に)"]
    }
  ]
}
```"""
        # 最後の注意事項
        prompt_notes = """
重要な注意:
1. 必ず上記の「利用可能なスキル」リストに含まれるスキルIDのみを使用してください。該当するスキルがない場合はanalyzeを設定してください。
2. スキルに渡すインプット情報は、具体的かつ明確な指示で記載してください。
3. スキルに渡すインプット情報は、1つのみすることが可能です。
4. 最終ステップはリスクシナリオの結論（異常の有無など）を出力するものにしてください。
"""
        
        # 3つの文字列を結合して最終的なプロンプトを作成
        prompt = prompt + json_template + prompt_notes
        
        self.logger.info(f"LLMプロンプト全文:\n{prompt}")
        
        try:
            # より強力なシステムメッセージを追加してLLMに計画を作成させる
            skill_ids_list = ", ".join([skill.id for skill in all_skills])
            
            system_message = f"""あなたは内部監査のエキスパートです。様々なスキルを保持する担当者へタスクを移管するための計画を作成します。
重要：生成する計画では、必ず以下の利用可能なスキルIDのみを使用してください：
{skill_ids_list} , analyze
またスキルへの指示は、必ず明確かつ具体的に、経緯を知らない担当者にもわかるように丁寧に記載してください。
 - 回答フォーマット
{{
  "steps": [
    {{
      "id": "step1",
      "skill_id": "利用可能なスキルidのいずれか",
      "description": "スキルで実施したいことの説明",
      "input_data": ["XXXを回答してください。(スキルに渡す指示を明確かつ具体的に)"],
      "output_data": ["XXXについての...(スキルからの回答に含めてほしい情報を明確かつ具体的に)"],
    }},
    {{
      "id": "step2",
      "skill_id": "利用可能なスキルidのいずれか",
      "description": "スキルで実施したいことの説明",
      "input_data": ["AAAを回答してください。(スキルに渡す指示を明確かつ具体的に)"],
      "output_data": ["AAAについての...(スキルからの回答に含めてほしい情報を明確かつ具体的に)"]
    }}
  ]
}}

"""

            response = await self.llm_client.ainvoke([
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ], response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "execution_plan_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "description": "ステップの一意なID"},
                                        "skill_id": {"type": "string", "description": "使用するスキルのID"},
                                        "description": {"type": "string", "description": "ステップの説明"},
                                        "input_data": {"type": "array", "items": {"type": "string"}, "description": "スキルへの指示"},
                                        "output_data": {"type": "array", "items": {"type": "string"}, "description": "スキルからの期待される出力形式"}
                                    },
                                    "required": ["id", "skill_id", "description", "input_data", "output_data"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["steps"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            })
            
            # LLMの応答をログに出力
            self.logger.info(f"LLMレスポンス全文:\n{response.content}")
            
            # JSONを抽出
            plan_data = self._parse_plan_from_llm(response.content)
            
            # LLMが生成した計画データをログに出力（スキルIDに焦点を当てる）
            steps_summary = []
            for step in plan_data.get("steps", []):
                # descriptionフィールドがない場合は空文字列を使用する
                description = step.get('description', '')
                if description is None:
                    description = ''
                # フォーマットしたステップ情報を追加
                steps_summary.append(f"{step.get('id')}: {step.get('skill_id')} - {description[:50]}...")
            
            self.logger.info(f"LLMが生成した計画ステップ数: {len(plan_data.get('steps', []))}")
            for step_summary in steps_summary:
                self.logger.info(f"  ステップ: {step_summary}")
            
            # ExecutionPlanオブジェクトを作成
            plan = self._create_plan_from_data(plan_data, parameters, scenario_text)
            
            self.logger.info(f"実行計画が生成されました: {len(plan.steps)}ステップ")
            return plan
            
        except Exception as e:
            self.logger.error(f"実行計画の生成中にエラーが発生しました: {e}")
            # 簡易的なフォールバック計画を返す代わりにエラーを再送出
            raise e
    
    def _parse_plan_from_llm(self, llm_response):
        """
        LLMの応答から実行計画データを抽出する

        Args:
            llm_response: LLMの応答

        Returns:
            dict: 抽出された実行計画データ
        """
        self.logger.info("LLMの応答から実行計画を抽出中...")
        
        try:
            # JSONを抽出する正規表現パターン（Markdownのコードブロックから）
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', llm_response, re.DOTALL)
            
            if json_match:
                # Markdownコードブロックから抽出
                json_str = json_match.group(1)
                self.logger.info("Markdownコードブロックから実行計画を抽出しました")
            else:
                # 直接JSONオブジェクトを探す - 最初と最後の中括弧を含むテキスト全体を抽出
                json_match = re.search(r'(\{[\s\S]*\})', llm_response)
                if json_match:
                    json_str = json_match.group(1)
                    self.logger.info("テキストから直接JSONオブジェクトを抽出しました")
                else:
                    # それでも見つからない場合はテキスト全体を試す
                    self.logger.warning("JSONパターンが見つかりませんでした。テキスト全体を解析します。")
                    json_str = llm_response
            
            # JSON文字列を解析
            try:
                plan_data = json.loads(json_str)
                
                # 解析したJSONの内容をログに出力
                self.logger.info(f"解析したJSON: {json.dumps(plan_data, ensure_ascii=False)[:200]}...")
                
                # 基本構造の確認
                if not isinstance(plan_data, dict):
                    self.logger.warning("解析されたJSONがdict型ではありません。dict型に変換します。")
                    plan_data = {"steps": []}
                
                # stepsキーがあることを確認
                if "steps" not in plan_data:
                    self.logger.warning("解析されたJSONに'steps'キーがありません。空のステップリストを追加します。")
                    # LLMが返した回答にフォーマットミスがある可能性があるため、トップレベルの配列の可能性をチェック
                    if isinstance(plan_data, list) and len(plan_data) > 0 and isinstance(plan_data[0], dict):
                        # トップレベルの配列を見つけた場合、それをstepsとして使用
                        self.logger.info("トップレベルの配列をstepsとして使用します")
                        plan_data = {"steps": plan_data}
                    else:
                        plan_data["steps"] = []
                
                # stepsがリスト型であることを確認
                if not isinstance(plan_data["steps"], list):
                    self.logger.warning("'steps'がリスト型ではありません。リスト型に変換します。")
                    # stepsが文字列の場合は、パース試行
                    if isinstance(plan_data["steps"], str):
                        try:
                            steps_text = plan_data["steps"]
                            # JSON配列に見える場合
                            if steps_text.strip().startswith('[') and steps_text.strip().endswith(']'):
                                try:
                                    plan_data["steps"] = json.loads(steps_text)
                                except:
                                    plan_data["steps"] = []
                            else:
                                # 単一ステップの説明と解釈
                                plan_data["steps"] = [{"description": steps_text, "skill_id": "analyze_data"}]
                        except:
                            plan_data["steps"] = []
                    else:
                        plan_data["steps"] = []
                
                # ここで取得したstepsの内容をログに出力
                self.logger.info(f"steps内容: {json.dumps(plan_data['steps'], ensure_ascii=False)[:200]}...")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSONの解析に失敗しました: {e}")
                self.logger.error(f"解析しようとしたJSON文字列: {json_str[:200]}...")
                plan_data = {"steps": []}
            
            # ステップが空でないことを確認
            if not plan_data["steps"]:
                self.logger.warning("ステップリストが空です。デフォルトステップを追加します。")
                plan_data["steps"] = [
                    {
                        "id": "step1",
                        "description": "データ収集と分析を実行",
                        "skill_id": "analyze_product_data",
                        "input_data": {}
                    }
                ]
            
            # 基本的な検証
            for i, step in enumerate(plan_data["steps"]):
                if not isinstance(step, dict):
                    self.logger.warning(f"ステップ{i+1}がdict型ではありません。修正します。")
                    # 文字列の場合は説明として扱う
                    if isinstance(step, str):
                        plan_data["steps"][i] = {
                            "id": f"step{i+1}",
                            "description": step,
                            "skill_id": "analyze_product_data",
                            "input_data": {}
                        }
                    else:
                        # その他の型の場合はスキップ
                        plan_data["steps"][i] = {
                            "id": f"step{i+1}",
                            "description": f"ステップ {i+1}",
                            "skill_id": "analyze_product_data",
                            "input_data": {}
                        }
            
            self.logger.info(f"実行計画の抽出に成功しました。ステップ数: {len(plan_data['steps'])}")
            return plan_data
            
        except Exception as e:
            self.logger.error(f"実行計画の抽出中にエラーが発生しました: {e}")
            # デフォルトの計画を返す
            return {
                "steps": [
                    {
                        "id": "step1",
                        "description": "データ収集と分析を実行",
                        "skill_id": "analyze_product_data",
                        "input_data": {}
                    }
                ]
            }
    
    def _create_plan_from_data(self, plan_data, parameters, scenario_text=None):
        """
        計画データを実行計画オブジェクトに変換する

        Args:
            plan_data: LLMから生成された計画データ
            parameters: 入力パラメータ
            scenario_text: シナリオテキスト

        Returns:
            ExecutionPlan: 生成された実行計画
        """
        steps = []
        product_id = parameters.get("product_id", "unknown")
        threshold = float(parameters.get("threshold", 5.0))

        # ステップデータを正規化して追加
        for step_data in plan_data.get("steps", []):
            # ステップIDを設定
            if "id" not in step_data or not step_data["id"]:
                step_data["id"] = f"step{len(steps)+1}"

            # スキルIDを正規化
            if "skill_id" not in step_data or not step_data["skill_id"]:
                # skill_nameがあればskill_idとして使用
                if "skill_name" in step_data:
                    step_data["skill_id"] = step_data["skill_name"]
                    self.logger.info(f"skill_nameをskill_idとして使用します: {step_data['skill_id']}")
                else:
                    # デフォルトのスキルIDを設定
                    step_data["skill_id"] = "analyze_product_data"
                    self.logger.warning(f"ステップ {step_data['id']} にスキルIDがありません。デフォルト値を使用します。")
            
            # 現在設定されているスキルIDをログに記録
            self.logger.info(f"ステップ {step_data['id']} のスキルID: {step_data['skill_id']}")

            # input_dataを正規化
            # input_dataがリスト型の場合は辞書型に変換
            if "input_data" not in step_data:
                step_data["input_data"] = {}
                self.logger.info(f"ステップ {step_data['id']} にinput_dataがありません。空の辞書を使用します。")
            elif isinstance(step_data["input_data"], list):
                input_list = step_data["input_data"]
                # LLMが生成したoutput_dataも取得 (なければ空リスト)
                output_list = step_data.get("output_data", [])

                # エージェントへの指示文字列を生成
                instruction_parts = []
                if input_list:
                    instruction_parts.append("以下の情報を取得・分析してください:")
                    instruction_parts.extend([f"- {item}" for item in input_list])

                if output_list:
                    # output_dataが存在する場合のみ期待する出力形式を追加
                    if instruction_parts: # 既に入力指示があれば改行
                        instruction_parts.append("\\n")
                    instruction_parts.append("期待する出力形式は次の通りです:")
                    instruction_parts.extend([f"- {item}" for item in output_list])

                # 最終的な指示文字列を作成
                instruction_str = "\\n".join(instruction_parts)

                input_dict = {"input": instruction_str} # "input"キーを使用
                step_data["input_data"] = input_dict
                # ログメッセージを修正
                self.logger.info(f"リスト型のinput/output_dataから辞書型のinput_dataを生成しました: {input_dict}")
            elif isinstance(step_data["input_data"], dict):
                 # すでに辞書型の場合はそのまま使用する（ログのみ出す）
                 self.logger.info(f"ステップ {step_data['id']} のinput_dataは既に辞書型です: {step_data['input_data']}")
            else:
                 # 想定外の型の場合は空の辞書にする
                 self.logger.warning(f"ステップ {step_data['id']} のinput_dataが予期しない型 ({type(step_data['input_data'])}) です。空の辞書を使用します。")
                 step_data["input_data"] = {}

            # パラメータをinput_dataに追加
            for param_name, param_value in parameters.items():
                # input_dataが辞書であることを保証
                if not isinstance(step_data["input_data"], dict):
                     step_data["input_data"] = {} # 万が一辞書でなければ初期化
                step_data["input_data"][param_name] = param_value

            # PlanStepを作成して追加
            try:
                # PlanStepのデバッグ情報を出力
                self.logger.info(f"PlanStep作成中: {json.dumps(step_data, ensure_ascii=False)}")
                steps.append(PlanStep(**step_data))
                self.logger.info(f"ステップ追加成功: {step_data['id']}, スキルID: {step_data['skill_id']}")
            except Exception as e:
                self.logger.error(f"ステップの作成中にエラーが発生しました: {e}")
                self.logger.error(f"問題のあるステップデータ: {step_data}")
                # エラーを無視してデフォルトのステップを作成
                default_step = {
                    "id": step_data.get("id", f"step{len(steps)+1}"),
                    "description": step_data.get("description", "情報取得ステップ"),
                    "skill_id": "analyze_product_data",
                    "input_data": {"product_id": product_id, "threshold": threshold}
                }
                self.logger.info(f"デフォルトステップを作成します: {default_step}")
                steps.append(PlanStep(**default_step))

        # 少なくとも1つのステップがあることを確認
        if not steps:
            self.logger.warning("計画にステップがありません。デフォルトステップを追加します。")
            default_step = PlanStep(
                id="step1",
                description="製品データの取得と分析",
                skill_id="analyze_product_data",
                input_data={"product_id": product_id, "threshold": threshold}
            )
            steps.append(default_step)

        # 実行計画を作成
        plan_id = str(uuid4())
        execution_plan = ExecutionPlan(
            plan_id=plan_id,
            product_id=product_id,
            threshold=threshold,
            steps=steps,
            scenario_text=scenario_text
        )

        return execution_plan
    
