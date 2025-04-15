import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio

# 自作コンポーネントをインポート
from samples.python.agents.smart_kakaku_signal.scenario_analyzer import ScenarioAnalyzer
from samples.python.agents.smart_kakaku_signal.plan_generator import DynamicPlanGenerator
from samples.python.agents.smart_kakaku_signal.data_analyzer import DynamicDataAnalyzer
from samples.python.agents.smart_kakaku_signal.agent import execute_step, ExecutionPlan

# ロガーの設定
logger = logging.getLogger(__name__)

class ScenarioExecutionEngine:
    """
    シナリオ実行エンジン
    この中心的なクラスがシナリオ分析、計画生成、実行、結果分析を調整する
    """
    
    def __init__(self, llm_client, registry):
        """
        初期化
        
        Args:
            llm_client: LLMクライアント
            registry: エージェントレジストリ
        """
        self.llm_client = llm_client
        self.registry = registry
        
        # 各コンポーネントの初期化
        self.scenario_analyzer = ScenarioAnalyzer(llm_client)
        self.plan_generator = DynamicPlanGenerator(llm_client, registry)
        self.data_analyzer = DynamicDataAnalyzer(llm_client)
        
        logger.info("シナリオ実行エンジンを初期化しました")
    
    async def execute_scenario(
        self, 
        scenario_text: str, 
        parameters: Dict[str, Any], 
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        シナリオを実行して結果を返す
        
        Args:
            scenario_text: 自然言語のシナリオテキスト
            parameters: 入力パラメータ
            debug: デバッグモード（詳細ログを出力）
            
        Returns:
            Dict: 実行結果
        """
        logger.info(f"=== シナリオ実行エンジンを起動 ===")
        logger.info(f"シナリオ: {scenario_text[:100]}...")
        logger.info(f"パラメータ: {parameters}")
        
        try:
            # 1. シナリオを分析
            logger.info("シナリオを分析中...")
            scenario_analysis = await self.scenario_analyzer.analyze_scenario(scenario_text,parameters)
            
            if debug:
                logger.debug(f"シナリオ分析結果: {json.dumps(scenario_analysis, ensure_ascii=False, indent=2)}")
            
            # 2. 実行計画を生成
            logger.info("実行計画を生成中...")
            execution_plan = await self.plan_generator.generate_execution_plan(
                scenario_analysis, 
                parameters,
                scenario_text
            )
            
            # デバッグログ: 実行計画の属性を表示
            logger.info(f"実行計画生成完了: ID={execution_plan.plan_id}, ステップ数={len(execution_plan.steps)}")
            logger.info(f"実行計画の属性一覧: {dir(execution_plan)}")
            
            # 実行計画のscenario_text属性が存在するか確認
            try:
                scenario_text_attr = getattr(execution_plan, 'scenario_text', None)
                logger.info(f"scenario_text属性: {'存在します (長さ: ' + str(len(scenario_text_attr)) + '文字)' if scenario_text_attr else '存在しません'}")
            except Exception as e:
                logger.error(f"scenario_text属性確認時にエラー: {e}")
            
            # 3. 計画を実行
            logger.info("計画を実行中...")
            step_results = {}
            
            for i, step in enumerate(execution_plan.steps):
                logger.info(f"ステップ {i+1}/{len(execution_plan.steps)}: {step.description} を実行中...")
                
                # 前のステップの結果を入力として使用
                # if i > 0:
                #     prev_steps = execution_plan.steps[:i]
                #     # 前のステップの出力を集約
                #     for prev_step in prev_steps:
                #         if prev_step.is_completed and prev_step.output_data and "result" in prev_step.output_data:
                #             # 前のステップの情報を明示的に名前付きで追加
                #             step.input_data[f"previous_step_{prev_step.id}_result"] = prev_step.output_data.get("result")
                #             step.input_data[f"previous_step_{prev_step.id}_agent"] = prev_step.output_data.get("agent", "不明")
                
                # # 製品IDをパラメータから自動的に追加
                # for param_name, param_value in parameters.items():
                #     if param_name not in step.input_data:
                #         step.input_data[param_name] = param_value
                
                # ステップを実行
                logger.info(f"ステップ {step.id} を実行: スキルID={step.skill_id}, 説明={step.description}")
                logger.info(f"ステップ入力データ: {json.dumps(step.input_data, ensure_ascii=False)}")
                logger.info(f"エージェントへのタスク送信内容: {json.dumps(step.input_data, ensure_ascii=False, indent=2)}")
                updated_step = await execute_step(step)
                execution_plan.steps[i] = updated_step
                
                if updated_step.is_completed and updated_step.output_data:
                    step_results[step.id] = {
                        "description": step.description,
                        "result": updated_step.output_data.get("result", ""),
                        "agent": updated_step.output_data.get("agent", "不明")
                    }
                    logger.info(f"ステップ {step.id} が完了: {step.description}")
                    logger.info(f"使用されたエージェント: {updated_step.output_data.get('agent', '不明')}")
                    logger.info(f"結果サマリー: {updated_step.output_data.get('result', '')[:100]}...")
                else:
                    logger.warning(f"ステップ {step.id} の実行に失敗: {updated_step.error or '不明なエラー'}")
            
            # 結果がない場合のチェック
            if not step_results:
                logger.error("実行結果が取得できませんでした")
                return {
                    "scenario_text": scenario_text,
                    "parameters": parameters,
                    "is_anomaly": None,
                    "is_data_sufficient": False,
                    "analysis": "実行ステップから結果を取得できませんでした",
                    "details": {}
                }
            
            # 4. 収集したデータを分析
            logger.info("収集したデータを分析中...")
            analysis_result = await self.data_analyzer.analyze_collected_data(
                scenario_text,
                scenario_analysis,
                step_results,
                parameters
            )
            
            # 5. 結果をまとめて返す
            result = {
                "scenario_text": scenario_text,
                "parameters": parameters,
                "is_anomaly": analysis_result.get("is_anomaly"),
                "is_data_sufficient": analysis_result.get("is_data_sufficient", False),
                "analysis": analysis_result.get("analysis", ""),
                "details": analysis_result,
                "step_results": step_results
            }
            
            # 詳細結果をログに出力
            is_anomaly = analysis_result.get("is_anomaly")
            is_sufficient = analysis_result.get("is_data_sufficient")
            
            if is_anomaly is None:
                logger.info("分析結果: データ不足のため判断保留")
            elif is_anomaly:
                logger.info("分析結果: 異常を検出")
            else:
                logger.info("分析結果: 異常なし")
                
            logger.info(f"分析: {analysis_result.get('analysis', '')[:200]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"シナリオ実行中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "scenario_text": scenario_text,
                "parameters": parameters,
                "is_anomaly": None,
                "is_data_sufficient": False,
                "analysis": f"シナリオ実行中にエラーが発生しました: {str(e)}",
                "details": {},
                "error": str(e)
            } 