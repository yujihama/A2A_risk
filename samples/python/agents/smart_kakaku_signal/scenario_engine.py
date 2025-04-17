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
        logger.info(f"シナリオ: {scenario_text[:100]}...")
        logger.info(f"パラメータ: {parameters}")
        
        try:
            # 1. シナリオを分析
            scenario_analysis = await self.scenario_analyzer.analyze_scenario(scenario_text,parameters)
            # 2. 実行計画を生成
            execution_plan = await self.plan_generator.generate_execution_plan(
                scenario_analysis, 
                parameters,
                scenario_text
            )
            logger.info(f"実行計画生成完了: ID={execution_plan.plan_id}, ステップ数={len(execution_plan.steps)}")
            # 3. LangGraphグラフで計画を実行
            logger.info("LangGraphグラフで計画を実行します...")
            from samples.python.agents.smart_kakaku_signal.plan_to_langgraph import build_langgraph_from_plan
            from samples.python.common.client.smart_client import SmartA2AClient
            agent_executor = SmartA2AClient(self.registry)
            graph = build_langgraph_from_plan(
                execution_plan,
                agent_executor,
                llm_client=self.llm_client,
                scenario_text=scenario_text,
                scenario_analysis=scenario_analysis,
                parameters=parameters,
                data_analyzer=self.data_analyzer
            )
            step_results = await graph.ainvoke({})
            logger.info(f"LangGraphグラフ実行結果: {step_results}")
            return step_results
        except Exception as e:
            logger.error(f"シナリオ実行中にエラーが発生しました: {e}")
            raise e
            
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