import asyncio
import logging
import json
import yaml
import os
from A2A_risk.samples.python.agents.risk_agent.graph_builder import create_dynamic_agent_graph
from A2A_risk.samples.python.agents.risk_agent.state import DynamicAgentState
from A2A_risk.samples.python.agents.risk_agent.config import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from A2A_risk.samples.python.agents.risk_agent.agent import initialize_registry, registry
# 必要に応じて LLM, DummyClient, DummyAnalyzer なども import/定義

class DummyClient:
    async def find_and_send_task(self, skill_id, message):
        logger = logging.getLogger(__name__)
        logger.info(f"[DummyClient] find_and_send_task called: skill={skill_id}, message={message}")
        await asyncio.sleep(0.5)
        return {"result": {"id": f"task_{skill_id}"}}
    async def get_task(self, task_id):
        logger = logging.getLogger(__name__)
        logger.info(f"[DummyClient] get_task called: task_id={task_id}")
        await asyncio.sleep(1)
        return {"status": "COMPLETED", "result": f"Dummy result for {task_id}. Found 5 relevant entries."}

class DummyAnalyzer:
    async def analyze_collected_data(self, objective, history, hypotheses, data_summary):
        logger = logging.getLogger(__name__)
        logger.info("[DummyAnalyzer] analyze_collected_data called")
        await asyncio.sleep(0.5)
        return {
            "is_anomaly": False,
            "is_data_sufficient": True,
            "analysis": "Final analysis report (dummy).",
            "summary_of_findings": [h for h in hypotheses if h['status'] in ['supported', 'rejected']],
            "recommendations": "Continue monitoring."
        }

class LLM:
    def __init__(self, model="gpt-4.1-mini", temperature=0):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature).with_structured_output(method="json_mode")


    async def ainvoke(self, prompt):
        return await self.llm.ainvoke(prompt)

def main():
    # ロギング設定
    pass

async def async_main():
    # ロギング設定
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('A2A_risk/samples/python/agents/risk_agent/test_react.log', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config_path = os.path.join(os.path.dirname(__file__), 'agent_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    await initialize_registry(config)
    agents = registry.list_agents()
    # agentsをディクショナリに変換
    agents_list = [{"agent_name": agent.name, "skill_id": agent.skills[0].id, "skill_name": agent.skills[0].name, "skill_description": agent.skills[0].description} for agent in agents]

    test_llm = LLM()
    test_smart_a2a_client = DummyClient()
    test_data_analyzer = DummyAnalyzer()

    available_actions = [
        {'type': 'query_data_agent', 'description': 'Query a data agent for specific information related to hypotheses.'},
        {'type': 'generate_hypothesis', 'description': 'Generate new hypotheses based on current data and objective.'},
        {'type': 'evaluate_hypothesis', 'description': 'Evaluate a specific hypothesis using available data.'},
        {'type': 'conclude', 'description': 'Conclude the investigation and generate final report.'},
        {'type': 'error', 'description': 'Handle an unexpected error state.'},
        {'type': 'refine_hypothesis', 'description': '仮説を分割・再定義する。'},
        {'type': 'plan_verification', 'description': 'supported 仮説に対し裏付けプランを立案する'},
        {'type': 'execute_verification_step', 'description': 'verification プランの個々のステップを実行する'},
        {'type': 'summarize_verification', 'description': 'verification 実行結果を総合評価する'},
    ]
    # available_data_agents_and_skills = [
    #     {'agent_name': 'PurchaseAgent', 'skill_id': 'analyze_order', 'description': '発注データを分析します。'},
    #     {'agent_name': 'EmployeeAgent', 'skill_id': 'analyze_employees', 'description': '従業員情報を分析します。'},
    #     {'agent_name': 'RequestAgent', 'skill_id': 'analyze_requests', 'description': '稟議書・申請書情報を分析します。'},
    # ]

    graph = create_dynamic_agent_graph(
        llm=test_llm,
        smart_a2a_client=test_smart_a2a_client,
        data_analyzer=test_data_analyzer,
        max_iterations=50
    )

    initial_state = DynamicAgentState(
        objective="特定の従業員と取引先の間で、単価が過去平均より著しく高い購買がないか検知する",
        history=[],
        current_hypotheses=[],
        collected_data_summary={},
        active_plan=None,
        next_action=None,
        final_result=None,
        available_actions=available_actions,
        available_data_agents_and_skills=agents_list,
        error_message=None,
        max_iterations=50,
        current_iteration=0,
        data_points_collected=0,
        cycles_since_last_hypothesis=0,
        max_queries_without_hypothesis=2,
        consecutive_query_count=0,
        currently_investigating_hypothesis_id=None,
        eval_repeat_count=0,
        verification_repeat_count=0,
        current_verification_steps=None,
        verification_results=None,
        eda_summary=None,
        eda_stats={}
    )

    logging.info("Starting graph execution...")
    final_state = None
    config = {"recursion_limit": 100}
    async for output in graph.astream(initial_state, config=config):
        node_name = list(output.keys())[0]
        node_output = output[node_name]
        logging.info(f"--- Output from node: {node_name} ---")
        if "__end__" in output:
             final_state = output["__end__"]
             logging.info("--- Graph ended ---")
             break
        else:
             final_state = node_output

    logging.info("--- Graph execution finished ---")

    if final_state:
        print("\n--- Final State ---")
        print(f"Objective: {final_state.get('objective')}")
        print(f"Iterations: {final_state.get('current_iteration')}")
        print(f"Error Message: {final_state.get('error_message')}")
        print(f"Final Result: {json.dumps(final_state.get('final_result'), ensure_ascii=False, indent=2)}")
        print(f"\nFinal Hypotheses: {json.dumps(final_state.get('current_hypotheses'), ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(async_main()) 