from langgraph.graph import StateGraph, END
from functools import partial
from A2A_risk.samples.python.agents.risk_agent.nodes import eda_node, decision_maker_node, action_executor_node, query_data_agent_node, plan_verification_node, execute_verification_step_node, summarize_verification_node
from A2A_risk.samples.python.agents.risk_agent.routing import route_action, route_after_execution, route_after_query
from A2A_risk.samples.python.agents.risk_agent.state import DynamicAgentState
import logging

def create_dynamic_agent_graph(llm, smart_a2a_client, data_analyzer, max_iterations=50) -> StateGraph:
    logger = logging.getLogger(__name__)
    workflow = StateGraph(DynamicAgentState)

    # ノード追加
    workflow.add_node("eda", partial(eda_node, llm=llm))
    workflow.add_node("decision_maker", partial(decision_maker_node, llm=llm))
    workflow.add_node("action_executor", partial(action_executor_node, llm=llm, smart_a2a_client=smart_a2a_client, data_analyzer=data_analyzer))
    workflow.add_node("query_data_agent_node", partial(query_data_agent_node, smart_a2a_client=smart_a2a_client, llm=llm))
    workflow.add_node("plan_verification", partial(plan_verification_node, llm=llm))
    workflow.add_node("execute_verification_step", partial(execute_verification_step_node, smart_a2a_client=smart_a2a_client, llm=llm))
    workflow.add_node("summarize_verification", partial(summarize_verification_node, llm=llm))

    # エントリポイント
    workflow.set_entry_point("eda")
    workflow.add_edge("eda", "decision_maker")

    workflow.add_conditional_edges(
        "decision_maker",
        route_action,
        {
            "action_executor": "action_executor",
            "query_data_agent_node": "query_data_agent_node",
            "plan_verification": "plan_verification",
            "execute_verification_step": "execute_verification_step",
            "summarize_verification": "summarize_verification",
            "eda": "eda",
            "__end__": END
        }
    )
    for node_name in ["action_executor", "plan_verification", "execute_verification_step", "summarize_verification"]:
        workflow.add_conditional_edges(
            node_name,
            route_after_execution,
            {
                "decision_maker": "decision_maker",
                "plan_verification": "plan_verification",
                "execute_verification_step": "execute_verification_step",
                "summarize_verification": "summarize_verification",
                "__end__": END
            }
        )
    for node_name in ["query_data_agent_node",]:
        workflow.add_conditional_edges(
            node_name,
            route_after_query,
            {
                "decision_maker": "decision_maker",
                "plan_verification": "plan_verification",
                "execute_verification_step": "execute_verification_step",
                "summarize_verification": "summarize_verification",
                "__end__": END
            }
        )
    graph = workflow.compile()
    logger.info("Dynamic agent graph compiled successfully.")
    return graph 