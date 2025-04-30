import logging
from typing import Literal, Union
from A2A_risk.samples.python.agents.risk_agent.state import DynamicAgentState

logger = logging.getLogger(__name__)

def route_action(state: DynamicAgentState) -> Union[str, Literal["__end__"]]:
    logger.info("--- Routing Action ---")
    if state.get('error_message'):
        logger.error(f"Error detected in state: {state['error_message']}. Ending workflow.")
        return "__end__"

    next_action = state.get('next_action')
    if next_action:
        action_type = next_action.get('action_type')
        log_msg = f"Next action type decided: {action_type}"
        if state.get('currently_investigating_hypothesis_id'):
             log_msg += f" (Focus: {state.get('currently_investigating_hypothesis_id')})"
        logger.info(log_msg)
        if action_type == "error":
             logger.warning("Routing to __end__ based on 'error' action decided by Decision Maker.")
             return "__end__"
        elif action_type == "query_data_agent":
            logger.info("Routing to query_data_agent_node.")
            return "query_data_agent_node"
        elif action_type == "plan_verification":
            return "plan_verification"
        elif action_type == "execute_verification_step":
            return "execute_verification_step"
        elif action_type == "summarize_verification":
            return "summarize_verification"
        else:
            logger.info("Routing to action_executor.")
            return "action_executor"
    else:
        logger.warning("No next_action determined by Decision Maker node or forced action. Ending workflow unexpectedly.")
        state['error_message'] = "Decision Maker failed to produce a valid next action."
        return "__end__"

def route_after_query(state: DynamicAgentState) -> Union[str, Literal["__end__"]]:
    """query_data_agent_node の実行後に呼び出されるルーティング関数"""
    logger.info("--- Routing After Query ---")
    if state.get('error_message'):
        logger.error(f"Error detected after query: {state['error_message']}. Ending workflow.")
        return "__end__"
    if state.get('current_iteration', 0) >= state.get('max_iterations', 10):
         logger.warning(f"Max iterations reached after query. Ending workflow.")
         return "__end__"

    # If verification nodes are pending, jump accordingly
    if state.get('next_action'):
        na_type = state['next_action'].get('action_type')
        if na_type in ["plan_verification", "execute_verification_step", "summarize_verification"]:
            return na_type

    log_msg = "Continuing to decision_maker after query."
    if state.get('currently_investigating_hypothesis_id'):
        log_msg += f" (Focus: {state.get('currently_investigating_hypothesis_id')})"
    logger.info(log_msg)
    return "decision_maker"

def route_after_execution(state: DynamicAgentState) -> Union[str, Literal["__end__"]]:
    logger.info("--- Routing After Execution ---")
    if state.get('error_message'):
        logger.error(f"Error detected after execution: {state['error_message']}. Ending workflow.")
        return "__end__"
    if state.get('final_result') is not None:
         logger.info("Conclusion reached. Ending workflow.")
         return "__end__"
    if state.get('current_iteration', 0) >= state.get('max_iterations', 10):
         logger.warning(f"Max iterations reached after execution. Ending workflow.")
         return "__end__"

    # Handle verification flow
    if state.get('next_action'):
        na_type = state['next_action'].get('action_type')
        if na_type in ["plan_verification", "execute_verification_step", "summarize_verification"]:
            return na_type

    log_msg = "Continuing to decision_maker."
    if state.get('currently_investigating_hypothesis_id'):
        log_msg += f" (Focus: {state.get('currently_investigating_hypothesis_id')})"
    logger.info(log_msg)
    return "decision_maker" 