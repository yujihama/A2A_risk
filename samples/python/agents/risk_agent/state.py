from typing import TypedDict, List, Dict, Any, Optional, Literal
from operator import add
from typing_extensions import Annotated

class Hypothesis(TypedDict):
    id: str
    text: str
    priority: float
    status: Literal["new", "investigating", "supported", "rejected", "needs_revision", "inconclusive"]
    evaluation_reasoning: Optional[str]
    required_next_data: Optional[str]
    complexity_score: Optional[float]
    verification: Optional[Dict]
    parent_hypothesis_id: Optional[str]

class NextAction(TypedDict):
    action_type: Literal[
        "query_data_agent",
        "generate_hypothesis",
        "evaluate_hypothesis",
        "refine_plan",
        "execute_plan_step",
        "plan_verification",
        "execute_verification_step",
        "summarize_verification",
        "conclude",
        "error",
        "refine_hypothesis",
        "triangulate_data",
        "escalate_to_expert",
        "sample_and_test"
    ]
    parameters: Dict[str, Any]

class HistoryEntry(TypedDict):
    type: Literal["thought", "action", "observation"]
    content: Any
    timestamp: str

class DynamicAgentState(TypedDict):
    objective: str
    history: Annotated[List[HistoryEntry], add]
    current_hypotheses: List[Hypothesis]
    collected_data_summary: Dict[str, Any]
    active_plan: Optional[Dict]
    next_action: Optional[NextAction]
    final_result: Optional[Dict]
    available_actions: List[Dict]
    available_data_agents_and_skills: List[Dict]
    error_message: Optional[str]
    max_iterations: int
    current_iteration: int
    data_points_collected: int
    cycles_since_last_hypothesis: int
    max_queries_without_hypothesis: int
    consecutive_query_count: int
    currently_investigating_hypothesis_id: Optional[str]
    eval_repeat_count: int
    verification_repeat_count: int
    current_verification_steps: Optional[List[Dict]]
    verification_results: Optional[Dict]
    eda_summary: Optional[str]
    eda_stats: Optional[Dict]
    analysis_result: Dict[str, Any]