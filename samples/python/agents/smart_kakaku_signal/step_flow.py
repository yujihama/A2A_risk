from typing import List, Callable, Optional
from samples.python.agents.smart_kakaku_signal.agent import PlanStep, ExecutionPlan

class StepNode:
    def __init__(self, step: PlanStep):
        self.step = step
        self.next_nodes: List['StepNode'] = []
        self.condition_funcs: List[Optional[Callable[[dict], bool]]] = []  # 各エッジの遷移条件

    def add_next(self, node: 'StepNode', condition_func: Optional[Callable[[dict], bool]] = None):
        self.next_nodes.append(node)
        self.condition_funcs.append(condition_func)

    def get_next(self, skill_result: dict) -> Optional['StepNode']:
        """
        skill_resultに応じて次ノードを決定
        条件関数がNoneの場合は常に遷移
        """
        for node, cond in zip(self.next_nodes, self.condition_funcs):
            if cond is None or cond(skill_result):
                return node
        return None

class StepFlow:
    def __init__(self, nodes: List[StepNode]):
        self.nodes = nodes
        self.start_node = nodes[0] if nodes else None

    def invoke(self, agent_executor):
        """
        フロー全体を実行する標準インターフェース
        agent_executor: execute_skill(skill_id, input_data) を持つクライアント
        """
        return self.traverse(agent_executor)

    def traverse(self, agent_executor):
        """
        ノードをたどりながらスキル実行を行う（invokeから呼び出し）
        """
        current = self.start_node
        results = []
        while current:
            step = current.step
            print(f"実行: {step.id} (スキルID: {step.skill_id})")
            # スキル実行
            try:
                result = agent_executor.execute_skill(step.skill_id, step.input_data)
                print(f"  スキル実行結果: {result}")
            except Exception as e:
                print(f"  スキル実行エラー: {e}")
                result = None
            results.append({"step_id": step.id, "result": result})
            # 次ノードへ
            current = current.get_next(result if result is not None else {})
        return results

def build_flow_from_plan(plan: ExecutionPlan) -> StepFlow:
    """
    ExecutionPlanから直列フロー（StepFlow）を構築
    """
    nodes = [StepNode(step) for step in plan.steps]
    for i in range(len(nodes) - 1):
        nodes[i].add_next(nodes[i+1])
    return StepFlow(nodes) 