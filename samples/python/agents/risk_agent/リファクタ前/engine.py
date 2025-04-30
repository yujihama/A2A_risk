import logging
from typing import Dict, List, Any

from ..core.edge import Edge
from ..core.node_base import Node, NodeResult

logger = logging.getLogger(__name__)


class ToolBox:  # 簡易 DI コンテナ
    def __init__(self, *, llm=None, smart_a2a_client=None, data_analyzer=None):
        self.llm = llm
        self.smart_a2a_client = smart_a2a_client
        # 旧コード互換: data_client エイリアスを持たせる
        self.data_client = smart_a2a_client
        self.data_analyzer = data_analyzer


class Engine:
    def __init__(self, nodes: Dict[str, Node], edges: List[Edge], *, start_node: str):
        self.nodes = nodes
        self.edges = edges
        self.current_node = start_node
        # outgoing 辞書作成
        self.outgoing: Dict[str, List[Edge]] = {}
        for e in edges:
            self.outgoing.setdefault(e.src, []).append(e)
            # priority ソートは後で行う
        for edgelist in self.outgoing.values():
            edgelist.sort(key=lambda e: e.priority)

    async def step(self, state, toolbox) -> bool:  # noqa: ANN001
        """1 ステップ実行。True=継続, False=終了"""
        node = self.nodes[self.current_node]
        logger.info(f"[ENGINE] Node enter: {self.current_node}")
        result: NodeResult = await node.run(state, toolbox)
        # state patch
        logger.info(f"[ENGINE] (before patch) state['next_action']: {state.get('next_action')}")
        logger.info(f"[ENGINE] (patch) result.patch['next_action']: {result.patch.get('next_action') if result.patch else None}")
        state.update(result.patch)
        logger.info(f"[ENGINE] (after patch) state['next_action']: {state.get('next_action')}")
        # history 追加
        if result.events:
            state.setdefault("history", []).extend(result.events)

        # --- チェックポイント保存 ---
        from ..utils import save_checkpoint
        checkpoint_path = save_checkpoint(state, self.current_node)
        logger.info(f"[CHECKPOINT] {self.current_node} のチェックポイントを保存: {checkpoint_path}")

        # エッジ判定
        logger.info(f"[ENGINE] (edge check) state['next_action']: {state.get('next_action')}")
        edges = self.outgoing.get(self.current_node, [])
        for idx, edge in enumerate(edges):
            expr = getattr(edge.condition, '_code', None)
            expr_str = expr.co_consts[0] if expr else str(edge.condition)
            logger.info(f"[ENGINE] (edge {idx}) {edge.src} -> {edge.dst} expr: {expr_str}")
        for edge in edges:
            if await edge.is_true(state, result, toolbox):
                logger.info(f"[ENGINE] Transition {edge.src} -> {edge.dst} by {edge.condition.__class__.__name__}")
                self.current_node = edge.dst
                return True
        logger.warning("[ENGINE] No outgoing edge satisfied, terminating.")
        return False 