import logging
from typing import Any, Dict, List
import uuid
import aiosqlite
from ..core.node_base import Node, NodeResult, make_history_entry
from ..core.graph_loader import load_subgraphs
from pathlib import Path
import copy
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ..state import DynamicAgentState
from ..backend.main import update_state_from_subgraph_progress

logger = logging.getLogger(__name__)

class ForkHypothesesNode(Node):
    id = "fork_hypotheses"

    async def run(self, state: Dict[str, Any], toolbox):
        logger.info("--- Node: ForkHypotheses ---")
        patch = {}
        events = []
        # 初期仮説リストを取得
        hypotheses = state.get("current_hypotheses", [])
        if not hypotheses:
            patch = {"fork_error": "No hypotheses to fork."}
            return NodeResult(observation="fork_error", patch=patch, events=events)

        # サブグラフYAMLのパス（共通テンプレート）
        child_graph_path = Path(__file__).parent.parent / "core" / "graphs" / "child_graph.yml"
        subgraph_paths = [str(child_graph_path) for _ in hypotheses]
        subgraphs = load_subgraphs(subgraph_paths)

        forked_states = []
        child_results = []
        current_hypotheses = []
        subgraph_history = []  # 進捗記録用リスト

        # SQLiteコネクションとチェックポインターインスタンスはループの外で生成
        conn = await aiosqlite.connect("checkpoints.sqlite3")
        checkpointer = AsyncSqliteSaver(conn=conn)

        for i, hyp in enumerate(hypotheses[0:1]):
            forked_state = copy.deepcopy(state)
            forked_state["current_hypotheses"] = [hyp]
            forked_state["target_hypothesis_id"] = hyp.get("id")
            nodes, edges, start_node = subgraphs[i]
            # LangGraph StateGraphとしてサブグラフを構築
            workflow = StateGraph(DynamicAgentState)
            for node_id, node_obj in nodes.items():
                async def node_func(state, config, *, _node=node_obj, _toolbox=toolbox):
                    result = await _node.run(state, _toolbox)
                    update_dict = {**result.patch}
                    if result.events:
                        update_dict["history"] = result.events
                    return update_dict
                workflow.add_node(node_id, node_func)
            # エッジ追加
            outgoing = {}
            for e in edges:
                outgoing.setdefault(e.src, []).append(e)
            for src, elist in outgoing.items():
                elist.sort(key=lambda e: e.priority)
                if len(elist) == 1:
                    workflow.add_edge(src, elist[0].dst)
                    continue
                async def _router(state, *, _edges=elist, _toolbox=toolbox):
                    for e in _edges:
                        try:
                            if await e.condition.evaluate(state, None, _toolbox):
                                return e.dst
                        except Exception:
                            continue
                    return END
                target_map = {e.dst: e.dst for e in elist}
                target_map[END] = END
                workflow.add_conditional_edges(src, _router, target_map)
            workflow.add_edge(START, start_node)
            for node_id in nodes:
                if not any(e.src == node_id for e in edges):
                    workflow.add_edge(node_id, END)
            
            # チェックポインターの設定
            # conn = await aiosqlite.connect("checkpoints.sqlite3") # ループの外に移動
            # checkpointer = AsyncSqliteSaver(conn=conn) # ループの外に移動

            subgraph = workflow.compile(checkpointer=checkpointer)
            
            # サブグラフを実行
            final_state = None
            final_node = None
            # サブグラフごとにユニークなthread_idを生成
            parent_thread_id = state.get("thread_id", "main_graph")
            hyp_id = hyp.get('id', "unknown_hyp_id") # idがNoneの場合のフォールバックを設定
            subgraph_thread_id = f"{parent_thread_id}_{hyp_id}_{uuid.uuid4().hex}" # 修正
            config = {"configurable": {"thread_id": subgraph_thread_id}, "recursion_limit": 100}
            async for output in subgraph.astream(forked_state, config):
                node_name = list(output.keys())[0]
                node_state = output[node_name]
                # 進捗をpatchに記録
                # subgraph_progress.append({
                #     "hypothesis_id": hyp.get("id"),
                #     "node": node_name,
                #     "state": node_state,
                # })
                # 親グラフstateにhistoryとcurrent_hypothesesを都度反映
                # print(f"history: {node_state.get('history', [])}")
                # print(f"current_hypotheses: {node_state.get('current_hypotheses', [])}")
                await update_state_from_subgraph_progress({
                    "history": node_state.get("history", []),
                    "current_hypotheses": node_state.get("current_hypotheses", [])
                })
                if node_name == END:
                    break
                final_node = node_name
                final_state = node_state
            child_results.append({
                "hypothesis_id": hyp.get("id"),
                "final_state": final_state,
                "final_node": final_node
            })
            forked_states.append(forked_state)
            current_hypotheses.extend(node_state.get("current_hypotheses", []))
            subgraph_history.extend(node_state.get("history", []))


        # ループ終了後にコネクションをクローズ
        if conn:
            await conn.close()

        events.append(make_history_entry(
            "node",
            {"name": "fork_hypotheses", "forked_hypotheses": [h.get("id") for h in hypotheses]},
            state
        ))
        patch = {
            "current_hypotheses": current_hypotheses,
            "history": subgraph_history,  # 進捗情報
        }
        return NodeResult(observation="forked", patch=patch, events=events) 