import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv
import argparse
import yaml
import sys
# import sqlite3 # コメントアウトまたは削除
import aiosqlite # 追加
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver  # fallback
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from importlib import import_module
import uuid
from collections import defaultdict

from .core.graph_loader import load_graph
from .リファクタ前.engine import Engine, ToolBox
from .core.llm_provider import OpenAILLMWrapper
from .agent import initialize_registry, registry
from .state import DynamicAgentState

logger = logging.getLogger(__name__)

# load .env from current dir or any parent
load_dotenv()  # default search
for parent in Path(__file__).resolve().parents:
    env_path = parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        break


# def build_engine(graph_path: Optional[str] = None) -> Engine:  # noqa: ANN001
#     if graph_path is None:
#         base_dir = Path(__file__).parent / "core" / "graphs"
#         if os.getenv("USE_OPENAI", "0") == "1":
#             candidate = base_dir / "llm_graph.yml"
#             graph_path = candidate.as_posix() if candidate.exists() else None

#         if not graph_path:
#             graph_path = (base_dir / "simple_graph.yml").as_posix()
#     nodes, edges, start = load_graph(graph_path)
#     return Engine(nodes, edges, start_node=start)


# Fallback Dummy client if SmartA2A client unavailable


class DummyClient:  # noqa: D101
    async def run_query(self, skill_id, query):  # noqa: D401, ANN001
        logger.info("[DummyClient] run_query skill=%s query=%s", skill_id, query)
        # return dummy list of dicts
        return [{"dummy": True, "value": 42}]


# チェックポイント管理用 (永続化)
# conn = sqlite3.connect("checkpoints.sqlite3", check_same_thread=False)
# checkpointer = AsyncSqliteSaver(conn=conn)


def build_langgraph_from_yaml(yaml_path, toolbox, checkpointer):
    nodes, edges, start_node = load_graph(yaml_path)
    # ノードラッパー関数を生成
    node_funcs = {}
    for node_id, node_obj in nodes.items():
        async def node_func(state, config, *, _node=node_obj, _toolbox=toolbox):
            result = await _node.run(state, _toolbox)
            # LangGraph では戻り値は "state への差分" のみを返すのが推奨。
            # history は Annotated[List, add] でリデューサに add が設定されているため、
            # ここでは events をそのまま差分として返すだけで自動的に追記される。
            update_dict = {**result.patch}
            if result.events:
                update_dict["history"] = result.events
            return update_dict
        node_funcs[node_id] = node_func

    workflow = StateGraph(DynamicAgentState)
    for node_id, func in node_funcs.items():
        workflow.add_node(node_id, func)
    # --- 条件付きエッジへ変換 ---
    # 同一 src をまとめ、1 本だけなら無条件、複数なら router 関数を作成
    outgoing: dict[str, list] = defaultdict(list)
    for e in edges:
        outgoing[e.src].append(e)

    for src, elist in outgoing.items():
        # 優先度でソート (小さいほど高優先度)
        elist.sort(key=lambda e: e.priority)

        # 1 本だけなら無条件エッジ
        if len(elist) == 1:
            workflow.add_edge(src, elist[0].dst)
            continue

        # 複数ある場合は add_conditional_edges を使用
        async def _router(state, *, _edges=elist, _toolbox=toolbox):
            """state を評価して遷移先ノードIDを返す"""
            for e in _edges:
                try:
                    if await e.condition.evaluate(state, None, _toolbox):
                        return e.dst
                except Exception:
                    # 条件評価失敗は False とみなす
                    continue
            # どれもマッチしなければ END
            return END

        # target_map は {key: dst} 形式。ここでは dst 名を key としてそのまま使う
        target_map = {e.dst: e.dst for e in elist}
        target_map[END] = END  # フォールバック
        workflow.add_conditional_edges(src, _router, target_map)
    workflow.add_edge(START, start_node)
    # 終端ノード（conclude, error など）
    for node_id in nodes:
        if not any(e.src == node_id for e in edges):
            workflow.add_edge(node_id, END)
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


# --- ヘルパー関数を追加 ---
async def get_thread_id_from_checkpoint(conn: aiosqlite.Connection, checkpoint_id: str) -> Optional[str]:
    """checkpoint_id から thread_id を取得する"""
    try:
        # テーブル名が 'checkpoints' であると仮定
        async with conn.execute(
            "SELECT thread_id FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            else:
                logger.error(f"チェックポイントID '{checkpoint_id}' が見つかりません。")
                return None
    except aiosqlite.Error as e:
        # テーブルが存在しない場合なども考慮 (OperationalError)
        logger.error(f"チェックポイントDBからの thread_id 取得中にエラー: {e}")
        return None
# --- ここまで追加 ---

async def run_with_checkpoint(yaml_path, resume_checkpoint_id=None, initial_state=None):
    # --- ここで checkpointer を初期化 ---
    conn = await aiosqlite.connect("checkpoints.sqlite3") # 変更後: aiosqlite を使う
    checkpointer = AsyncSqliteSaver(conn=conn)
    # ------------------------------------

    llm = OpenAILLMWrapper() if os.getenv("USE_OPENAI", "0") == "1" else None
    data_client = None
    try:
        from A2A_risk.samples.python.common.client.client import SmartA2AClient
        data_client = SmartA2AClient()
    except Exception:
        # from .engine_runner import DummyClient # 同じモジュール内なので直接参照できる
        data_client = DummyClient()
    toolbox = ToolBox(llm=llm, smart_a2a_client=data_client)

    graph = build_langgraph_from_yaml(yaml_path, toolbox, checkpointer)
    # --- thread_id の決定と実行 ---
    thread_id = None # スコープを広げる
    try: # finally で conn.close() するために try ブロックを追加
        if resume_checkpoint_id:
            # --- thread_id を checkpoint_id から取得 ---
            thread_id = await get_thread_id_from_checkpoint(conn, resume_checkpoint_id)
            if not thread_id:
                print(f"エラー: チェックポイント {resume_checkpoint_id} に対応する thread_id が見つかりません。")
                # await conn.close() # finally で閉じる
                return # 実行中断

            # --- resume_config の修正 ---
            resume_config = {"configurable": {
                "thread_id": thread_id,
                "checkpoint_id": resume_checkpoint_id
            }}
            print(f"[再開] thread_id: {thread_id} (指定した checkpoint_id: {resume_checkpoint_id} に対応)")
            # state_patch = initial_state or {} # 再開時は initial_state を渡さない
            await graph.ainvoke(None, resume_config) # 第1引数を None に変更
        else:
            # 新規実行時は新しい UUID を生成
            thread_id = str(uuid.uuid4())
            print(f"[新規実行] thread_id: {thread_id}")
            config = {"configurable": {"thread_id": thread_id},"recursion_limit": 50}
            await graph.ainvoke(initial_state or {}, config) # initial_state が None の場合 {} を使う

        # --- 履歴取得時の config ---
        # 実行に使った thread_id を含む config を用意する必要がある
        if not thread_id:
             # このケースは再開失敗時など
             print("ERROR: thread_id が不明なため履歴を取得できません。")
             # await conn.close() # finally で閉じる
             return

        history_config = {"configurable": {"thread_id": thread_id}}

        # --- 修正後の履歴取得 ---
        print("--- Checkpoint History ---")
        history = []
        i = 0
        async for snapshot in graph.aget_state_history(history_config): # thread_id を含む config を使用
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id", "unknown")
            # thread_id が snapshot.config に含まれているか確認 (通常は history_config で指定したものと同じはず)
            current_thread_id = snapshot.config.get("configurable", {}).get("thread_id", "unknown")
            print(f"{i}: thread_id={current_thread_id}, checkpoint_id={checkpoint_id}, node={snapshot.next}")
            logging.info(f"[CHECKPOINT] {i}: thread_id={current_thread_id}, checkpoint_id={checkpoint_id}, node={snapshot.next}")
            history.append(snapshot)
            i += 1
        print("--- 各ノード完了時のcheckpoint_idを上記に出力しました ---")

    finally:
        # --- DB接続を閉じる ---
        if conn:
            await conn.close()
            print("チェックポイントDB接続を閉じました。")

    # return は不要 (関数の最後に到達するため)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RiskAgent YAML graph")
    parser.add_argument("graph", nargs="?", help="Path to graph YAML", default=None)
    parser.add_argument("--objective", help="Objective / risk scenario text", default=None)
    parser.add_argument("--checkpoint_id", help="再開するチェックポイントID", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    async def main():
        yaml_path = args.graph
        if yaml_path is None:
            base_dir = Path(__file__).parent / "core" / "graphs"
            if os.getenv("USE_OPENAI", "0") == "1":
                candidate = base_dir / "llm_graph.yml"
                yaml_path = candidate.as_posix() if candidate.exists() else None
            if not yaml_path:
                yaml_path = (base_dir / "simple_graph.yml").as_posix()

        # エージェント初期化とavailable_data_agents_and_skillsのセット
        initial_state = {}
        if args.objective:
            initial_state["objective"] = args.objective
        cfg_path = Path(__file__).parent / "agent_config.yaml"
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                await initialize_registry(cfg)
                agents = registry.list_agents()
                agents_list = []
                for agent in agents:
                    if hasattr(agent, "skills") and agent.skills:
                        for sk in agent.skills:
                            agents_list.append({
                                "agent_name": agent.name,
                                "skill_id": sk.id,
                                "skill_name": sk.name,
                                "skill_description": sk.description,
                            })
                    else:
                        agents_list.append({
                            "agent_name": agent.name,
                            "skill_id": None,
                            "skill_name": None,
                            "skill_description": getattr(agent, "description", None),
                        })
                initial_state["available_data_agents_and_skills"] = agents_list
            except Exception as exc:
                logging.warning("Failed to load agent_config.yaml or initialize registry: %s", exc)

        await run_with_checkpoint(
            yaml_path=yaml_path,
            resume_checkpoint_id=args.checkpoint_id,
            initial_state=initial_state
        )

    asyncio.run(main()) 