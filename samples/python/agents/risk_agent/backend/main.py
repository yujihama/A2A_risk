from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import sys # sys をインポート
from typing import Dict, Any # Dict, Any をインポート
import httpx

# ルートロガーではなく、このモジュール用のロガーを取得
logger = logging.getLogger(__name__) # logger = logging.getLogger("backend.main") でも可
logger.setLevel(logging.INFO) # このロガーのレベルを DEBUG に設定

# ハンドラが既に追加されていないか確認 (uvicornが追加する場合があるため)
if not logger.handlers:
    # StreamHandler を作成して標準エラー出力にログを出す
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 他のライブラリのログレベル設定 (これはこのままで良い)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.INFO)

app = FastAPI(title="Risk Agent Backend") # アプリにタイトルを追加

# CORS設定 (開発用にフロントエンドのオリジンを許可)
# 本番環境ではより厳密な設定が必要です
origins = [
    "http://localhost:3000", # Next.js 開発サーバー
    # 必要に応じて他のオリジンも追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connections: set[WebSocket] = set()
latest_state: dict = {} # 最新の状態を保持
state_lock = asyncio.Lock()  # state更新用ロック

# ★ 新しいエンドポイントを追加
@app.post("/state/update", summary="Receive state updates from the engine")
async def update_state(
    new_engine_state: Dict[str, Any] = Body(...),
    reset: bool = False
):
    """LangGraphエンジンから状態更新を受け取り、WebSocketクライアントにブロードキャストする。
    親グラフの状態更新時に、既存のサブグラフ進捗 (current_hypotheses) をマージする。
    """
    global latest_state, state_lock
    logger.info(f"Received full state update via POST. Keys: {list(new_engine_state.keys())}, reset={reset}")

    async with state_lock:
        if reset:
            latest_state.clear()
            await broadcast({})
            return {"message": "State reset and broadcasted"}
        merged_state_for_broadcast = new_engine_state.copy()

        # current_hypotheses をマージ
        # 親エンジンからの hypotheses をベースに、既存の latest_state (サブグラフ進捗含む) の hypotheses をマージ
        engine_hypotheses = new_engine_state.get("current_hypotheses", [])
        # broadcast が呼ばれる前の latest_state から取得
        preserved_hypotheses_from_subgraph = latest_state.get("current_hypotheses", [])

        # merge_hypotheses を使ってマージ
        merged_hypotheses = merge_hypotheses(engine_hypotheses, preserved_hypotheses_from_subgraph)
        merged_state_for_broadcast["current_hypotheses"] = merged_hypotheses

        # History: 既存のhistoryに、親エンジンからのhistoryエントリのうち、まだ含まれていないものを追記
        # broadcast が呼ばれる前の latest_state から既存の history を取得
        existing_history = latest_state.get("history", []).copy() # 変更するためコピー
        new_engine_history_entries = new_engine_state.get("history", [])

        if new_engine_history_entries: # 親エンジンからのhistoryがあれば
            for entry in new_engine_history_entries:
                if entry not in existing_history: # まだ既存の履歴に含まれていなければ
                    existing_history.append(entry) # 既存のhistoryに追加
        
        merged_state_for_broadcast["history"] = existing_history
        # その他のキーは new_engine_state のものが優先される (merged_state_for_broadcast の初期値がコピーなため)

    # マージされた state で broadcast を呼び出す
    # broadcast 関数は内部で latest_state をこの merged_state_for_broadcast で更新する
    await broadcast(merged_state_for_broadcast)
    return {"message": "State update received, merged, and broadcasted"}

@app.get("/state/initial", summary="Get the initial state for the frontend")
async def get_state():
    """フロントエンドが初回ロード時に呼び出すエンドポイント。
    現在の最新のLangGraph状態を返します。
    """
    global latest_state
    logger.info(f"Serving initial state. State keys: {list(latest_state.keys())}")
    return latest_state

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続を処理し、状態更新をリアルタイムにプッシュします。"""
    await websocket.accept()
    connections.add(websocket)
    client_host = websocket.client.host if websocket.client else "Unknown"
    client_port = websocket.client.port if websocket.client else "N/A"
    connection_id = f"{client_host}:{client_port}" # ログ用ID
    logger.info(f"WebSocket connection established from {connection_id}")
    logger.debug(f"Current connections: {len(connections)}")
    # 接続時に最新の状態を送信
    try:
        logger.debug(f"Sending initial state to {connection_id}...")
        if latest_state:
            await websocket.send_json(latest_state)
        else:
            await websocket.send_json({}) # 空の状態を送る
        logger.debug(f"Initial state sent to {connection_id}.")
        # クライアントからのメッセージを待機し、切断を検知
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received message from {connection_id}: {data}") # Keep-aliveなどの確認用
    except WebSocketDisconnect as e:
        # ★ 切断理由もログに出力
        logger.info(f"WebSocket connection closed from {connection_id}. Code: {e.code}, Reason: {e.reason}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}", exc_info=True)
    finally:
        # 接続が閉じたかエラーが発生したらセットから削除
        logger.debug(f"Removing connection {connection_id} from active set.")
        if websocket in connections:
            connections.remove(websocket)
        logger.debug(f"Connection {connection_id} removed. Current connections: {len(connections)}")


async def broadcast(state=None):
    global latest_state
    if state is not None:
        latest_state = state  # 親グラフの全stateで上書き
    # state_jsonは常にlatest_state
    try:
        print(f"latest_state_history_len(broadcast): {len(latest_state.get('history', []))}")
        hist = latest_state.get('history', [])
        if hist:
            print(f"latest_state_history(broadcast): {hist[-1]}")
        else:
            print("latest_state_history(broadcast): <empty>")
        state_json = json.dumps(latest_state, default=str)
    except Exception as e:
        logger.error(f"Failed to serialize state for broadcast: {e}", exc_info=True)
        return
    if not connections:
        logger.debug("No WebSocket clients connected, skipping broadcast.")
        return
    logger.info(f"Broadcasting state update to {len(connections)} clients.")
    active_connections = list(connections)
    results = await asyncio.gather(
        *[ws.send_text(state_json) for ws in active_connections],
        return_exceptions=True
    )
    failed_connections_info = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            ws = active_connections[i]
            client_host = ws.client.host if ws.client else "Unknown"
            client_port = ws.client.port if ws.client else "N/A"
            failed_connections_info.append(f"{client_host}:{client_port} ({type(result).__name__}: {result})")
    if failed_connections_info:
        logger.warning(f"Failed to send broadcast to some clients: {', '.join(failed_connections_info)}")

def merge_hypotheses(existing, new):
    # idで重複排除しつつマージ
    existing_dict = {h.get('id'): h for h in existing}
    for h in new:
        existing_dict[h.get('id')] = h
    return list(existing_dict.values())

async def notify_self_backend(state: dict):
    """同一マシン上で動いているバックエンドの /state/progress エンドポイントへ
    非同期でパッチ(state) を POST するヘルパ。エンジン実行プロセスなど
    バックエンドとは別プロセスから呼び出されることを想定している。
    """
    url = "http://localhost:8000/state/progress"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=state, timeout=5.0)
            response.raise_for_status()
            logger.debug(f"[notify_self_backend] Successfully notified self backend. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"[notify_self_backend] Error notifying self backend: {e}", exc_info=True)

async def update_state_from_subgraph_progress(progress: dict, *, internal: bool = False):
    """サブグラフ実行中に逐次生成される progress パッチを処理する。

    * 外部プロセス(エンジン側) から直接呼ばれる場合は internal=False となり、
      HTTP 経由でバックエンドの /state/progress へ転送して終了する。
    * バックエンド FastAPI プロセス内で internal=True で呼ばれた場合のみ、
      latest_state へマージして WebSocket ブロードキャストを行う。
    """

    if not internal:
        # バックエンド外のプロセスで呼ばれたとみなし、HTTP で転送
        await notify_self_backend(progress)
        return

    # -------- バックエンドプロセス内: state をマージしてクライアントへ送信 --------
    global latest_state
    async with state_lock:
        # history や current_hypotheses のみ追記・マージ
        if "history" in progress and progress["history"]:
            latest_state.setdefault("history", []).extend(progress["history"])
        if "current_hypotheses" in progress and progress["current_hypotheses"]:
            latest_state["current_hypotheses"] = merge_hypotheses(
                latest_state.get("current_hypotheses", []),
                progress["current_hypotheses"]
            )

    # 最新状態をブロードキャスト
    await broadcast()  # 引数なしでlatest_stateを送信

# ---------------------------------------------------------------------------
# 進捗のみのパッチを受け取るエンドポイント
# LangGraph サブグラフからの細かい update 用
# ---------------------------------------------------------------------------

@app.post("/state/progress", summary="Receive incremental progress updates from subgraph")
async def receive_progress(progress: Dict[str, Any] = Body(...)):
    """engine_runner 側から飛んでくる progress パッチを latest_state にマージし、
    WebSocket クライアントへブロードキャストする。
    """
    logger.info(f"Received progress update via POST. Keys: {list(progress.keys())}")
    # internal=True としてバックエンド内処理を呼び出す
    await update_state_from_subgraph_progress(progress, internal=True)
    return {"message": "Progress update received and broadcasted"}
