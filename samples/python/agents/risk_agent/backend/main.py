from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import sys # sys をインポート
from typing import Dict, Any # Dict, Any をインポート

# ルートロガーではなく、このモジュール用のロガーを取得
logger = logging.getLogger(__name__) # logger = logging.getLogger("backend.main") でも可
logger.setLevel(logging.DEBUG) # このロガーのレベルを DEBUG に設定

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

# ★ 新しいエンドポイントを追加
@app.post("/state/update", summary="Receive state updates from the engine")
async def update_state(state: Dict[str, Any] = Body(...)):
    """LangGraphエンジンから状態更新を受け取り、WebSocketクライアントにブロードキャストする。"""
    logger.info(f"Received state update via POST. Keys: {list(state.keys())}")
    # 受け取った state で latest_state を更新し、broadcast を呼び出す
    # broadcast は非同期なので await する
    await broadcast(state) # ← コメントアウトを解除
    return {"message": "State update received and broadcasted"}

@app.get("/state/initial", summary="Get the initial state for the frontend")
async def get_state():
    """フロントエンドが初回ロード時に呼び出すエンドポイント。
    現在の最新のLangGraph状態を返します。
    """
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


async def broadcast(state: dict):
    """
    LangGraphの実行ループから呼び出され、更新された状態を
    接続中の全てのWebSocketクライアントにブロードキャストします。
    """
    # ★ broadcast オブジェクトの ID をログに出力
    logger.debug(f"--- broadcast function entered (ID: {id(broadcast)}) ---")
    global latest_state
    latest_state = state # 最新状態を更新

    # ★ 呼び出された時点での接続数をログに出力
    logger.debug(f"Broadcast called. Current connections: {len(connections)}")

    if not connections:
        # ★ ログレベルを DEBUG に変更して、表示されるようにする
        logger.debug("No WebSocket clients connected, skipping broadcast.")
        return

    # stateをJSON文字列に一度だけ変換
    try:
        # シリアライズ不可能な型がないか確認しつつ変換
        # ここで Pydantic モデルやカスタムエンコーダーを使うとより堅牢
        state_json = json.dumps(state, default=str) # default=str で暫定対応
    except TypeError as e:
        logger.error(f"Failed to serialize state for broadcast: {e}. State snippet: {str(state)[:200]}...", exc_info=True)
        return # シリアライズできない場合は送信しない
    except Exception as e:
         logger.error(f"Unexpected error during state serialization: {e}", exc_info=True)
         return

    logger.info(f"Broadcasting state update to {len(connections)} clients.")
    # asyncio.gatherで並行送信を試みる
    # 送信中に接続が切れるケースを考慮し、送信前に接続リストをコピー
    active_connections = list(connections)
    results = await asyncio.gather(
        *[ws.send_text(state_json) for ws in active_connections],
        return_exceptions=True # エラー発生時も継続
    )

    # 送信失敗した接続をログ記録 (エラー時の削除は finally で行う)
    failed_connections_info = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            ws = active_connections[i]
            client_host = ws.client.host if ws.client else "Unknown"
            client_port = ws.client.port if ws.client else "N/A"
            failed_connections_info.append(f"{client_host}:{client_port} ({type(result).__name__}: {result})")

    if failed_connections_info:
        logger.warning(f"Failed to send broadcast to some clients: {', '.join(failed_connections_info)}")


# サーバー単体起動確認用は削除しても良い (uvicorn コマンド推奨)
# if __name__ == "__main__":
#    ... 