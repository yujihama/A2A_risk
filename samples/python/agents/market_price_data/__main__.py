import logging
import uvicorn

# A2A共通ライブラリ (プロジェクトルートからの絶対パスに修正)
from samples.python.common.server.server import A2AServer
from samples.python.common.types import (
    AgentCard, AgentProvider, AgentCapabilities, AgentSkill, AgentAuthentication
)

# このエージェント固有の実装 (相対インポートを確認)
from .agent import run_agent
from .task_manager_impl import MarketPriceTaskManager

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# エージェントカードの定義
# プロトタイプ計画に基づいて定義
MARKET_PRICE_AGENT_CARD = AgentCard(
    name="MarketPriceAgent",
    description="製品IDを元に市場価格情報を提供するA2Aプロトコル対応エージェント。",
    url="http://localhost:8003", # サーバーがリッスンするURL (ポート番号を8003に変更)
    provider=AgentProvider(
        organization="Example Inc."
    ),
    version="0.1.0-prototype",
    capabilities=AgentCapabilities(
        streaming=False, # ストリーミングは未実装
        pushNotifications=False, # Push通知は未実装
        stateTransitionHistory=True # タスクの状態履歴は管理する
    ),
    # authentication=AgentAuthentication(schemes=["none"]), # 認証は今回なし
    defaultInputModes=["text"], # 主にテキストで指示を受け取る
    defaultOutputModes=["text"], # 主にテキストで応答する
    skills=[
        AgentSkill(
            id="search_market_price",
            name="市場価格検索",
            description="製品IDを元に製品の市場価格情報（名前、定価、市場価格、最終更新日）を検索します。",
            inputModes=["text"], # テキストでProductIDを受け取る想定
            outputModes=["text"],
            examples=["製品ID P001 の市場価格を教えて", "製品ID P003 はいくらですか？"]
        )
    ]
)

if __name__ == "__main__":
    logger.info("Initializing MarketPriceAgent server...")

    # TaskManagerのインスタンス化 (LangChainエージェントの実行関数を渡す)
    task_manager = MarketPriceTaskManager(agent_runner=run_agent)
    logger.info("MarketPriceTaskManager initialized.")

    # A2AServerのインスタンス化
    # ポートは購入情報エージェントと被らないようにする
    a2a_server = A2AServer(
        host="0.0.0.0",
        port=8003, # ポート番号を8003に変更
        endpoint="/a2a", # A2Aのエンドポイント (仮)
        agent_card=MARKET_PRICE_AGENT_CARD,
        task_manager=task_manager
    )
    logger.info(f"A2AServer initialized. Agent card available at /.well-known/agent.json")
    logger.info(f"A2A endpoint configured at: {a2a_server.endpoint}")

    # サーバー起動
    logger.info(f"Starting server on {a2a_server.host}:{a2a_server.port}...")
    try:
        a2a_server.start()
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)

    logger.info("Server stopped.") 