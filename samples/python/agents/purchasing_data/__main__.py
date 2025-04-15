# import sys # 削除
# import os # 削除
import logging
import uvicorn

# A2A共通ライブラリ (プロジェクトルートからの絶対パスに修正)
from samples.python.common.server.server import A2AServer
from samples.python.common.types import (
    AgentCard, AgentProvider, AgentCapabilities, AgentSkill
)

# このエージェント固有の実装 (相対インポートを確認)
from .agent import run_agent
from .task_manager_impl import PurchasingDataTaskManager

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# エージェントカードの定義
# プロトタイプ計画に基づいて定義
PURCHASING_DATA_AGENT_CARD = AgentCard(
    name="PurchasingDataAgent",
    description="Provides information about purchasing data via A2A protocol.",
    url="http://localhost:8001", # サーバーがリッスンするURL
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
    defaultInputModes=["text"], # テキストで指示を受け取る
    defaultOutputModes=["text"], # テキストで応答する
    skills=[
        AgentSkill(
            id="analyze_product_data",
            name="Analyze Product Data",
            description="Product data について様々な統計情報を回答します。合計や平均などを計算できます。",
            inputModes=["text"],
            outputModes=["text"],
            examples=[
                "Find information about product P001", 
                "What is the price of P003?",
                "List products under 5000 yen",
                "Which product has the largest quantity in stock?",
                "Find products with price between 10000 and 20000 yen that have more than 10 in stock"
            ]
        )
    ]
)

if __name__ == "__main__":
    logger.info("Initializing PurchasingDataAgent server...")

    # TaskManagerのインスタンス化 (QueryAgentの実行関数を渡す)
    task_manager = PurchasingDataTaskManager(agent_runner=run_agent)
    logger.info("PurchasingDataTaskManager initialized.")

    # A2AServerのインスタンス化
    # TODO: ホスト、ポート、エンドポイントは設定ファイルや環境変数から読み込むのが望ましい
    a2a_server = A2AServer(
        host="0.0.0.0",
        port=8001, # A2Aサーバーのポート
        endpoint="/a2a", # A2Aのエンドポイント
        agent_card=PURCHASING_DATA_AGENT_CARD,
        task_manager=task_manager
    )
    logger.info(f"A2AServer initialized. Agent card available at /.well-known/agent.json")
    logger.info(f"A2A endpoint configured at: {a2a_server.endpoint}")

    # サーバー起動
    logger.info(f"Starting server on {a2a_server.host}:{a2a_server.port}...")
    # uvicorn.run(a2a_server.app, host=a2a_server.host, port=a2a_server.port)
    # A2AServer内のstartメソッドを使う (uvicornの起動も含まれる)
    try:
        a2a_server.start()
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)

    logger.info("Server stopped.")
