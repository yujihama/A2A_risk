import logging
import uvicorn

# A2A共通ライブラリ
from samples.python.common.server.server import A2AServer
from samples.python.common.types import (
    AgentCard, AgentProvider, AgentCapabilities, AgentSkill, AgentAuthentication
)

# このエージェント固有の実装
from .agent import run_agent
from .task_manager_impl import InventoryDataTaskManager

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# エージェントカードの定義
INVENTORY_DATA_AGENT_CARD = AgentCard(
    name="InventoryDataAgent",
    description="在庫データ情報をA2Aプロトコルで提供します。",
    url="http://localhost:8002", # PurchasingDataAgentとは異なるポート
    provider=AgentProvider(
        organization="Example Inc."
    ),
    version="0.1.0-prototype",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=True
    ),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="search_inventory_data",
            name="Search Inventory Data",
            description="商品IDに基づいて在庫情報（保管場所、在庫数、再注文レベルなど）を検索します。",
            inputModes=["text"],
            outputModes=["text"],
            examples=["商品ID I001の在庫情報を教えてください", "I002はどこに保管されていますか？"]
        )
    ]
)

if __name__ == "__main__":
    logger.info("Initializing InventoryDataAgent server...")

    # TaskManagerのインスタンス化
    task_manager = InventoryDataTaskManager(agent_runner=run_agent)
    logger.info("InventoryDataTaskManager initialized.")

    # A2AServerのインスタンス化
    a2a_server = A2AServer(
        host="0.0.0.0",
        port=8002, # PurchasingDataAgentとは異なるポート
        endpoint="/a2a",
        agent_card=INVENTORY_DATA_AGENT_CARD,
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