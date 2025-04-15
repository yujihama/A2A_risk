import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 相対パスでツールをインポート (同じディレクトリ内)
from .tools import InventorySearchTool

# .envファイルから環境変数を読み込む (任意)
load_dotenv()

# LLMの初期化 (環境変数 `OPENAI_API_KEY` が必要)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ツールのリスト
tools = [InventorySearchTool()]

# プロンプトテンプレートの定義
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは在庫情報を検索する日本語アシスタントです。提供されたツールを使用して在庫データを検索してください。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# エージェントの作成
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutorの作成
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def run_agent(input_query: str) -> dict:
    """指定された入力クエリでエージェントを実行し、結果を返す"""
    # AgentExecutorは非同期実行 (`ainvoke`) もサポート
    # FastAPIと連携するため非同期で呼び出す
    response = await agent_executor.ainvoke({"input": input_query})
    return response

# --- 以下は直接実行した場合のテスト用コード (任意) ---
def _test_agent():
    import asyncio
    # 例: 商品ID I001 の情報を検索
    test_query = "商品ID I001 の在庫情報を教えてください"
    print(f"Testing agent with query: {test_query}")
    result = asyncio.run(run_agent(test_query))
    print("\nAgent Response:")
    print(result)

    # 例: 存在しない商品IDを検索
    test_query_not_found = "商品ID I999 の在庫はどうなっていますか？"
    print(f"\nTesting agent with query: {test_query_not_found}")
    result_not_found = asyncio.run(run_agent(test_query_not_found))
    print("\nAgent Response (Not Found):")
    print(result_not_found)

if __name__ == "__main__":
    # このファイルが直接実行された場合にテストコードを実行
    print("Running agent test...")
    _test_agent() 