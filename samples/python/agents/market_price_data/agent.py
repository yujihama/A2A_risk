import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# 相対パスでツールをインポート (同じディレクトリ内)
from .tools import MarketPriceSearchTool

# .envファイルから環境変数を読み込む (任意)
load_dotenv()

# LLMの初期化 (環境変数 `OPENAI_API_KEY` が必要)
# TODO: プロトタイプ計画に基づき、Google Gemini等に変更可能にする
llm = ChatOpenAI(model="gpt-4o", temperature=0) # modelは適宜変更

# ツールのリスト
tools = [MarketPriceSearchTool()]

# プロンプトテンプレートの定義
# システムプロンプトでエージェントの役割やツールの使い方を指示
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは製品IDから市場価格情報を検索するアシスタントです。提供されたツールを使って市場価格データを検索してください。結果は日本語で返してください。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # エージェントの中間ステップ(思考過程やツール呼び出し結果)を格納
    ]
)

# エージェントの作成
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutorの作成
# verbose=True で実行時の詳細ログを出力 (デバッグ用)
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
    # 例: 商品ID P002 の情報を検索
    test_query = "製品ID P002 の市場価格を教えてください。"
    print(f"Testing agent with query: {test_query}")
    result = asyncio.run(run_agent(test_query))
    print("\nAgent Response:")
    print(result)

    # 例: 存在しない商品IDを検索
    test_query_not_found = "製品ID P999 の市場価格はいくらですか？"
    print(f"\nTesting agent with query: {test_query_not_found}")
    result_not_found = asyncio.run(run_agent(test_query_not_found))
    print("\nAgent Response (Not Found):")
    print(result_not_found)

if __name__ == "__main__":
    # このファイルが直接実行された場合にテストコードを実行
    # 依存関係 (dotenv, langchain-openaiなど) のインストールが必要
    print("Running agent test...")
    _test_agent() 