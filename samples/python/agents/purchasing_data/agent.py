import os
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union

# 新しいQueryAgentクラスをインポート
from .query_agent import QueryAgent

# 環境変数の読み込み
load_dotenv()

# QueryAgentのシングルトンインスタンス
_agent_instance = None

def get_agent_instance() -> QueryAgent:
    """
    QueryAgentのシングルトンインスタンスを取得または作成する
    
    Returns:
        QueryAgent: エージェントのインスタンス
    """
    global _agent_instance
    if _agent_instance is None:
        # デフォルトのデータファイルパスとモデルを使用してインスタンスを作成
        _agent_instance = QueryAgent()
    return _agent_instance

async def run_agent(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    エージェントを実行し、クエリに対する応答を取得する
    
    Args:
        input_data (Union[str, Dict[str, Any]]): ユーザーからの入力。文字列または辞書形式。
            辞書の場合: {"input": "クエリ文字列", "product_id": "PXXX", ...}
        
    Returns:
        Dict[str, Any]: 応答を含む辞書 {"output": "応答テキスト"}
    """
    agent = get_agent_instance()
    
    # 入力データからクエリ文字列と product_id を抽出
    input_query = ""
    product_id = None
    if isinstance(input_data, str):
        input_query = input_data
    elif isinstance(input_data, dict):
        input_query = input_data.get("input", "")
        product_id = input_data.get("product_id") # product_id を取得
        # 他のパラメータも必要に応じて抽出
    else:
        raise TypeError("input_data must be either str or dict")

    try:
        # QueryAgentのprocess_queryメソッドを呼び出す
        # product_id も渡すように変更（process_query が対応している必要あり）
        result = await agent.process_query(input_query, product_id=product_id)
        return {"output": result}
    except Exception as e:
        # エラーハンドリング
        error_message = f"エラーが発生しました: {str(e)}"
        return {"output": error_message, "error": str(e)}

# テスト用の関数
async def test_agent():
    """
    エージェントのテストを実行する
    """
    test_queries = [
        "製品ID P001 の詳細情報を教えてください",
        "5000円以下の製品を探してください",
        "在庫数が最も多い製品は何ですか？",
        "ノートパソコンの価格はいくらですか？",
        "10000円から20000円の間の製品で、在庫が10個以上あるものを教えてください"
    ]
    
    for query in test_queries:
        print(f"\n----- クエリ: {query} -----")
        response = await run_agent(query)
        print(f"応答: {response['output']}")
    
    return "テスト完了"

if __name__ == "__main__":
    print("エージェントのテストを実行します...")
    asyncio.run(test_agent())
