"""
エージェントを実行するためのシンプルなスクリプト

必要なパッケージ:
- pandas
- langchain-openai
- openai
- python-dotenv

以下のコマンドでインストールできます:
pip install pandas langchain-openai openai python-dotenv
"""

import asyncio
import os
from dotenv import load_dotenv
from query_agent import QueryAgent

# .envファイルから環境変数を読み込む
load_dotenv()

async def main():
    # エージェントのインスタンスを作成
    agent = QueryAgent()
    
    print("\n===== 購入データ検索エージェント =====")
    print("自然言語で質問してください。終了するには 'exit' と入力してください。")
    
    # 使用例を表示
    print("\n使用例:")
    print("- 「製品ID P001 の詳細情報を教えて」")
    print("- 「5000円以下の製品を探して」")
    print("- 「在庫数が最も多い製品は何？」")
    print("- 「ノートパソコンの価格はいくら？」")
    print("- 「P001の平均価格を教えて」")
    
    # 対話ループ
    while True:
        query = input("\n質問: ")
        if query.lower() in ["exit", "quit", "終了"]:
            print("プログラムを終了します。")
            break
        
        if not query.strip():
            continue
            
        try:
            # クエリを処理
            result = await agent.process_query(query)
            print(f"\n回答: {result}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 