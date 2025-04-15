# import sys # 削除
# import os # 削除
import asyncio
import argparse
import logging
import sys

# agent.py から run_graph 関数をインポート (相対パスに変更)
from .agent import run_graph
# from samples.python.agents.kakaku_ijou_signal.agent import run_graph # 絶対パスはコメントアウト

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """メイン関数: コマンドライン引数を解析して適切な関数を呼び出す"""
    parser = argparse.ArgumentParser(description='価格異常兆候エージェント')
    
    # 単一の製品IDを指定する場合のサブコマンド
    parser.add_argument('product_id', nargs='?', 
                       help='調査する製品ID（例: P001, I001）')
    
    # テストオプション
    parser.add_argument('--test-all', action='store_true', 
                       help='すべてのテストケースを実行する')
    
    args = parser.parse_args()
    
    # 引数がない場合はヘルプを表示
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    if args.test_all:
        # テスト実行
        from .agent import _test_graph
        print("すべてのテストケースを実行します...")
        asyncio.run(_test_graph())
    elif args.product_id:
        # 単一の製品IDで実行
        print(f"製品ID: {args.product_id} の情報をリクエストします...")
        asyncio.run(run_graph(args.product_id))
    else:
        parser.print_help()

if __name__ == "__main__":
    # このファイルが直接実行された場合に main 関数を呼び出す
    # 依存関係 (dotenv, langchain-openai, langgraph, langchain-coreなど) のインストールと
    # 環境変数 (OPENAI_API_KEY, PURCHASING_AGENT_URL) の設定、
    # および PurchasingDataAgent サーバーの起動が必要
    main()
