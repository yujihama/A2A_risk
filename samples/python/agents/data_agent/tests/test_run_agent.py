"""
データエージェントを対話的にテスト実行するためのシンプルなスクリプト。

内部で QueryAgent を直接インスタンス化し、デフォルトのデータをロードします。
"""
import asyncio
import os
import logging
import yaml
import argparse

# .envファイル読み込みのために必要
from dotenv import load_dotenv

# agent ディレクトリ内の query_processor をインポート
from ..agent.query_processor import QueryAgent

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む (OpenAI APIキーなど)
load_dotenv()

# 設定ファイルとデータファイルのデフォルトパス
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "config"))
DEFAULT_DATA_PATH = os.path.join(CURRENT_DIR, "data", "dummy_data.csv")

async def main(config_filename: str):
    try:
        # 設定ファイルのフルパスを構築
        config_path = os.path.join(CONFIG_DIR, config_filename)
        if not os.path.isfile(config_path):
             logger.error(f"指定された設定ファイルが見つかりません: {config_path}")
             print(f"エラー: 指定された設定ファイルが見つかりません: {config_path}")
             return

        # 設定を読み込む
        config = {}
        llm_model = 'gpt-4o-mini'
        data_source = DEFAULT_DATA_PATH
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"テスト用に設定ファイルを読み込みました: {config_path}")
            llm_model = config.get('llm_model', llm_model)
            data_source_from_config = config.get('data_source')

            # データソースパスの解決
            if data_source_from_config:
                 if not os.path.isabs(data_source_from_config):
                     # 設定ファイルからの相対パスとして解釈
                     config_dir_for_data = os.path.dirname(config_path)
                     data_source_abs = os.path.abspath(os.path.join(config_dir_for_data, data_source_from_config))
                     if os.path.exists(data_source_abs):
                         data_source = data_source_abs
                     else:
                         logger.warning(f"設定ファイルに指定されたデータソースが見つかりません（{data_source_from_config} -> {data_source_abs}）。デフォルトデータパスを使用します: {DEFAULT_DATA_PATH}")
                 else:
                     # 絶対パスの場合
                     if os.path.exists(data_source_from_config):
                         data_source = data_source_from_config
                     else:
                          logger.warning(f"設定ファイルに指定された絶対パスのデータソースが見つかりません: {data_source_from_config}。デフォルトデータパスを使用します: {DEFAULT_DATA_PATH}")
            else:
                 logger.warning(f"設定ファイルに 'data_source' が指定されていません。デフォルトデータパスを使用します: {DEFAULT_DATA_PATH}")

        except Exception as e:
             logger.warning(f"設定ファイルの読み込み/解析エラー: {e}. デフォルト値で続行します。")
             # エラーが発生した場合もデフォルト値を使用

        # エージェントのインスタンスを作成 (設定ファイルまたはデフォルトのモデルを使用)
        agent = QueryAgent(model=llm_model)

        # データをロード (設定ファイルまたはデフォルトのデータソースを使用)
        logger.info(f"テスト用データをロードします: {data_source}")
        agent.load_data(data_source, build_embedding=True)

        print("\n===== データ対話エージェント (テストモード) =====")
        print(f"使用データ: {data_source}")
        print(f"使用モデル: {llm_model}")
        print("自然言語で質問してください。終了するには 'exit' または 'quit' と入力してください。")

        # 使用例を表示
        print("\n使用例:")
        print("- 「製品ID P001 の詳細情報を教えて」")
        print("- 「5000円以下の製品を探して」")
        print("- 「在庫数が最も多い製品は何？」")
        print("- 「ノートパソコンの価格はいくら？」")
        print("- 「価格が10000円から20000円の間の製品で、在庫が10個以上あるものを教えてください」")

        # 対話ループ
        while True:
            query = input("\n質問: ")
            if query.lower() in ["exit", "quit", "終了"]:
                print("プログラムを終了します。")
                break

            if not query.strip():
                continue

            # クエリを処理
            result = await agent.process_query(query)
            print(f"\n回答:\n{result}")

    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        logger.error(f"テスト実行中にエラーが発生しました: {e}")
        print(f"\nエラー: {e}")
    except ImportError as e:
        logger.error(f"必要なライブラリが見つかりません: {e}")
        print(f"\nエラー: 必要なライブラリ ({e}) がインストールされていません。pip install PyYAML pandas などでインストールしてください。")
    except Exception as e:
        logger.critical(f"予期せぬ致命的なエラーが発生しました: {e}", exc_info=True)
        print(f"\n予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    logger.info("テストスクリプトを開始します...")

    # コマンドライン引数を解析
    parser = argparse.ArgumentParser(description="データエージェントを対話的にテスト実行するスクリプト")
    parser.add_argument(
        'config_file',
        nargs='?',
        default='purchasing_config.yaml',
        help='configディレクトリ内のYAML設定ファイル名 (例: purchasing_config.yaml)'
    )
    args = parser.parse_args()

    # main関数を実行
    asyncio.run(main(args.config_file)) 