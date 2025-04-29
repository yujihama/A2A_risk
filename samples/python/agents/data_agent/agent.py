import os
import asyncio
import logging
import yaml
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union
import argparse # argparse をインポート

# query_agentを明示的にインポート
from .query_agent import QueryAgent

# ロガー設定
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# 設定ファイルとデータファイルのデフォルトパス
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(CURRENT_DIR, "data", "dummy_data.csv")
DEFAULT_LLM_MODEL = 'gpt-4o' # デフォルトLLMモデルを定数化

# QueryAgentのシングルトンインスタンスと設定
_agent_instance: Optional[QueryAgent] = None
_config: Optional[Dict[str, Any]] = None

def initialize_agent_config(config_data: Dict[str, Any]):
    """
    エージェントの設定データを初期化する。

    Args:
        config_data (Dict[str, Any]): ロードされた設定データ。
    """
    global _config, _agent_instance
    logger.info("エージェント設定を初期化します。")
    _config = config_data
    # 設定が変わった可能性があるので、インスタンスをリセット
    _agent_instance = None
    logger.info("エージェント設定が初期化されました。")

def get_agent_instance() -> QueryAgent:
    """
    QueryAgentのシングルトンインスタンスを取得または作成し、データをロードする

    Returns:
        QueryAgent: データロード済みのエージェントインスタンス

    Raises:
        RuntimeError: エージェントの初期化またはデータロードに失敗した場合
    """
    global _agent_instance, _config
    if _agent_instance is None:
        logger.info("QueryAgentのインスタンスを初期化します。")

        # _config が初期化されているかチェック
        if _config is None:
            logger.error("エージェント設定が初期化されていません。先に initialize_agent_config を呼び出してください。")
            raise RuntimeError("エージェント設定が初期化されていません。")

        try:
            # 設定値またはデフォルト値を使用
            llm_model = _config.get('llm_model', DEFAULT_LLM_MODEL)
            # データソースパスは __main__ で解決済みの想定
            data_source = _config.get('data_source', DEFAULT_DATA_PATH)

            logger.info(f"使用するLLMモデル: {llm_model}")
            # logger.info(f"使用するデータソース: {data_source}") # __main__側でログ出力済みの想定

            # インスタンスを作成
            _agent_instance = QueryAgent(model=llm_model)

            # データをロード
            logger.info(f"データソースをロードします: {data_source}")
            _agent_instance.load_data(data_source)
            logger.info(f"データロード完了 (ソース: {data_source})。")

        except ImportError as e:
             logger.error(f"エージェントの初期化に必要なライブラリが不足しています: {e}", exc_info=True)
             _agent_instance = None # 失敗したらNoneに戻す
             raise RuntimeError(f"エージェント初期化失敗: 必要なライブラリがありません ({e})") from e
        except (FileNotFoundError, ValueError, TypeError) as e:
             logger.error(f"データソースのロードまたは設定ファイルの処理に失敗しました: {e}", exc_info=True)
             _agent_instance = None
             raise RuntimeError(f"エージェント初期化失敗: データ/設定に問題があります ({e})") from e
        except Exception as e:
            logger.error(f"QueryAgentの初期化またはデータロード中に予期せぬエラーが発生しました: {e}", exc_info=True)
            _agent_instance = None
            raise RuntimeError(f"エージェント初期化中に予期せぬエラーが発生しました ({e})") from e

    if _agent_instance is None:
         # この状態は通常起こらないはずだが念のため
         raise RuntimeError("不明な理由によりエージェントインスタンスが利用できません。")

    return _agent_instance

async def run_agent(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    エージェントを実行し、クエリに対する応答を取得する

    Args:
        input_data (Union[str, Dict[str, Any]]): ユーザーからの入力。
            文字列の場合: クエリ文字列。
            辞書の場合: {"input": "クエリ文字列", "product_id": "PXXX", ...}。
                      任意の追加パラメータを含めることができる。

    Returns:
        Dict[str, Any]: 応答を含む辞書 {"output": "応答テキスト", "error": "エラーメッセージ(オプション)"}
    """
    try:
        agent = get_agent_instance()
    except RuntimeError as e:
        logger.error(f"エージェントインスタンスの取得に失敗: {e}")
        return {"output": str(e), "error": str(e)}
    except Exception as e:
         logger.error(f"エージェントインスタンス取得中に予期せぬエラー: {e}", exc_info=True)
         return {"output": "エージェントの準備中に予期せぬエラーが発生しました。", "error": str(e)}

    # 入力データからクエリ文字列と追加パラメータを抽出
    input_query = ""
    additional_params = {}
    if isinstance(input_data, str):
        input_query = input_data
    elif isinstance(input_data, dict):
        input_query = input_data.get("input", "")
        # 'input' キー以外のすべての項目を追加パラメータとして渡す
        additional_params = {k: v for k, v in input_data.items() if k != "input"}
    else:
        logger.error(f"無効な入力データ型です: {type(input_data)}")
        return {"output": "入力データは文字列または辞書形式である必要があります。", "error": "Invalid input type"}

    if not input_query.strip():
        logger.warning("空のクエリを受け取りました。")
        return {"output": "クエリが空です。", "error": "Empty query"}

    try:
        # QueryAgentのprocess_queryメソッドを呼び出す
        # クエリ文字列と追加パラメータ(kwargs)を渡す
        logger.info(f"クエリ '{input_query}' を処理します。パラメータ: {additional_params}")
        result = await agent.process_query(input_query, **additional_params)
        return {"output": result}
    except Exception as e:
        # エラーハンドリング
        logger.error(f"クエリ処理中にエラーが発生しました: {e}", exc_info=True)
        error_message = f"クエリ処理中にエラーが発生しました: {str(e)}"
        return {"output": error_message, "error": str(e)}

# (test_agent 関数は __main__ や test_run_agent に役割を移譲するため削除またはコメントアウト)
# async def test_agent():
#     ...

# (if __name__ == "__main__" ブロックも通常は削除)
