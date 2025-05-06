"""
データエージェントを A2A サーバーとして起動するためのスクリプト。

設定ファイルに基づいてエージェントカード情報やデータソースを決定し、
指定されたホストとポートで A2A プロトコルリクエストを受け付けます。

例:
python -m samples.python.agents.data_agent --config config/sales_data_config.yaml --port 8001
"""
import asyncio
import argparse
import os
import sys
import logging
import yaml
from typing import Dict, Any

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A2A共通ライブラリ
try:
    from A2A_risk.samples.python.common.server.server import A2AServer
    from A2A_risk.samples.python.common.types import (
        AgentCard, AgentProvider, AgentCapabilities, AgentSkill
    )
except ImportError:
    logger.critical("A2A共通ライブラリが見つかりません。パスを確認してください。")
    sys.exit(1)

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# このエージェント固有の実装
try:
    from A2A_risk.samples.python.agents.data_agent.agent import QueryAgent, run_agent, initialize_agent_config, get_agent_instance
    from A2A_risk.samples.python.agents.data_agent.task_manager.task_manager_impl import DataAgentTaskManager
except ImportError as e:
    logger.critical(f"必要なモジュールが見つかりません: {e}")
    sys.exit(1)

# コマンドライン引数のデフォルト設定
DEFAULT_CONFIG_PATH = os.path.join(current_dir, 'config', 'purchasing_config.yaml')

def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込み、デフォルト値で補完する"""
    config = {
        'agent_name': "Generic Data Agent",
        'agent_description': "Analyzes data based on configuration.",
        'host': "0.0.0.0",
        'port': 8001,
        'llm_model': "gpt-4o-mini",
        'data_source': None, # データソースは必須
        'organization': "Your Organization",
        'version': "1.0.0",
        'endpoint': "/a2a",
        'defaultInputModes': ["text"],
        'defaultOutputModes': ["text"],
        'skills': [
            {
                "id": "query_data",
                "name": "Query Data",
                "description": "Query the configured data source using natural language.",
                "examples": [
                    "What is the total sales amount?",
                    "Show me products with price over 10000",
                    "Which category has the highest average rating?"
                ],
                "inputModes": ["text"],
                "outputModes": ["text"]
            }
        ]
    }
    if not os.path.exists(config_path):
        # 設定ファイルが必須ではない場合 (例: デフォルトデータを使う場合) は警告に留めることも可能
        # ここでは必須とするためエラー
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        raise FileNotFoundError(f"指定された設定ファイルが見つかりません: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)
        logger.info(f"設定ファイルを読み込みました: {config_path}")

        # 読み込んだ設定でデフォルト値を上書き
        config.update(loaded_config)

        # 必須キーのチェック
        if config.get('data_source') is None:
            logger.error(f"設定ファイルに必須キー 'data_source' がありません: {config_path}")
            raise ValueError(f"設定ファイルに 'data_source' が必要です: {config_path}")

        # データソースの相対パス解決
        if isinstance(config['data_source'], str) and not os.path.isabs(config['data_source']):
             config_dir = os.path.dirname(config_path)
             config['data_source'] = os.path.abspath(os.path.join(config_dir, config['data_source']))
             logger.info(f"相対データソースパスを解決しました: {config['data_source']}")

        # ポート番号を整数に変換
        try:
             config['port'] = int(config['port'])
        except (ValueError, TypeError):
             logger.warning(f"設定ファイルのポート番号が無効です: {config['port']}。デフォルト (8001) を使用します。")
             config['port'] = 8001

        # スキル定義のバリデーション
        if not isinstance(config['skills'], list):
            logger.warning(f"設定ファイルの skills はリスト形式である必要があります。デフォルト値を使用します。")
            config['skills'] = [
                {
                    "id": "query_data",
                    "name": "Query Data",
                    "description": "Query the configured data source using natural language.",
                    "examples": [
                        "What is the total sales amount?",
                        "Show me products with price over 10000",
                        "Which category has the highest average rating?"
                    ],
                    "inputModes": ["text"],
                    "outputModes": ["text"]
                }
            ]
        else:
            # 各スキルの必須フィールドをチェック
            validated_skills = []
            for skill in config['skills']:
                if not isinstance(skill, dict):
                    logger.warning(f"無効なスキル定義: {skill}. スキップします。")
                    continue
                
                # 必須フィールドの確認
                if 'id' not in skill or 'name' not in skill or 'description' not in skill:
                    logger.warning(f"スキル定義に必須フィールド (id, name, description) がありません: {skill}. スキップします。")
                    continue
                
                # 任意フィールドのデフォルト設定
                if 'examples' not in skill:
                    skill['examples'] = []
                if 'inputModes' not in skill:
                    skill['inputModes'] = ["text"]
                if 'outputModes' not in skill:
                    skill['outputModes'] = ["text"]
                
                validated_skills.append(skill)
            
            if validated_skills:
                config['skills'] = validated_skills
            else:
                logger.warning("有効なスキル定義がありません。デフォルト値を使用します。")
                config['skills'] = [
                    {
                        "id": "query_data",
                        "name": "Query Data",
                        "description": "Query the configured data source using natural language.",
                        "examples": [
                            "What is the total sales amount?",
                            "Show me products with price over 10000",
                            "Which category has the highest average rating?"
                        ],
                        "inputModes": ["text"],
                        "outputModes": ["text"]
                    }
                ]

        # 入出力モードのバリデーション
        for mode_key in ['defaultInputModes', 'defaultOutputModes']:
            if not isinstance(config[mode_key], list):
                logger.warning(f"設定ファイルの {mode_key} はリスト形式である必要があります。デフォルト値を使用します。")
                config[mode_key] = ["text"]

        return config
    except yaml.YAMLError as e:
        logger.error(f"設定ファイルの解析エラー ({config_path}): {e}")
        raise ValueError(f"設定ファイル {config_path} の形式が正しくありません。") from e
    except Exception as e:
         logger.error(f"設定ファイルの読み込み中に予期せぬエラー ({config_path}): {e}")
         raise

def main(args: argparse.Namespace):
    """メイン処理: A2Aサーバーを起動する"""
    try:
        # 設定ファイルを読み込み
        config = load_config(args.config)

        # 設定値を変数に展開 (可読性のため)
        host = args.host if args.host else config.get('host', "0.0.0.0")
        port = args.port if args.port else config.get('port', 8001)
        agent_name = config.get('agent_name', "Generic Data Agent")
        agent_description = config.get('agent_description', "Analyzes data based on configuration.")
        data_source = config['data_source'] # 必須キーなので必ず存在する
        llm_model = config.get('llm_model', "gpt-4o-mini")
        organization = config.get('organization', "Your Organization")
        version = config.get('version', "1.0.0")
        endpoint = config.get('endpoint', "/a2a")
        # 追加項目
        default_input_modes = config.get('defaultInputModes', ["text"])
        default_output_modes = config.get('defaultOutputModes', ["text"])
        skills = config.get('skills', [])

        # ★重要: agent.py の設定データを初期化
        logger.info(f"エージェントの設定を初期化します...")
        initialize_agent_config(config) # 読み込んだ設定データを渡す

        # TaskManagerのインスタンス化 (run_agent 関数を渡す)
        task_manager = DataAgentTaskManager(agent_runner=run_agent)
        logger.info("TaskManager initialized.")

        # Agent Card の動的生成
        agent_url = f"http://{host}:{port}" # TODO: 正確なURL生成 (https考慮など)
        agent_card = AgentCard(
            name=agent_name,
            description=agent_description,
            url=agent_url,
            provider=AgentProvider(organization=organization), # 設定ファイルから取得
            version=version, # 設定ファイルから取得
            capabilities=AgentCapabilities(streaming=False, pushNotifications=False, stateTransitionHistory=True),
            defaultInputModes=default_input_modes, # 設定ファイルから取得
            defaultOutputModes=default_output_modes, # 設定ファイルから取得
            skills=[
                AgentSkill(
                    id=skill.get('id', 'query_data'),
                    name=skill.get('name', 'Query Data'),
                    description=skill.get('description', 'Query data source'),
                    inputModes=skill.get('inputModes', default_input_modes),
                    outputModes=skill.get('outputModes', default_output_modes),
                    examples=skill.get('examples', [])
                ) for skill in skills
            ]
        )
        logger.info(f"AgentCard generated for '{agent_name}'")

        # A2AServerのインスタンス化
        a2a_server = A2AServer(
            host=host,
            port=port,
            endpoint=endpoint, # 設定ファイルから取得
            agent_card=agent_card,
            task_manager=task_manager
        )
        logger.info(f"A2AServer initialized. Agent card available at {agent_url}/.well-known/agent.json")
        logger.info(f"A2A endpoint configured at: {agent_url}{a2a_server.endpoint}")

        # ★ここでデータロードを明示的に実行
        try:
            get_agent_instance()
            logger.info("QueryAgentのデータロードが完了しました。")
        except Exception as e:
            logger.error(f"QueryAgentの初期化・データロードに失敗しました: {e}")
            raise

        # サーバー起動
        logger.info(f"Starting A2A server on {host}:{port}...")
        try:
            # A2AServer.start()を直接呼び出す
            a2a_server.start() # この関数内で uvicorn.run() が呼ばれ、ブロッキングモードでサーバーが起動
        except Exception as e:
            logger.error(f"サーバー起動に失敗しました: {e}", exc_info=True)
            raise

    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        logger.error(f"起動エラー: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"必要なライブラリが見つかりません: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"予期せぬ致命的なエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データ対話 A2A エージェントサーバー")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"使用する設定ファイルのパス (デフォルト: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None, # デフォルトは設定ファイルの値を使用
        help=f"リッスンするホスト名/IPアドレス (デフォルト: 設定ファイルの値 or 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None, # デフォルトは設定ファイルの値を使用
        help=f"リッスンするポート番号 (デフォルト: 設定ファイルの値 or 8001)"
    )
    args = parser.parse_args()

    try:
        # main関数を同期的に実行
        main(args)
    except KeyboardInterrupt:
        logger.info("サーバーを停止します...")
