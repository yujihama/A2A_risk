import asyncio
import argparse
import json
import os
import logging
import sys
import re
import yaml
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 必要なモジュールをインポート
from samples.python.agents.smart_kakaku_signal.agent import initialize_registry
from samples.python.agents.smart_kakaku_signal.scenario_manager import ScenarioManager
from samples.python.agents.smart_kakaku_signal.scenario_engine import ScenarioExecutionEngine

# ロガーの取得
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギング設定を行う関数
def setup_logging(log_file=None):
    """
    ロギングの設定を行う関数
    
    Args:
        log_file: ログファイルのパス（指定がなければ標準出力のみ）
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # ディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # ファイルハンドラを追加
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(file_handler)
        logger.info(f"ログをファイルに出力します: {log_file}")
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # 既存のハンドラを上書き
    )

def parse_params(params_str):
    """
    JSONパラメータを柔軟に解析する関数
    PowerShellのエスケープ問題にも対応
    """
    if not params_str:
        return {}
        
    # PowerShellエスケープ文字を削除
    params_str = params_str.replace('\\', '')
    
    # 単純なキーバリューペアのパターンを検出して修正
    if not (params_str.startswith('{') and params_str.endswith('}')):
        params_str = '{' + params_str + '}'
    
    # 余分なクォートを削除
    params_str = params_str.replace('""', '"').replace("''", "'")
    
    try:
        return json.loads(params_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析エラー: {e}。シンプルなパラメータ解析を試みます。")
        
        # シンプルな解析を試みる
        result = {}
        # product_id: P001 のようなパターンを検出
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^,}\s]+)'
        matches = re.findall(pattern, params_str)
        for key, value in matches:
            # 数値に変換可能な場合は数値として扱う
            if value.isdigit():
                result[key] = int(value)
            elif value.lower() in ('true', 'false'):
                result[key] = value.lower() == 'true'
            else:
                result[key] = value
                
        if result:
            logger.info(f"シンプル解析結果: {result}")
            return result
            
        # 解析に失敗した場合はデフォルト値を返す
        logger.error(f"パラメータの解析に失敗しました: {params_str}")
        return {"product_id": "P001"}  # デフォルト値

def update_config(config_path, updates):
    """
    設定ファイルを更新する
    
    Args:
        config_path: 設定ファイルのパス
        updates: 更新内容の辞書（例: {"agents.purchasing_data.url": "http://localhost:5001"}）
    """
    try:
        # 設定ファイルを読み込む
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 変更を適用
        for key_path, value in updates.items():
            keys = key_path.split('.')
            current = config
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    # 最後の要素は値を更新
                    current[key] = value
                else:
                    # 途中の要素は辞書を作成（存在しない場合）
                    if key not in current:
                        current[key] = {}
                    current = current[key]
        
        # 設定ファイルに書き戻す
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"設定ファイルを更新しました: {config_path}")
        return True
    except Exception as e:
        logger.error(f"設定ファイルの更新中にエラーが発生しました: {e}")
        return False

async def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="柔軟なシナリオベース異常検知システム")
    
    # グローバルオプション
    parser.add_argument("--config", help="使用する設定ファイルのパス", 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_config.yaml"))
    parser.add_argument("--update-config", help="設定ファイルを更新 (例: 'agents.purchasing_data.url=http://localhost:5001')")
    parser.add_argument("--test-all", action="store_true", help="すべてのテストシナリオを実行")
    parser.add_argument("--log-file", help="ログを出力するファイルパス（例: 'logs/smart_kakaku.log'）")
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # シナリオ実行コマンド
    run_parser = subparsers.add_parser("run", help="シナリオを実行")
    run_parser.add_argument("--scenario-id", help="実行するシナリオID")
    run_parser.add_argument("--scenario-text", help="シナリオのテキスト（直接指定）")
    run_parser.add_argument("--params", help="シナリオのパラメータ (JSON形式)", default="{}")
    run_parser.add_argument("--debug", action="store_true", help="デバッグモード")
    
    # シナリオ保存コマンド
    save_parser = subparsers.add_parser("save", help="シナリオを保存")
    save_parser.add_argument("--id", required=True, help="シナリオID")
    save_parser.add_argument("--name", required=True, help="シナリオ名")
    save_parser.add_argument("--description", required=True, help="シナリオの説明")
    
    # シナリオ一覧表示コマンド
    list_parser = subparsers.add_parser("list", help="シナリオ一覧を表示")
    
    # シナリオ削除コマンド
    delete_parser = subparsers.add_parser("delete", help="シナリオを削除")
    delete_parser.add_argument("--id", required=True, help="削除するシナリオID")
    
    args = parser.parse_args()
    
    # ロギング設定を更新（コマンドライン引数からログファイルを設定）
    if args.log_file:
        setup_logging(args.log_file)
    
    # 設定ファイルの更新処理
    if args.update_config:
        try:
            updates = {}
            for item in args.update_config.split(','):
                key, value = item.split('=')
                updates[key.strip()] = value.strip()
            
            success = update_config(args.config, updates)
            if success:
                logger.info("設定を更新しました。新しい設定が適用されます。")
            else:
                logger.error("設定の更新に失敗しました。")
                return
        except Exception as e:
            logger.error(f"設定の更新中にエラーが発生しました: {e}")
            return
    
    # テストモードの場合は全シナリオを実行
    if args.test_all:
        logger.info("すべてのテストシナリオを実行します")
        # エージェントレジストリを初期化
        await initialize_registry()
        
        # シナリオディレクトリのパス
        scenario_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
        
        # シナリオマネージャーの初期化
        scenario_manager = ScenarioManager(scenario_dir)
        
        # すべてのシナリオを取得
        scenarios = scenario_manager.get_all_scenarios()
        
        # LLMクライアントの初期化
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # シナリオ実行エンジンの初期化
        from samples.python.agents.smart_kakaku_signal.agent import registry
        engine = ScenarioExecutionEngine(llm, registry)
        
        total_scenarios = len(scenarios)
        logger.info(f"テスト実行: 合計 {total_scenarios} シナリオを実行します")
        
        for i, scenario in enumerate(scenarios):
            try:
                logger.info(f"シナリオ {i+1}/{total_scenarios}: '{scenario['name']}' を実行します")
                
                # デフォルトパラメータ
                parameters = {"product_id": "P001"}
                
                # シナリオの実行
                result = await engine.execute_scenario(
                    scenario_text=scenario["description"],
                    parameters=parameters,
                    debug=True
                )
                
                # 結果の表示
                print(f"\n=== シナリオ {i+1}/{total_scenarios} 実行結果: {scenario['name']} ===")
                print(f"異常検出: {'あり' if result['is_anomaly'] else '異常なし' if result['is_anomaly'] is False else '判断保留'}")
                print(f"データ十分性: {'十分' if result['is_data_sufficient'] else '不十分'}")
                print(f"分析: {result['analysis']}")
                print("---\n")
                
            except Exception as e:
                logger.error(f"シナリオの実行中にエラーが発生しました: {e}")
        
        logger.info("すべてのテストシナリオの実行が完了しました")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # シナリオディレクトリのパス
    scenario_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
    
    # シナリオマネージャーの初期化
    scenario_manager = ScenarioManager(scenario_dir)
    
    # シナリオの保存
    if args.command == "save":
        try:
            file_path = scenario_manager.save_scenario(args.id, args.name, args.description)
            logger.info(f"シナリオを保存しました: {file_path}")
        except Exception as e:
            logger.error(f"シナリオの保存中にエラーが発生しました: {e}")
        return
    
    # シナリオ一覧の表示
    elif args.command == "list":
        try:
            scenarios = scenario_manager.get_all_scenarios()
            logger.info(f"登録されているシナリオ: {len(scenarios)}件")
            for scenario in scenarios:
                print(f"ID: {scenario['scenario_id']}")
                print(f"名前: {scenario['name']}")
                print(f"説明: {scenario['description'][:100]}...")
                print(f"作成日時: {scenario.get('created_at', '不明')}")
                print("---")
        except Exception as e:
            logger.info(f"シナリオ一覧の取得中にエラーが発生しました: {e}")
        return
    
    # シナリオの削除
    elif args.command == "delete":
        try:
            success = scenario_manager.delete_scenario(args.id)
            if success:
                logger.info(f"シナリオを削除しました: {args.id}")
            else:
                logger.error(f"シナリオの削除に失敗しました: {args.id}")
        except Exception as e:
            logger.error(f"シナリオの削除中にエラーが発生しました: {e}")
        return
    
    # シナリオの実行
    elif args.command == "run":
        # エージェントレジストリを初期化
        await initialize_registry()
        
        # LLMクライアントの初期化
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        try:
            # シナリオテキストの取得
            if args.scenario_id:
                scenario = scenario_manager.get_scenario(args.scenario_id)
                if not scenario:
                    logger.error(f"エラー: シナリオID '{args.scenario_id}' が見つかりません")
                    return
                scenario_text = scenario["description"]
                logger.info(f"シナリオ '{scenario['name']}' を実行します")
            elif args.scenario_text:
                scenario_text = args.scenario_text
                logger.info("直接指定されたシナリオを実行します")
            else:
                logger.error("エラー: scenario-id または scenario-text のいずれかを指定してください")
                return
                
            # パラメータの解析 - 改良版を使用
            try:
                parameters = parse_params(args.params)
                logger.info(f"解析されたパラメータ: {parameters}")
            except Exception as e:
                logger.error(f"パラメータの解析中にエラーが発生しました: {e}")
                logger.info("デフォルトパラメータを使用します: {'product_id': 'P001'}")
                parameters = {"product_id": "P001"}
            
            # シナリオ実行エンジンの初期化
            from samples.python.agents.smart_kakaku_signal.agent import registry
            engine = ScenarioExecutionEngine(llm, registry)
            
            # シナリオの実行
            result = await engine.execute_scenario(
                scenario_text=scenario_text,
                parameters=parameters,
                debug=args.debug
            )
            
            # 結果の表示
            print("\n=== 実行結果 ===")
            print(f"異常検出: {'あり' if result['is_anomaly'] else '異常なし' if result['is_anomaly'] is False else '判断保留'}")
            print(f"データ十分性: {'十分' if result['is_data_sufficient'] else '不十分'}")
            print(f"分析: {result['analysis']}")
            
            # デバッグモードの場合は詳細情報も表示
            if args.debug:
                print("\n=== 詳細情報 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except Exception as e:
            logger.error(f"シナリオの実行中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    args_first_parse = argparse.ArgumentParser(add_help=False)
    args_first_parse.add_argument("--log-file", help=argparse.SUPPRESS)
    first_args, _ = args_first_parse.parse_known_args()
    
    # ロギングの設定
    setup_logging(first_args.log_file)
    
    # メイン関数を実行
    asyncio.run(main()) 