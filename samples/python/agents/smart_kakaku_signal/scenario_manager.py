import os
import yaml
import datetime
import logging
from typing import Dict, Any, List, Optional

# ロガーの設定
logger = logging.getLogger(__name__)

class ScenarioManager:
    """
    シナリオを管理するクラス
    """
    
    def __init__(self, scenario_directory: str):
        """
        初期化
        
        Args:
            scenario_directory: シナリオファイルを保存するディレクトリパス
        """
        self.scenario_directory = scenario_directory
        os.makedirs(scenario_directory, exist_ok=True)
        logger.info(f"シナリオディレクトリを初期化: {scenario_directory}")
    
    def save_scenario(self, scenario_id: str, name: str, description: str) -> str:
        """
        シナリオを保存する
        
        Args:
            scenario_id: シナリオID
            name: シナリオ名
            description: シナリオの説明（自然言語シナリオ）
            
        Returns:
            str: 保存されたファイルのパス
        """
        file_path = os.path.join(self.scenario_directory, f"{scenario_id}.yaml")
        
        scenario = {
            "scenario_id": scenario_id,
            "name": name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(scenario, file, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"シナリオを保存しました: {scenario_id}")
        return file_path
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        シナリオを取得する
        
        Args:
            scenario_id: シナリオID
            
        Returns:
            Optional[Dict]: シナリオデータ（見つからない場合はNone）
        """
        file_path = os.path.join(self.scenario_directory, f"{scenario_id}.yaml")
        if not os.path.exists(file_path):
            logger.warning(f"シナリオが見つかりません: {scenario_id}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as file:
            scenario = yaml.safe_load(file)
            
        logger.info(f"シナリオを読み込みました: {scenario_id}")
        return scenario
    
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """
        すべてのシナリオを取得する
        
        Returns:
            List[Dict]: シナリオデータのリスト
        """
        scenarios = []
        for filename in os.listdir(self.scenario_directory):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                file_path = os.path.join(self.scenario_directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        scenario = yaml.safe_load(file)
                        scenarios.append(scenario)
                except Exception as e:
                    logger.error(f"シナリオの読み込み中にエラーが発生しました ({filename}): {e}")
                    
        logger.info(f"シナリオを {len(scenarios)}件 読み込みました")
        return scenarios
    
    def delete_scenario(self, scenario_id: str) -> bool:
        """
        シナリオを削除する
        
        Args:
            scenario_id: 削除するシナリオのID
            
        Returns:
            bool: 削除に成功したかどうか
        """
        file_path = os.path.join(self.scenario_directory, f"{scenario_id}.yaml")
        if not os.path.exists(file_path):
            logger.warning(f"削除対象のシナリオが見つかりません: {scenario_id}")
            return False
            
        try:
            os.remove(file_path)
            logger.info(f"シナリオを削除しました: {scenario_id}")
            return True
        except Exception as e:
            logger.error(f"シナリオの削除中にエラーが発生しました: {e}")
            return False 