import csv
import os
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field

# 現在のファイルの絶対パスを取得し、そこからデータファイルの絶対パスを構築
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(CURRENT_DIR, "data", "market_price_data.csv")

class MarketPriceSearchInput(BaseModel):
    product_id: str = Field(description="Search for market price data by ProductID.")


class MarketPriceSearchTool(BaseTool):
    name: str = "market_price_search"
    description: str = "Search for product market price information (ProductName, RegularPrice, MarketPrice, LastUpdated) based on ProductID."
    args_schema: Type[BaseModel] = MarketPriceSearchInput

    def _run(
        self,
        product_id: str,
    ) -> str:
        """Use the tool."""
        try:
            print(f"データファイルパス: {DATA_FILE_PATH}")
            with open(DATA_FILE_PATH, mode='r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['ProductID'] == product_id:
                        return f"製品情報: 名前={row['ProductName']}, 定価={row['RegularPrice']}円, 市場価格={row['MarketPrice']}円, 最終更新日={row['LastUpdated']}"
            return f"製品ID '{product_id}' は見つかりませんでした。"
        except FileNotFoundError:
            return f"エラー: データファイルが見つかりません: {DATA_FILE_PATH}"
        except Exception as e:
            return f"エラーが発生しました: {e}"

    # 非同期実行が必要な場合は _arun を実装する
    # async def _arun(
    #     self,
    #     product_id: str,
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     # ここでは同期版と同じロジックを非同期で実行する例
    #     # 実際には非同期I/Oライブラリ (例: aiofiles, aiocsv) を使うのが望ましい
    #     return self._run(product_id)

# 必要に応じて他のツールもここに追加 