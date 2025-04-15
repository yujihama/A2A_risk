import csv
import os
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field

# 現在のファイルの絶対パスを取得し、そこからデータファイルの絶対パスを構築
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(CURRENT_DIR, "data", "inventory_data.csv")

class InventorySearchInput(BaseModel):
    product_id: str = Field(description="Search for inventory data by ProductID.")


class InventorySearchTool(BaseTool):
    name: str = "inventory_search"
    description: str = "Search for inventory information (Location, StockQuantity, ReorderLevel, LastUpdated) based on ProductID."
    args_schema: Type[BaseModel] = InventorySearchInput

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
                        return (f"在庫情報: 商品ID={product_id}, 商品名={row['ProductName']}, "
                               f"保管場所={row['Location']}, 在庫数={row['StockQuantity']}, "
                               f"再注文レベル={row['ReorderLevel']}, 最終更新日={row['LastUpdated']}")
            return f"商品ID '{product_id}' の在庫情報は見つかりませんでした。"
        except FileNotFoundError:
            return f"エラー: データファイルが {DATA_FILE_PATH} に見つかりません。"
        except Exception as e:
            return f"エラーが発生しました: {e}"

    # 非同期実行が必要な場合は _arun を実装する
    # async def _arun(
    #     self,
    #     product_id: str,
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     return self._run(product_id) 