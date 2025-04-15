import csv
import os
from typing import Optional, Type, List, Dict, Any, Union

from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel, Field

# 現在のファイルの絶対パスを取得し、そこからデータファイルの絶対パスを構築
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(CURRENT_DIR, "data", "dummy_data.csv")

class CSVSearchInput(BaseModel):
    product_id: str = Field(description="Search for product data by ProductID.")


class CSVSearchTool(BaseTool):
    name: str = "csv_product_search"
    description: str = "Search for product information (ProductName, Price, Quantity) in the CSV data based on ProductID."
    args_schema: Type[BaseModel] = CSVSearchInput

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
                        return f"Found product: Name={row['ProductName']}, Price={row['Price']}, Quantity={row['Quantity']}"
            return f"Product with ID '{product_id}' not found."
        except FileNotFoundError:
            return f"Error: Data file not found at {DATA_FILE_PATH}"
        except Exception as e:
            return f"An error occurred: {e}"

    # 非同期実行が必要な場合は _arun を実装する
    # async def _arun(
    #     self,
    #     product_id: str,
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     # ここでは同期版と同じロジックを非同期で実行する例
    #     # 実際には非同期I/Oライブラリ (例: aiofiles, aiocsv) を使うのが望ましい
    #     return self._run(product_id)

# 拡張された検索ツール入力
class AdvancedCSVSearchInput(BaseModel):
    product_name: Optional[str] = Field(None, description="製品名による検索（部分一致）")
    min_price: Optional[float] = Field(None, description="最低価格フィルター")
    max_price: Optional[float] = Field(None, description="最高価格フィルター")
    min_quantity: Optional[int] = Field(None, description="最小数量フィルター")
    max_quantity: Optional[int] = Field(None, description="最大数量フィルター")
    sort_by: Optional[str] = Field(None, description="結果のソートフィールド（ProductName, Price, Quantity）")
    sort_order: Optional[str] = Field("asc", description="ソート順序：昇順'asc'、降順'desc'")
    limit: Optional[int] = Field(None, description="返す結果の最大数")

class AdvancedCSVSearchTool(BaseTool):
    name: str = "advanced_csv_search"
    description: str = "名前、価格範囲、数量範囲などの高度なフィルターで製品を検索します。結果のソートと制限が可能です。"
    args_schema: Type[BaseModel] = AdvancedCSVSearchInput

    def _run(
        self,
        product_name: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_quantity: Optional[int] = None,
        max_quantity: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
    ) -> str:
        """高度な検索条件で製品を検索する"""
        try:
            print(f"データファイルパス: {DATA_FILE_PATH}")
            with open(DATA_FILE_PATH, mode='r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                all_rows = list(reader)
                
                # フィルタリング処理
                filtered_rows = []
                for row in all_rows:
                    # 数値型に変換
                    row_price = float(row['Price'].replace('$', '').replace(',', ''))
                    row_quantity = int(row['Quantity'])
                    
                    # 各検索条件に基づいてフィルタリング
                    if product_name and product_name.lower() not in row['ProductName'].lower():
                        continue
                    if min_price is not None and row_price < min_price:
                        continue
                    if max_price is not None and row_price > max_price:
                        continue
                    if min_quantity is not None and row_quantity < min_quantity:
                        continue
                    if max_quantity is not None and row_quantity > max_quantity:
                        continue
                    
                    # すべての条件を満たす行を追加
                    filtered_rows.append(row)
                
                # 結果がない場合
                if not filtered_rows:
                    return "指定された条件に一致する製品は見つかりませんでした。"
                
                # ソート処理
                if sort_by:
                    valid_sort_fields = ['ProductName', 'Price', 'Quantity']
                    if sort_by in valid_sort_fields:
                        reverse = sort_order.lower() == 'desc'
                        
                        # 数値フィールドの場合は数値として比較
                        if sort_by == 'Price':
                            filtered_rows.sort(
                                key=lambda x: float(x[sort_by].replace('$', '').replace(',', '')), 
                                reverse=reverse
                            )
                        elif sort_by == 'Quantity':
                            filtered_rows.sort(key=lambda x: int(x[sort_by]), reverse=reverse)
                        else:
                            filtered_rows.sort(key=lambda x: x[sort_by], reverse=reverse)
                
                # 件数制限
                if limit and limit > 0:
                    filtered_rows = filtered_rows[:limit]
                
                # 結果のフォーマット
                results = []
                for row in filtered_rows:
                    results.append(
                        f"ID={row['ProductID']}, 製品名={row['ProductName']}, "
                        f"価格={row['Price']}, 数量={row['Quantity']}"
                    )
                
                formatted_results = "\n".join(results)
                
                summary = (
                    f"検索結果: {len(filtered_rows)}件の製品が見つかりました\n"
                    f"{formatted_results}"
                )
                
                return summary
                
        except FileNotFoundError:
            return f"エラー: データファイルが見つかりません。場所: {DATA_FILE_PATH}"
        except Exception as e:
            return f"エラーが発生しました: {e}"

    # 非同期実行のためのメソッドを実装
    async def _arun(
        self,
        product_name: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_quantity: Optional[int] = None,
        max_quantity: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = "asc",
        limit: Optional[int] = None,
    ) -> str:
        """非同期で高度な検索条件で製品を検索する"""
        # 同期版のメソッドを呼び出す（実際の実装では非同期I/Oライブラリを使用するのが望ましい）
        return self._run(
            product_name=product_name,
            min_price=min_price,
            max_price=max_price,
            min_quantity=min_quantity,
            max_quantity=max_quantity,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit
        )

# 必要に応じて他のツールもここに追加
