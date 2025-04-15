import pandas as pd
import json
import re
import csv
import os
from typing import List, Dict, Any, Optional, Union
import asyncio
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# 現在のファイルの絶対パスを取得し、そこからデータファイルの絶対パスを構築
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(CURRENT_DIR, "data", "dummy_data.csv")

class QueryAgent:
    """
    自然言語クエリをデータフレーム操作に変換し、実行するエージェント
    """
    
    def __init__(self, csv_path=DATA_FILE_PATH, model="gpt-3.5-turbo"):
        """
        エージェントの初期化
        
        Args:
            csv_path (str): CSVファイルのパス
            model (str): 使用するLLMモデル
        """
        self.csv_path = csv_path
        self.model = model
        
        # CSVファイルを読み込み
        try:
            print(f"CSVファイルを読み込みます: {csv_path}")
            self.df = pd.read_csv(csv_path)
            print(f"データ読み込み成功: {len(self.df)}行")
            print(f"カラム: {list(self.df.columns)}")
            
            # 数値データの前処理（通貨記号や区切り文字の削除）- エスケープシーケンスを修正
            if 'Price' in self.df.columns:
                # 正規表現を使用する場合は r プレフィックスを使用
                self.df['Price'] = self.df['Price'].astype(str).replace(r'[$,]', '', regex=True).astype(float)
            
            # データの最初の数行を表示して確認
            print("データサンプル:")
            print(self.df.head(3).to_string())
            
        except Exception as e:
            print(f"CSVファイルの読み込みエラー: {e}")
            # 空のデータフレームを作成（エラー回避）
            self.df = pd.DataFrame()
        
        # LLMの初期化
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    async def process_query(self, query: str, **kwargs) -> str:
        """
        自然言語クエリを処理する
        
        Args:
            query (str): 自然言語の問い合わせ
            **kwargs: 追加のパラメータ (例: product_id="P123")
            
        Returns:
            str: 問い合わせ結果
        """
        try:
            # データフレームが空の場合
            if self.df.empty:
                return "データが読み込まれていないため、クエリを実行できません。"
                
            # パラメータの処理
            product_id = kwargs.get('product_id')
            effective_query = query
            
            # product_idパラメータがある場合は、クエリに追加情報を付与
            if product_id:
                # 製品IDを含む列を検索
                id_columns = [col for col in self.df.columns if 'ID' in col or 'id' in col]
                if id_columns:
                    id_column = id_columns[0]  # 最初のID列を使用
                    effective_query = f"{query} (対象製品ID: {product_id})"
                    print(f"製品ID '{product_id}' に関するクエリとして処理します。ID列: {id_column}")
                
            # クエリを操作系列に変換
            operations = await self._translate_query_to_operations(effective_query)
            
            if not operations:
                return "クエリをデータ操作に変換できませんでした。別の表現で試してください。"
            
            # デバッグ用に操作系列を表示
            print(f"実行する操作系列: {json.dumps(operations, indent=2, ensure_ascii=False)}")
            
            # 製品IDに基づくフィルタリング操作を追加（指定がある場合）
            if product_id and id_columns:
                id_column = id_columns[0]
                filter_op = {
                    "operation": "filter",
                    "column": id_column,
                    "condition": "==",
                    "value": product_id
                }
                # フィルタリング操作を先頭に追加
                operations.insert(0, filter_op)
                print(f"製品ID '{product_id}' でフィルタリング操作を追加しました")
            
            # 操作を実行
            result_df, intermediate_results = self._execute_operations(operations)
            
            if result_df.empty:
                return "条件に一致するデータが見つかりませんでした。"
            
            # 結果を自然言語に変換して返す
            return await self._format_results_as_text(query, operations, result_df, intermediate_results)
            
        except Exception as e:
            import traceback
            print(f"エラーのトレースバック: {traceback.format_exc()}")
            return f"エラーが発生しました: {e}"
    
    async def _translate_query_to_operations(self, query):
        """
        自然言語クエリをデータフレーム操作に変換する
        
        Args:
            query (str): 自然言語の問い合わせ
            
        Returns:
            List[Dict]: 操作系列
        """
        # 利用可能な操作とその説明を定義
        available_operations = {
            "filter": "特定の条件に基づいてデータをフィルタリングします",
            "select": "特定の列を選択します",
            "sum": "指定した列の合計を計算します",
            "mean": "指定した列の平均値を計算します",
            "count": "行数または特定の値の出現回数を数えます",
            "max": "指定した列の最大値を取得します",
            "min": "指定した列の最小値を取得します",
            "sort": "指定した列に基づいてデータをソートします",
            "head": "先頭の数行を取得します",
            "tail": "末尾の数行を取得します",
            "group_by": "特定の列でグループ化し、他の列を集計します",
            "join": "別のデータセットと結合します"
        }
        
        # 利用可能な条件演算子とその説明
        available_conditions = {
            "==": "等しい",
            "!=": "等しくない",
            ">": "より大きい",
            ">=": "以上",
            "<": "より小さい",
            "<=": "以下",
            "in": "リスト内に含まれる",
            "not in": "リスト内に含まれない",
            "contains": "文字列を含む",
            "starts_with": "特定の文字列で始まる",
            "ends_with": "特定の文字列で終わる"
        }
        
        # 利用可能な列とそのデータ型
        column_info = {
            "ProductID": "string - 製品の一意識別子 (例: P001)",
            "ProductName": "string - 製品名 (例: 高性能ノートパソコン)",
            "Price": "number - 販売価格（円） (例: 120000)",
            "Quantity": "number - 販売数量 (例: 5)"
        }
        
        prompt = f"""
        以下の自然言語クエリをpandasデータフレーム操作に変換してください。

        データフレーム構造:
        {json.dumps(column_info, indent=2, ensure_ascii=False)}

        利用可能な操作:
        {json.dumps(available_operations, indent=2, ensure_ascii=False)}

        利用可能な条件演算子:
        {json.dumps(available_conditions, indent=2, ensure_ascii=False)}

        クエリ: {query}

        操作系列のルール:
        1. 各操作は正確に定義された形式で記述すること
        2. 使用できる操作は上記の「利用可能な操作」のみ
        3. 条件演算子は上記の「利用可能な条件演算子」のみ使用可能
        4. 出力列名が必要な場合は "output" パラメータを使用
        5. 複数のステップを要する複雑なクエリは、複数の操作に分解すること

        各操作の必須パラメータ:
        - filter: column, condition, value
        - select: columns (配列)
        - sum: column, output (オプション)
        - mean: column, output (オプション)
        - count: column (オプション), output (オプション)
        - max: column, output (オプション)
        - min: column, output (オプション)
        - sort: column, ascending (オプション、デフォルトtrue)
        - head: rows (オプション、デフォルト5)
        - tail: rows (オプション、デフォルト5)
        - group_by: column (グループ化する列), target (集計対象の列), aggregation (集計方法: "sum", "mean", "count", "max", "min"), output (オプション)

        以下のJSON形式で操作系列を返してください:
        ```json
        [
          {{"operation": "filter", "column": "ProductID", "condition": "==", "value": "P001"}},
          {{"operation": "select", "columns": ["ProductID", "ProductName", "Price", "Quantity"]}}
        ]
        ```
        
        group_byの例:
        ```json
        {{"operation": "group_by", "column": "ProductID", "target": "Quantity", "aggregation": "sum", "output": "TotalQuantity"}}
        ```
        """
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": "あなたは自然言語クエリをデータフレーム操作に変換するエキスパートです。与えられた制約条件内で、最適な操作系列を生成してください。"},
            {"role": "user", "content": prompt}
        ])
        
        # レスポンスからJSONを抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        
        if not json_match:
            # JSONが見つからない場合のフォールバック
            return []
        
        operations_str = json_match.group(0) if "```json" not in response.content else json_match.group(1)
        
        try:
            operations = json.loads(operations_str)
            # 操作の検証
            validated_operations = []
            for op in operations:
                operation_type = op.get("operation")
                if operation_type in available_operations:
                    # 操作タイプが有効であれば追加
                    validated_operations.append(op)
                else:
                    print(f"警告: 未知の操作タイプ '{operation_type}' はスキップされます")
            
            return validated_operations
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー: {e}")
            return []
    
    def _execute_operations(self, operations):
        """
        操作系列を実行する
        
        Args:
            operations (List[Dict]): 実行する操作系列
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 結果のデータフレームと中間結果
        """
        # オリジナルのデータフレームをコピー
        result = self.df.copy()
        print(f"初期データフレームのカラム: {list(result.columns)}")
        
        # 中間結果を保存するための辞書
        intermediate_results = {}
        
        # Numpy/Pandasの特殊型をPythonの標準型に変換するヘルパー関数
        def convert_to_standard_type(value):
            if hasattr(value, 'item'):  # np.int64などのスカラー型
                return value.item()
            elif hasattr(value, 'tolist'):  # numpy配列など
                return value.tolist()
            else:
                return value
        
        for i, op in enumerate(operations):
            operation_type = op.get("operation")
            print(f"操作 {i+1}: {operation_type}")
            
            try:
                if operation_type == "filter":
                    column = op.get("column")
                    condition = op.get("condition")
                    value = op.get("value")
                    
                    # 特殊ケース: 中間結果に対するフィルタリング (例: sum_XXXに対するフィルタリング)
                    if column not in result.columns and column in intermediate_results:
                        print(f"中間結果の '{column}' に対してフィルタリングを実行します")
                        intermediate_value = intermediate_results[column]
                        
                        # フィルタリング条件の評価
                        passed_filter = False
                        if condition == "==":
                            passed_filter = intermediate_value == value
                        elif condition == "!=":
                            passed_filter = intermediate_value != value
                        elif condition == ">":
                            passed_filter = intermediate_value > value
                        elif condition == ">=":
                            passed_filter = intermediate_value >= value
                        elif condition == "<":
                            passed_filter = intermediate_value < value
                        elif condition == "<=":
                            passed_filter = intermediate_value <= value
                        
                        print(f"中間結果フィルター条件: {column}({intermediate_value}) {condition} {value} -> 結果: {passed_filter}")
                        
                        # フィルター結果に基づいて処理
                        if not passed_filter:
                            # フィルターを通過しなかった場合は空のデータフレームにする
                            result = pd.DataFrame()
                            print("フィルター条件を満たさないため、結果は空になります")
                        
                        continue  # 中間結果に対するフィルタリングは完了
                    
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    
                    print(f"フィルター前のデータフレーム形状: {result.shape}")
                    
                    # 正しい比較処理
                    if condition == "==":
                        # 等しい
                        mask = result[column] == value
                    elif condition == "!=":
                        # 等しくない
                        mask = result[column] != value
                    elif condition == ">":
                        # より大きい
                        # 文字列の場合は数値に変換
                        compare_value = float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
                        mask = result[column] > compare_value
                    elif condition == ">=":
                        # 以上
                        compare_value = float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
                        mask = result[column] >= compare_value
                    elif condition == "<":
                        # より小さい
                        compare_value = float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
                        mask = result[column] < compare_value
                    elif condition == "<=":
                        # 以下
                        compare_value = float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
                        mask = result[column] <= compare_value
                    elif condition == "in" and isinstance(value, list):
                        # リスト内に含まれる
                        mask = result[column].isin(value)
                    elif condition == "not in" and isinstance(value, list):
                        # リスト内に含まれない
                        mask = ~result[column].isin(value)
                    elif condition == "contains" and isinstance(value, str):
                        # 文字列を含む
                        mask = result[column].astype(str).str.contains(value)
                    elif condition == "starts_with" and isinstance(value, str):
                        # 特定の文字列で始まる
                        mask = result[column].astype(str).str.startswith(value)
                    elif condition == "ends_with" and isinstance(value, str):
                        # 特定の文字列で終わる
                        mask = result[column].astype(str).str.endswith(value)
                    else:
                        print(f"警告: 未知または無効な条件演算子 '{condition}' です")
                        continue
                    
                    # フィルタリングを適用
                    result = result[mask]
                    
                    print(f"フィルター条件: {column} {condition} {value}")
                    print(f"フィルター後のデータフレーム形状: {result.shape}")
                    print(f"フィルター後のカラム: {list(result.columns)}")
                    
                elif operation_type == "select":
                    columns = op.get("columns") if "columns" in op else [op.get("column")]
                    # 存在しない列を選択するとエラーになるので検証
                    valid_columns = [col for col in columns if col in result.columns]
                    if not valid_columns:
                        print(f"警告: 選択された列 {columns} は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    result = result[valid_columns]
                    print(f"選択後のカラム: {list(result.columns)}")
                    
                elif operation_type == "sum":
                    column = op.get("column")
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    output_name = op.get("output", f"sum_{column}")
                    sum_value = result[column].sum()
                    # numpyのint64型などをPythonの標準型に変換
                    intermediate_results[output_name] = convert_to_standard_type(sum_value)
                    # 結果を表形式で保持
                    if len(result) > 1:
                        # 集計操作の場合は1行になる
                        result = pd.DataFrame([{output_name: sum_value}])
                    else:
                        # すでに1行以下の場合は、その行に追加
                        result[output_name] = sum_value
                    
                    print(f"合計計算: {column} の合計は {intermediate_results[output_name]}")
                    
                elif operation_type == "mean":
                    column = op.get("column")
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    output_name = op.get("output", f"mean_{column}")
                    mean_value = result[column].mean()
                    # numpyのfloat64型などをPythonの標準型に変換
                    intermediate_results[output_name] = convert_to_standard_type(mean_value)
                    # 結果を表形式で保持
                    if len(result) > 1:
                        result = pd.DataFrame([{output_name: mean_value}])
                    else:
                        result[output_name] = mean_value
                    
                    print(f"平均計算: {column} の平均は {intermediate_results[output_name]}")
                    
                elif operation_type == "count":
                    column = op.get("column", None)
                    # 列が指定されている場合は存在を確認
                    if column and column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    output_name = op.get("output", "count")
                    if column:
                        count_value = result[column].count()
                    else:
                        count_value = len(result)
                    # 標準型に変換
                    intermediate_results[output_name] = convert_to_standard_type(count_value)
                    result = pd.DataFrame([{output_name: count_value}])
                    
                    print(f"カウント計算: 結果は {intermediate_results[output_name]}")
                    
                elif operation_type == "max":
                    column = op.get("column")
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    output_name = op.get("output", f"max_{column}")
                    max_value = result[column].max()
                    # 標準型に変換
                    intermediate_results[output_name] = convert_to_standard_type(max_value)
                    if len(result) > 1:
                        result = pd.DataFrame([{output_name: max_value}])
                    else:
                        result[output_name] = max_value
                    
                    print(f"最大値計算: {column} の最大値は {intermediate_results[output_name]}")
                    
                elif operation_type == "min":
                    column = op.get("column")
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    output_name = op.get("output", f"min_{column}")
                    min_value = result[column].min()
                    # 標準型に変換
                    intermediate_results[output_name] = convert_to_standard_type(min_value)
                    if len(result) > 1:
                        result = pd.DataFrame([{output_name: min_value}])
                    else:
                        result[output_name] = min_value
                    
                    print(f"最小値計算: {column} の最小値は {intermediate_results[output_name]}")
                    
                elif operation_type == "sort":
                    column = op.get("column")
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: 列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    ascending = op.get("ascending", True)
                    result = result.sort_values(by=column, ascending=ascending)
                    
                elif operation_type == "head":
                    rows = op.get("rows", 5)
                    result = result.head(rows)
                    
                elif operation_type == "tail":
                    rows = op.get("rows", 5)
                    result = result.tail(rows)
                    
                elif operation_type == "group_by":
                    column = op.get("column")
                    
                    # パラメータ名の互換性対応
                    # 旧形式: agg_column, agg_func
                    # 新形式: target, aggregation
                    agg_column = op.get("target", op.get("agg_column"))
                    agg_func = op.get("aggregation", op.get("agg_func", "sum"))
                    output_name = op.get("output")
                    
                    # 列の存在を確認
                    if column not in result.columns:
                        print(f"警告: グループ化の列 '{column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    
                    if agg_column and agg_column not in result.columns:
                        print(f"警告: 集計の列 '{agg_column}' は存在しません。利用可能な列: {list(result.columns)}")
                        continue
                    
                    print(f"グループ化: {column} でグループ化し、{agg_column} を {agg_func} で集計します")
                    
                    # 集計を実行
                    if agg_func == "sum":
                        grouped = result.groupby(column)[agg_column].sum()
                    elif agg_func == "mean":
                        grouped = result.groupby(column)[agg_column].mean()
                    elif agg_func == "count":
                        grouped = result.groupby(column)[agg_column].count()
                    elif agg_func == "max":
                        grouped = result.groupby(column)[agg_column].max()
                    elif agg_func == "min":
                        grouped = result.groupby(column)[agg_column].min()
                    else:
                        print(f"警告: 未知または無効な集計関数 '{agg_func}' です")
                        continue
                    
                    # 結果をデータフレームに変換
                    if output_name:
                        # 出力列名が指定されている場合はリネーム
                        grouped = grouped.reset_index()
                        grouped.columns = [column, output_name]
                    else:
                        # 指定がない場合は自動生成
                        output_name = f"{agg_func}_{agg_column}"
                        grouped = grouped.reset_index()
                        grouped.columns = [column, output_name]
                    
                    result = grouped
                    
                    # 中間結果にも保存（個別の値に対するフィルタリングのため）
                    for idx, row in grouped.iterrows():
                        key = f"{output_name}_{row[column]}"
                        value = row[output_name]
                        intermediate_results[key] = convert_to_standard_type(value)
                    
                    print(f"グループ化後のカラム: {list(result.columns)}")
                    print(f"グループ化後のデータ形状: {result.shape}")
                    
                elif operation_type == "compare":
                    left = op.get("left")
                    right = op.get("right")
                    condition = op.get("condition")
                    output_name = op.get("output", "comparison_result")
                    
                    # 左辺と右辺の値を取得 (変数名、数値またはその演算)
                    left_value = intermediate_results.get(left, left)
                    if isinstance(right, str) and any(oper in right for oper in ['+', '-', '*', '/']):
                        # 演算式を評価
                        # 変数名を値に置き換え
                        for var_name, var_value in intermediate_results.items():
                            right = right.replace(var_name, str(var_value))
                        right_value = eval(right)
                    else:
                        right_value = intermediate_results.get(right, right)
                    
                    # 比較実行
                    if condition == "==":
                        comparison_result = left_value == right_value
                    elif condition == "!=":
                        comparison_result = left_value != right_value
                    elif condition == ">":
                        comparison_result = left_value > right_value
                    elif condition == ">=":
                        comparison_result = left_value >= right_value
                    elif condition == "<":
                        comparison_result = left_value < right_value
                    elif condition == "<=":
                        comparison_result = left_value <= right_value
                    
                    # 比較結果を保存 (標準型に変換)
                    intermediate_results[output_name] = bool(comparison_result)  # 確実にbool型にする
                    result = pd.DataFrame([{
                        "left_value": convert_to_standard_type(left_value),
                        "right_value": convert_to_standard_type(right_value),
                        "condition": condition,
                        output_name: bool(comparison_result)
                    }])
            
            except Exception as e:
                import traceback
                print(f"操作 '{operation_type}' の実行中にエラーが発生しました: {e}")
                print(f"詳細なエラー: {traceback.format_exc()}")
                # エラーが発生しても処理を続行
        
        # 最終結果を表示
        print(f"最終結果のカラム: {list(result.columns)}")
        print(f"最終結果の形状: {result.shape}")
        print("最終結果のサンプル:")
        print(result.head(3).to_string())
        
        # 中間結果も返す（デバッグや詳細な説明のため）
        return result, intermediate_results
    
    async def _format_results_as_text(self, query, operations, result_df, intermediate_results):
        """
        結果を自然言語のテキストに変換する
        
        Args:
            query (str): 元のクエリ
            operations (List[Dict]): 実行された操作系列
            result_df (pd.DataFrame): 結果のデータフレーム
            intermediate_results (Dict): 中間結果の辞書
            
        Returns:
            str: フォーマットされたテキスト結果
        """
        # JSON変換用のヘルパー関数
        def default_converter(obj):
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif hasattr(obj, 'tolist'):  # numpy配列やint64などの型
                return obj.tolist()
            elif hasattr(obj, 'item'):   # np.int64などのスカラー型
                return obj.item()
            else:
                # その他の型は文字列に変換
                return str(obj)
                
        # 結果の内容を取得
        result_str = result_df.to_string(index=False)
        num_rows = len(result_df)
        num_cols = len(result_df.columns)
        
        # 中間結果の値をJSON変換可能な型に変換
        processed_intermediate_results = {}
        for key, value in intermediate_results.items():
            if hasattr(value, 'item'):  # np.int64などのスカラー型
                processed_intermediate_results[key] = value.item()
            elif hasattr(value, 'tolist'):  # numpy配列など
                processed_intermediate_results[key] = value.tolist()
            else:
                processed_intermediate_results[key] = value
        
        prompt = f"""
        以下の情報を使用して、クエリへの回答を自然で読みやすい日本語で生成してください。
        
        元のクエリ: {query}
        
        実行された操作:
        {json.dumps(operations, indent=2, ensure_ascii=False, default=default_converter)}
        
        結果 ({num_rows}行 x {num_cols}列):
        {result_str}
        
        中間結果（もしあれば）:
        {json.dumps(processed_intermediate_results, indent=2, ensure_ascii=False, default=default_converter)}
        
        回答は以下のガイドラインに従ってください:
        1. 単なるデータの羅列ではなく、質問に対する直接的な回答を提供する
        2. 重要な数値や結果は強調する
        3. 結果がない場合や条件に一致するデータがない場合はその旨を説明する
        4. 必要に応じて、関連する追加情報や示唆も提供する
        5. 回答は簡潔で明確であること
        6. 専門用語や技術的な説明は避け、一般的な言葉で説明する
        
        回答の最大長は300文字程度に抑えてください。
        """
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": "あなたはデータ分析結果を自然な日本語で説明するエキスパートです。技術的な詳細を一般の人にもわかりやすく伝えることが得意です。"},
            {"role": "user", "content": prompt}
        ])
        
        # 必要に応じてレスポンスを整形
        result_text = response.content.strip()
        return result_text

# テスト用関数
async def test_query_agent():
    agent = QueryAgent()
    
    test_queries = [
        "製品ID P001 の詳細情報を教えてください",
        "5000円以下の製品を探してください",
        "在庫数が最も多い製品は何ですか？",
        "ノートパソコンの価格はいくらですか？",
        "10000円から20000円の間の製品で、在庫が10個以上あるものを教えてください",
        "P001の平均価格を教えて"
    ]
    
    for query in test_queries:
        print(f"\n==== クエリ: {query} ====")
        result = await agent.process_query(query)
        print(result)
    
    # 対話モード
    print("\n==== 対話モード ====")
    print("質問を入力してください。終了するには 'exit' と入力してください。")
    
    while True:
        user_query = input("\n質問: ")
        if user_query.lower() == "exit":
            print("プログラムを終了します。")
            break
        
        result = await agent.process_query(user_query)
        print(result)

# モジュールが直接実行された場合のエントリーポイント
if __name__ == "__main__":
    asyncio.run(test_query_agent()) 