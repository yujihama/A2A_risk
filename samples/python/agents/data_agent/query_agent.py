import pandas as pd
import json
import re
import csv
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import numpy as np
import logging

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# .envファイルから環境変数を読み込む
load_dotenv()

class QueryAgent:
    """
    自然言語クエリをデータフレーム操作に変換し、実行する汎用エージェント
    """
    
    def __init__(self, model="gpt-4o-mini"):
        """
        エージェントの初期化
        
        Args:
            model (str): 使用するLLMモデル
        """
        self.model = model
        self.df = pd.DataFrame() # 初期状態は空のDataFrame
        self.column_info = {} # 列情報を格納する辞書
        self.data_source_info = None # データソース情報を保持 (オプション)

        # LLMの初期化
        try:
            self.llm = ChatOpenAI(model=model, temperature=0)
            logger.info(f"LLMクライアントを初期化しました: model={model}")
        except Exception as e:
            logger.error(f"LLMクライアントの初期化に失敗しました: {e}")
            self.llm = None # 初期化失敗

    def load_data(self, data_source: Any):
        """
        指定されたデータソースからデータを読み込み、データフレームと列情報を設定する

        Args:
            data_source (Any): データソース。ファイルパス(str)、pandas DataFrame、または辞書(DB接続情報など)。
        """
        self.data_source_info = data_source # Save data source info
        logger.info(f"データソースの読み込みを開始します: {type(data_source)}")
        try:
            if isinstance(data_source, str):
                # ファイルパスの場合
                file_path = data_source
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

                logger.info(f"ファイルを読み込みます: {file_path}")
                # ファイル拡張子に基づいて読み込み方法を選択
                if file_path.lower().endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                     self.df = pd.read_excel(file_path)
                # 他のファイル形式 (json, parquetなど) のサポートを追加可能
                # elif file_path.lower().endswith('.json'):
                #     self.df = pd.read_json(file_path)
                # elif file_path.lower().endswith('.parquet'):
                #     self.df = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"サポートされていないファイル形式です: {file_path}")

            elif isinstance(data_source, pd.DataFrame):
                # DataFrameオブジェクトの場合
                logger.info("pandas DataFrameを直接読み込みます")
                self.df = data_source.copy() # 念のためコピー
            # 辞書形式のデータソース (例: DB接続情報) のサポートを追加可能
            # elif isinstance(data_source, dict) and 'db_connection_string' in data_source:
            #     logger.info(f"データベースから読み込みます: {data_source.get('table_name', 'N/A')}")
            #     # ここにDB読み込みロジックを実装 (例: SQLAlchemyを使用)
            #     # conn_str = data_source['db_connection_string']
            #     # query = data_source.get('query', f"SELECT * FROM {data_source.get('table_name')}")
            #     # engine = create_engine(conn_str)
            #     # self.df = pd.read_sql(query, engine)
            #     raise NotImplementedError("データベースからの読み込みはまだ実装されていません")
            else:
                raise TypeError("サポートされていないデータソース形式です。ファイルパス(str)、pandas DataFrame、または対応する辞書を指定してください。")

            if self.df.empty:
                 logger.warning("読み込んだデータが空です。処理を続行しますが、結果は得られない可能性があります。")
                 self.column_info = {}
                 return # 空の場合は以降の処理をスキップ

            logger.info(f"データ読み込み成功: {len(self.df)}行 x {len(self.df.columns)}列")
            logger.debug(f"カラム一覧: {list(self.df.columns)}")

            # データ型の推定と数値データの前処理を試みる
            self._preprocess_data()

            # 列情報を動的に生成
            self.column_info = self._generate_column_info()
            logger.info("列情報の生成完了")
            # logger.debug(f"生成された列情報:\n{json.dumps(self.column_info, indent=2, ensure_ascii=False)}") # Original line causing error
            # Safer alternative for f-string:
            try:
                column_info_json_str = json.dumps(self.column_info, indent=2, ensure_ascii=False)
                logger.debug(f"生成された列情報:\n{column_info_json_str}")
            except TypeError as e:
                 logger.error(f"カラム情報のJSON変換中にエラー: {e}")
                 logger.debug(f"Raw column info: {self.column_info}")

            # データの最初の数行を表示して確認 (Debugレベル)
            logger.debug(f"データサンプル:{self.df.head(3).to_string()}")

        except FileNotFoundError as e:
            logger.error(f"データ読み込みエラー: {e}")
            self.df = pd.DataFrame()
            self.column_info = {}
            raise # エラーを再送出して呼び出し元に通知
        except ValueError as e:
            logger.error(f"データ読み込みエラー: {e}")
            self.df = pd.DataFrame()
            self.column_info = {}
            raise
        except TypeError as e:
            logger.error(f"データ読み込みエラー: {e}")
            self.df = pd.DataFrame()
            self.column_info = {}
            raise
        except ImportError as e:
             logger.error(f"データ読み込みに必要なライブラリが不足しています: {e}")
             self.df = pd.DataFrame()
             self.column_info = {}
             raise
        except Exception as e:
            logger.error(f"予期せぬデータ読み込みエラー: {e}", exc_info=True)
            self.df = pd.DataFrame()
            self.column_info = {}
            raise

    def _preprocess_data(self):
        """データフレームのデータ型を推定し、数値列のクリーンアップを試みる"""
        logger.info("データ型の前処理を開始します...")
        for col in self.df.columns:
            # 全てがNaNまたはNoneのカラムはスキップ
            if self.df[col].isnull().all():
                logger.debug(f"列 '{col}' は全て欠損値のため、前処理をスキップします。")
                continue

            # object型で、数値に変換できそうな列を試す
            if self.df[col].dtype == 'object':
                try:
                    # 通貨記号やカンマを除去
                    # 先に .astype(str) を適用して NaN を文字列 'nan' にする
                    series_str = self.df[col].astype(str)
                    # '$', ',', '¥', '€' などの通貨記号を除去
                    cleaned_series = series_str.replace(r'[$,¥€]', '', regex=True)
                    # 全角数字や他の可能性のある文字も考慮する場合 (オプション)
                    # cleaned_series = cleaned_series.str.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})) # 全角->半角

                    # 数値に変換可能かチェック (空文字列や変換不能な文字列を除く)
                    numeric_check = pd.to_numeric(cleaned_series, errors='coerce')

                    # NaNでない割合を計算 (NaNを除外して計算)
                    notna_ratio = numeric_check.notna().sum() / self.df[col].notna().sum() if self.df[col].notna().sum() > 0 else 0

                    if notna_ratio > 0.8: # 元々非NaNだった値の8割以上が数値に変換できれば変換を試みる
                         # エラーを無視して変換し、変換できなかったものは NaN のままにする
                        self.df[col] = pd.to_numeric(cleaned_series, errors='coerce')
                        logger.info(f"列 '{col}' を数値型 (float64) に変換しました。({notna_ratio*100:.1f}% 変換成功)")
                    # 日付っぽい文字列を日付型に変換する処理などもここに追加可能
                    elif pd.to_datetime(self.df[col], errors='coerce').notna().mean() > 0.8:
                        try:
                            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                            logger.info(f"列 '{col}' を日付型に変換しました。")
                        except Exception as date_err:
                            logger.warning(f"列 '{col}' の日付型への変換中にエラーが発生しました: {date_err}")

                except Exception as e:
                    # 変換に失敗してもエラーとせず、元の型のままにする
                    logger.warning(f"列 '{col}' の数値型への変換中にエラーが発生しましたが、処理を続行します: {e}")
                    pass # 他の列の処理を続ける

            # 数値型だが object として読み込まれた可能性のあるものを変換 (通常read_csv等で発生しにくいが一応)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                 # すでに数値型の場合は何もしないか、必要なら型を統一する (例: int64 -> float64)
                 # ここでは何もしない
                 pass
            # カテゴリ型などの場合
            # elif pd.api.types.is_categorical_dtype(self.df[col]):
            #    logger.debug(f"列 '{col}' はカテゴリ型です。")

        logger.info("データ型の前処理が完了しました。")
        logger.debug(f"処理後のデータ型:{self.df.dtypes}")

    def _generate_column_info(self) -> Dict[str, str]:
        """データフレームから列名とデータ型の情報を生成する"""
        if self.df.empty:
            return {}

        col_info = {}
        logger.info("カラム情報の生成を開始します...")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            dtype_str = str(dtype)
            description = ""
            sample_values_str = ""

            try:
                # データ型に応じた説明を追加
                if pd.api.types.is_numeric_dtype(dtype):
                    description = "number (数値)"
                    # サンプル値を追加して具体例を示す (NaNを除外)
                    sample_values = self.df[col].dropna().unique()[:3] # ユニークな値のサンプル
                    if len(sample_values) > 0:
                        sample_values_str = f" (例: {', '.join(map(lambda x: f'{x:.2f}' if isinstance(x, float) else str(x), sample_values))})"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                     description = "datetime (日付時刻)"
                     # サンプル値を追加 (NaNを除外, NaTも除外してフォーマット)
                     valid_dates = self.df[col].dropna()
                     valid_dates = valid_dates[pd.notna(valid_dates)]
                     sample_values = valid_dates.unique()[:3]
                     if len(sample_values) > 0:
                        # numpy.datetime64をTimestampに変換してからstrftimeを使う
                        sample_values_str = f" (例: {', '.join(pd.to_datetime(sample_values).strftime('%Y-%m-%d').tolist())})"
                elif pd.api.types.is_categorical_dtype(dtype):
                    description = "category (カテゴリ)"
                    sample_values = self.df[col].dropna().unique()[:3]
                    if len(sample_values) > 0:
                        sample_values_str = f" (例: {', '.join(map(str, sample_values))})"
                elif dtype.name == 'bool':
                    description = "boolean (真偽値)"
                    sample_values = self.df[col].dropna().unique()[:2] # True/Falseの例
                    if len(sample_values) > 0:
                        sample_values_str = f" (例: {', '.join(map(str, sample_values))})"
                else: # object型など
                    description = "string (文字列)"
                     # ユニーク値のサンプル (NaNを除外し、文字列に変換)
                    sample_values = [str(v) for v in self.df[col].dropna().unique()[:3]]
                    if len(sample_values) > 0:
                        # 長すぎるサンプル値は省略
                        sample_values_str = f" (例: {', '.join(v[:20] + '...' if len(v) > 20 else v for v in sample_values)})"

                col_info[col] = f"{dtype_str} - {description}{sample_values_str}"
            except Exception as e:
                 logger.warning(f"列 '{col}' の情報生成中にエラー: {e}. スキップします。")
                 col_info[col] = f"{dtype_str} - (情報生成エラー)"

        logger.info(f"{len(col_info)}個のカラム情報を生成しました。")
        return col_info

    async def process_query(self, query: str, **kwargs) -> str:
        """
        自然言語クエリを処理する。追加のパラメータは kwargs で受け取る。
        
        Args:
            query (str): 自然言語の問い合わせ
            **kwargs: 追加のパラメータ (例: product_id="P123")
            
        Returns:
            str: 問い合わせ結果
        """
        logger.info(f"クエリ処理を開始: '{query}'")
        logger.debug(f"追加パラメータ: {kwargs}")

        if self.llm is None:
             logger.error("LLMクライアントが初期化されていません。クエリを処理できません。")
             return "エラー: LLMクライアントが利用できません。"

        try:
            # データフレームが空の場合
            if self.df.empty:
                logger.warning("データがロードされていないため、クエリを実行できません。")
                return "データが読み込まれていないため、クエリを実行できません。load_data()を呼び出してデータを読み込んでください。"

            # kwargs から特定のパラメータを抽出 (例: product_id)
            product_id = kwargs.get("product_id")
            effective_query = query # 元のクエリを保持しつつ、情報を付加

            # 追加パラメータをクエリ文脈に追加する処理 (オプション)
            if product_id:
                 # product_id を持つ列名を動的に探す (より頑健な方法)
                 product_id_col = next((col for col in self.df.columns if 'ID' in col.upper()), None) # 'ID'を含む列を探す
                 if product_id_col:
                     # クエリに直接情報を付加するのではなく、LLMへの指示やコンテキストとして与える方が良い場合もある
                     # ここではクエリ文字列にヒントとして追加
                     effective_query = f"{query} (対象の{product_id_col}: {product_id})"
                     logger.info(f"{product_id_col} '{product_id}' を考慮したクエリ: '{effective_query}'")
                 else:
                     logger.warning(f"product_id ('{product_id}') が指定されましたが、ID関連の列がデータフレームに見つかりません。product_id は無視される可能性があります。")
            # 他の kwargs パラメータも同様に処理可能
                
            # クエリを操作系列に変換
            operations = await self._translate_query_to_operations(effective_query)
            
            if not operations:
                logger.warning(f"クエリをデータ操作に変換できませんでした: '{query}'")
                return "クエリをデータ操作に変換できませんでした。別の表現で試してください。"
            
            logger.info(f"生成された操作系列: {len(operations)} ステップ")
            logger.debug(f"操作系列詳細:{json.dumps(operations, indent=2, ensure_ascii=False)}")
            
            # 操作を実行
            result_df, intermediate_results = self._execute_operations(operations)
            
            # result_dfがNoneの場合や空の場合の処理
            if result_df is None or result_df.empty:
                logger.info("クエリの条件に一致するデータが見つかりませんでした。")
                return "条件に一致するデータが見つかりませんでした。"
            
            logger.info(f"操作実行完了。最終結果: {len(result_df)}行 x {len(result_df.columns)}列")
            
            # 結果を自然言語に変換して返す
            formatted_result = await self._format_results_as_text(query, operations, result_df, intermediate_results)
            logger.info("結果のフォーマット完了")
            logger.debug(f"フォーマットされた結果:{formatted_result}")
            return formatted_result
            
        except ValueError as e: # データ未ロードなどの予測可能なエラー
            logger.error(f"クエリ処理中にエラーが発生しました: {e}")
            return f"エラー: {e}"
        except Exception as e:
            logger.error(f"クエリ処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return f"予期せぬエラーが発生しました: {e}"
    
    async def _translate_query_to_operations(self, query: str) -> Optional[List[Dict]]:
        """
        自然言語クエリをデータフレーム操作のJSON配列に変換する
        
        Args:
            query (str): 自然言語の問い合わせ
            
        Returns:
            Optional[List[Dict]]: 操作系列、または変換失敗時にNone
        """
        # 利用可能な操作とその説明を定義
        available_operations = {
            "filter": "特定の条件に基づいてデータをフィルタリングします (例: 価格が10000円以上)",
            "select": "特定の列を選択します (例: 製品名と価格のみ表示)",
            "sum": "指定した列の合計を計算します (例: 売上の合計)",
            "mean": "指定した列の平均値を計算します (例: 平均価格)",
            "count": "行数または特定の値の出現回数を数えます (例: 製品の総数、特定のカテゴリの製品数)",
            "max": "指定した列の最大値を取得します (例: 最高価格)",
            "min": "指定した列の最小値を取得します (例: 最低在庫数)",
            "sort": "指定した列に基づいてデータをソートします (例: 価格が高い順に並べる)",
            "head": "先頭の数行を取得します (例: 上位5件)",
            "tail": "末尾の数行を取得します (例: 下位3件)",
            "group_by": "特定の列でグループ化し、他の列を集計します (例: カテゴリごとの平均価格)",
            # "join": "別のデータセットと結合します" # 高度な操作は必要に応じて追加
        }
        
        # 利用可能な条件演算子とその説明
        available_conditions = {
            "==": "等しい (例: column == value)",
            "!=": "等しくない (例: column != value)",
            ">": "より大きい (例: column > value)",
            ">=": "以上 (例: column >= value)",
            "<": "より小さい (例: column < value)",
            "<=": "以下 (例: column <= value)",
            "in": "リスト内に含まれる (例: column in [value1, value2])",
            "not in": "リスト内に含まれない (例: column not in [value1, value2])",
            "contains": "文字列を含む (大文字小文字を区別しない部分一致, 例: column contains 'keyword')",
            "not contains": "文字列を含まない (大文字小文字を区別しない, 例: column not contains 'keyword')",
            "starts_with": "特定の文字列で始まる (例: column starts_with 'prefix')",
            "ends_with": "特定の文字列で終わる (例: column ends_with 'suffix')",
            "isna": "欠損値である (null/NaN, value不要)",
            "notna": "欠損値でない (null/NaNではない, value不要)"
        }

        # データフレームがロードされていない場合のエラーハンドリング
        if not self.column_info:
             logger.error("_translate_query_to_operations 呼び出し時に column_info が空です。")
             # ここでは例外を発生させず、Noneを返すことで process_query 側でハンドリングさせることも可能
             raise ValueError("データがロードされていないか、カラム情報の生成に失敗しました。")
        
        prompt = f"""
以下の自然言語クエリをpandasデータフレーム操作のJSON配列に変換してください。
応答は以下のようなJSON配列形式で返してください:
[
  {{ "operation": "※利用可能なoperationから選択", "column": "※利用可能なデータフレームのcolumnから選択", "condition": "※利用可能なconditionから選択", "value":  }},
  {{ "operation": "", "column": "", "ascending":  }},
  ...
]

- クエリ: {query}
- 対象のデータフレームの利用可能なcolumn（データフレーム構造):
{json.dumps(self.column_info, indent=2, ensure_ascii=False)}
- 利用可能なoperation:
        {json.dumps(available_operations, indent=2, ensure_ascii=False)}
- 利用可能なcondition:
        {json.dumps(available_conditions, indent=2, ensure_ascii=False)}

- operationのルール:
1. 応答はJSON配列のみを含むこと。他のテキスト (説明文など) は含めないでください。
2. 各操作は正確に定義された形式で記述すること。
3. 使用できるoperationは上記の「利用可能なoperation」リストから選択すること。
4. 使用できるconditionは上記の「利用可能なcondition」リストから選択すること。
5. filter操作で数値比較を行う場合、valueは必ず数値型か日付型のカラムを選択すること。
6. filter操作で文字列比較を行う場合、valueは文字列型 (例: "Laptop") にすること。大文字小文字を区別しない比較は 'contains'/'not contains' を使用する。
7. filter操作で 'in'/'not in' を使う場合、valueはJSON配列形式 (例: ["A", "B"]) にすること。
8. filter操作で 'isna'/'notna' を使う場合、"value"キーは含めないこと。
9. 列名や値に日本語が含まれる場合は、そのまま日本語を使用すること。
10. 複数のステップを要する複雑なクエリは、複数の操作オブジェクトを配列内に記述すること。
11. 出力列名が必要な場合 (sum, mean, count, max, min, group_by)、"output" パラメータを指定できる (任意)。指定しない場合はデフォルト名が使用される。

- 各operationの必須パラメータと形式:
-- filter: {{ "operation": "filter", "column": "<数値/日付列名>", "condition": "<条件演算子>", ["value": <値 または リスト>] }} # isna/notnaではvalue不要
-- select: {{ "operation": "select", "columns": ["<列名1>", "<列名2>"] }}
-- sum: {{ "operation": "sum", "column": "<数値列名>", ["output": "<出力列名>"] }}
-- mean: {{ "operation": "mean", "column": "<数値列名>", ["output": "<出力列名>"] }}
-- count: {{ "operation": "count", ["column": "<列名>"], ["output": "<出力列名>"] }} # columnなしで行数カウント
-- max: {{ "operation": "max", "column": "<比較可能列名>", ["output": "<出力列名>"] }}
-- min: {{ "operation": "min", "column": "<比較可能列名>", ["output": "<出力列名>"] }}
-- sort: {{ "operation": "sort", "column": "<列名>", ["ascending": <true/false>] }} # ascending デフォルト true
-- head: {{ "operation": "head", ["rows": <行数>] }} # rows デフォルト 5
-- tail: {{ "operation": "tail", ["rows": <行数>] }} # rows デフォルト 5
-- group_by: {{ "operation": "group_by", "column": "<グループ化列名 または 列名リスト>", "target": "<集計対象列名>", "aggregation": "<sum/mean/count/max/min>", ["output": "<出力列名>"] }} # 複数集計は現状未サポート

"""
        logger.info(f"LLMプロンプト生成完了 (_translate_query_to_operations):{prompt[:500]}...")

        if not self.llm:
             logger.error("LLMクライアントが利用できません。操作変換をスキップします。")
             return None

        try:
            response = await self.llm.ainvoke([
                    # システムプロンプトを強化
                    {"role": "system", "content": "あなたは自然言語クエリをpandasデータフレーム操作のJSON配列に正確に変換するエキスパートです。与えられたデータフレーム構造、利用可能な操作、条件演算子、ルールに厳密に従ってください。応答にはJSON配列以外のテキストを絶対に含めないでください。"},
                    {"role": "user", "content": prompt}
            ])
            response_content = response.content.strip()
            logger.info(f"LLM Raw Response (_translate_query_to_operations):{response_content}")

        except Exception as e:
            logger.error(f"LLM呼び出し中にエラーが発生しました (_translate_query_to_operations): {e}", exc_info=True)
            return None

        # レスポンスからJSONを抽出するロジックを改善
        json_str = None
        # ```json ... ``` ブロックを探す
        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_content, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug("JSONコードブロックから抽出しました。")
        else:
            # ``` がない場合は、レスポンス全体がJSON配列であると仮定する
            if response_content.strip().startswith('[') and response_content.strip().endswith(']'):
                 json_str = response_content
                 logger.debug("応答全体がJSON配列であると仮定して抽出しました。")
            else:
                 logger.warning(f"LLMからの応答に有効なJSONコードブロックまたはJSON配列が見つかりませんでした:{response_content}")
                 # フォールバックとして、応答全体を試す (より寛容だがリスクあり)
                 # json_str = response_content
                 return None

        if not json_str:
             logger.warning("抽出されたJSON文字列が空です。")
             return None

        try:
            operations = json.loads(json_str)
            # 簡単なバリデーション
            if not isinstance(operations, list):
                logger.warning(f"抽出されたJSONがリスト形式ではありません: {type(operations)}")
                return None
            if not operations: # 空のリストの場合
                logger.warning("抽出されたJSON操作リストが空です。")
                return None

            # 各操作の形式を軽くチェック (修正前)
            for i, op in enumerate(operations):
                if not isinstance(op, dict) or "operation" not in op:
                    logger.warning(f"不正な操作形式です (ステップ {i+1}): {op}")
                    return None # 不正な形式が含まれていたら全体を無効とする

            # --- カラム名の検証と修正処理を追加 ---
            validated_operations = await self._validate_and_correct_operations(query, operations)
            if validated_operations is None:
                 logger.warning("カラム名の検証または修正に失敗しました。操作を実行できません。")
                 return None # 検証/修正失敗

            logger.info(f"カラム名の検証/修正完了。最終的な操作リスト: {len(validated_operations)} ステップ")
            return validated_operations
            # --- ここまで追加 ---

        except json.JSONDecodeError as e:
            logger.error(f"LLMからの応答のJSON解析エラー: {e}")
            logger.debug(f"解析しようとしたJSON文字列: {json_str}")
            return None
        except Exception as e:
             logger.error(f"操作リストの処理中に予期せぬエラー: {e}", exc_info=True)
             return None

    # --- 新しいメソッドを追加 ---
    async def _validate_and_correct_operations(self, original_query: str, operations: List[Dict]) -> Optional[List[Dict]]:
        """操作リストのカラム名を検証し、不正な場合はLLMに修正を試みる"""
        logger.info("生成された操作リストのカラム名検証を開始します...")
        invalid_columns_found = set() # 重複を避けるためセットを使用
        available_columns = set(self.column_info.keys())

        for i, op in enumerate(operations):
            step_num = i + 1
            op_type = op.get("operation")
            columns_to_check = []

            if op_type == "filter":
                if "column" in op and isinstance(op["column"], str): columns_to_check.append(op["column"])
            elif op_type == "select":
                if "columns" in op and isinstance(op["columns"], list):
                     columns_to_check.extend([col for col in op["columns"] if isinstance(col, str)])
            elif op_type in ["sum", "mean", "max", "min", "count", "sort"]:
                if "column" in op and isinstance(op["column"], str): columns_to_check.append(op["column"])
            elif op_type == "group_by":
                if "column" in op:
                     if isinstance(op["column"], str):
                         columns_to_check.append(op["column"])
                     elif isinstance(op["column"], list):
                         columns_to_check.extend([col for col in op["column"] if isinstance(col, str)])
                if "target" in op and isinstance(op["target"], str): columns_to_check.append(op["target"])
            # 他の操作タイプがあれば追加

            for col in columns_to_check:
                if col not in available_columns:
                    logger.warning(f"ステップ {step_num} ({op_type}): 存在しないカラム '{col}' が指定されています。")
                    invalid_columns_found.add(col)

        if not invalid_columns_found:
            logger.info("カラム名の検証完了。不正なカラムは見つかりませんでした。")
            return operations # 問題なければ元の操作リストを返す

        # 不正なカラムが見つかった場合、LLMに修正を依頼
        logger.warning(f"存在しないカラムが見つかりました: {list(invalid_columns_found)}。LLMに修正を試みます...")

        correction_prompt = f"""
以下のpandasデータフレーム操作JSONには、データフレームに存在しないカラム名が含まれています。
存在するカラム名に修正してJSON全体を回答してください。対応すると思われるカラム名が無い場合は、そのままのカラム名を使用してください。
応答は修正後のJSON配列のみを含み、他のテキスト (説明文など) は含めないでください。

- 利用可能なカラム (カラム名: 説明):
{json.dumps(self.column_info, indent=2, ensure_ascii=False)}

- 不正なカラムを含む操作JSON:
{json.dumps(operations, indent=2, ensure_ascii=False)}

"""
        # - 元の自然言語クエリ: {original_query}
        # - 存在しないカラム名: {list(invalid_columns_found)}
        logger.info(f"LLM修正プロンプト生成完了:{correction_prompt}")

        if not self.llm:
            logger.error("LLMクライアントが利用できません。カラム名の修正ができません。")
            return None

        try:
            response = await self.llm.ainvoke([
                {"role": "system", "content": "あなたはpandas操作JSON内の不正なカラム名を、利用可能なカラム情報と元のクエリに基づいて修正するエキスパートです。修正後のJSON配列のみを返してください。"},
                {"role": "user", "content": correction_prompt}
            ])
            response_content = response.content.strip()
            logger.info(f"LLM Raw Response (Correction):{response_content}")

            # 修正後のJSONを抽出・解析 (元の抽出ロジックを再利用)
            corrected_json_str = None
            json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_content, re.DOTALL | re.IGNORECASE)
            if json_match:
                corrected_json_str = json_match.group(1).strip()
            elif response_content.strip().startswith('[') and response_content.strip().endswith(']'):
                corrected_json_str = response_content
            else:
                 logger.error(f"LLMからの修正応答に有効なJSONが見つかりませんでした:{response_content}")
                 return None

            if not corrected_json_str:
                 logger.error("LLMからの修正応答からJSONを抽出できませんでした。")
                 return None

            corrected_operations = json.loads(corrected_json_str)
            if not isinstance(corrected_operations, list):
                 logger.error(f"修正されたJSONがリスト形式ではありません: {type(corrected_operations)}")
                 return None

            # 修正後の簡単なバリデーション（オプション）
            for i, op in enumerate(corrected_operations):
                 if not isinstance(op, dict) or "operation" not in op:
                     logger.error(f"修正後のJSONに不正な操作形式が含まれています (ステップ {i+1}): {op}")
                     return None # 修正後も不正なら失敗とする

            logger.info("LLMによるカラム名の修正が完了しました。")
            return corrected_operations

        except json.JSONDecodeError as e:
            logger.error(f"LLMからの修正応答のJSON解析エラー: {e}")
            logger.debug(f"解析しようとした修正後JSON文字列: {corrected_json_str}")
            return None
        except Exception as e:
            logger.error(f"LLMによるカラム名修正中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return None
    # --- 追加メソッドここまで ---


    def _execute_operations(self, operations: List[Dict]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        操作系列を実行する
        
        Args:
            operations (List[Dict]): 実行する操作系列
            
        Returns:
            Tuple[Optional[pd.DataFrame], Dict]: 結果のデータフレーム (エラー時None) と中間結果
        """
        if self.df.empty:
             logger.warning("_execute_operations: 元データが空のため操作を実行できません。")
             return None, {}

        # オリジナルのデータフレームをコピーして操作
        current_df = self.df.copy()
        logger.info(f"操作実行開始。初期データフレーム: {len(current_df)}行 x {len(current_df.columns)}列")
        logger.debug(f"初期カラム: {list(current_df.columns)}")

        # 中間結果 (集計値など) を保存するための辞書
        intermediate_results = {}
        
        # 各ステップのエラーやワーニングを記録する辞書を追加
        operation_messages = []
        
        # Numpy/Pandasの特殊型をPythonの標準型に変換するヘルパー関数
        def convert_to_standard_type(value):
            if pd.isna(value): # PandasのNAチェック
                return None # 標準的なNoneに変換
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64, np.longdouble)):
                 # 無限大やNaNはNoneに変換する（JSONで扱えないため）
                 if np.isinf(value) or np.isnan(value):
                     return None
                 return float(value)
            elif isinstance(value, (np.complex_, np.complex64, np.complex128)):
                return {'real': value.real, 'imag': value.imag}
            elif isinstance(value, (np.bool_)):
                return bool(value)
            elif isinstance(value, (np.void)): # structured arrays など
                return None # or specific handling if needed
            elif isinstance(value, pd.Timestamp):
                 # ISO 8601 形式の文字列に変換
                 return value.isoformat()
            elif isinstance(value, pd.Timedelta):
                return value.total_seconds() # or str(value)
            elif hasattr(value, 'tolist'): # numpy配列など
                return value.tolist()
            elif isinstance(value, (pd.Series, pd.Index)):
                 return value.tolist()
            # 他の型 (例: Period) も必要に応じて追加
            # elif isinstance(value, pd.Period):
            #     return str(value)
            return value # その他の型はそのまま返す
        
        for i, op in enumerate(operations):
            operation_type = op.get("operation")
            step_num = i + 1
            step_messages = []  # このステップでのメッセージを記録
            
            logger.info(f"--- ステップ {step_num}: {operation_type} 実行開始 ---")
            logger.debug(f"操作パラメータ: {op}")
            logger.debug(f"ステップ {step_num} 開始時のデータ形状: {current_df.shape}")

            # 各操作の実行前にDataFrameがNoneまたは空になっていないかチェック
            if current_df is None or current_df.empty:
                msg = f"ステップ {step_num} ({operation_type}): 前のステップでデータが空になったため、以降の操作をスキップします。"
                logger.warning(msg)
                step_messages.append({"type": "warning", "message": msg})
                operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                break # データがない場合は以降の操作を中断
            
            try:
                if operation_type == "filter":
                    column = op.get("column")
                    condition = op.get("condition")
                    value = op.get("value") # isna/notna の場合は None

                    if not column or not condition:
                        msg = f"ステップ {step_num} (filter): 'column' または 'condition' が不足しています。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    
                    if condition not in ["isna", "notna"] and "value" not in op:
                        msg = f"ステップ {step_num} (filter): 条件 '{condition}' には 'value' が必要です。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    
                    # 列の存在を確認
                    if column not in current_df.columns:
                        msg = f"ステップ {step_num} (filter): 列 '{column}' はデータフレームに存在しません。利用可能な列: {list(current_df.columns)}。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    
                    target_series = current_df[column]
                    original_non_na = target_series.notna().sum() # 元々の非NaN数を記録
                    
                    # 条件に応じたマスクを作成
                    mask = None
                    if condition == "==":
                        mask = target_series == value
                    elif condition == "!=":
                        mask = target_series != value
                    elif condition in [">", ">=", "<", "<="]:
                         # value を数値に変換しようと試みる (比較対象のシリーズが数値型の場合)
                         try:
                             # target_series が数値型か確認
                             if pd.api.types.is_numeric_dtype(target_series):
                                 compare_value = pd.to_numeric(value) # valueも数値に変換
                                 if condition == ">": mask = target_series > compare_value
                                 elif condition == ">=": mask = target_series >= compare_value
                                 elif condition == "<": mask = target_series < compare_value
                                 elif condition == "<=": mask = target_series <= compare_value
                             # 日付型の場合の処理を追加
                             elif pd.api.types.is_datetime64_dtype(target_series):
                                 try:
                                     compare_value = pd.to_datetime(value) # valueを日付に変換
                                     if condition == ">": mask = target_series > compare_value
                                     elif condition == ">=": mask = target_series >= compare_value
                                     elif condition == "<": mask = target_series < compare_value
                                     elif condition == "<=": mask = target_series <= compare_value
                                 except ValueError as e:
                                     msg = f"ステップ {step_num} (filter): value '{value}' を日付に変換できませんでした。エラー: {e}"
                                     logger.warning(msg)
                                     step_messages.append({"type": "warning", "message": msg})
                                     # continueする前にメッセージを保存
                                     operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                                     continue
                             else:
                                 msg = f"ステップ {step_num} (filter): 列 '{column}' が数値型または日付型でないため、比較 ('{condition}') をスキップします。"
                                 logger.warning(msg)
                                 step_messages.append({"type": "warning", "message": msg})
                                 # continueする前にメッセージを保存
                                 operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                                 continue
                         except (ValueError, TypeError) as e:
                              msg = f"ステップ {step_num} (filter): value '{value}' を数値に変換できませんでした。数値比較 ('{condition}') をスキップします。エラー: {e}"
                              logger.warning(msg)
                              step_messages.append({"type": "warning", "message": msg})
                              # continueする前にメッセージを保存
                              operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                              continue
                    elif condition == "in":
                         if isinstance(value, list):
                             mask = target_series.isin(value)
                         else:
                             msg = f"ステップ {step_num} (filter): 条件 'in' の value はリスト形式である必要があります。スキップします。"
                             logger.warning(msg)
                             step_messages.append({"type": "warning", "message": msg})
                             # continueする前にメッセージを保存
                             operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                             continue
                    elif condition == "not in":
                         if isinstance(value, list):
                             mask = ~target_series.isin(value)
                         else:
                             msg = f"ステップ {step_num} (filter): 条件 'not in' の value はリスト形式である必要があります。スキップします。"
                             logger.warning(msg)
                             step_messages.append({"type": "warning", "message": msg})
                             # continueする前にメッセージを保存
                             operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                             continue
                    elif condition == "contains":
                         if isinstance(value, str):
                             # 文字列型に変換し、NaNはFalse扱いにする
                             mask = target_series.astype(str).str.contains(value, case=False, na=False)
                         else:
                             msg = f"ステップ {step_num} (filter): 条件 'contains' の value は文字列である必要があります。スキップします。"
                             logger.warning(msg)
                             step_messages.append({"type": "warning", "message": msg})
                             # continueする前にメッセージを保存
                             operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                             continue
                    elif condition == "not contains":
                         if isinstance(value, str):
                             mask = ~target_series.astype(str).str.contains(value, case=False, na=False)
                         else:
                              msg = f"ステップ {step_num} (filter): 条件 'not contains' の value は文字列である必要があります。スキップします。"
                              logger.warning(msg)
                              step_messages.append({"type": "warning", "message": msg})
                              # continueする前にメッセージを保存
                              operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                              continue
                    elif condition == "starts_with":
                         if isinstance(value, str):
                             mask = target_series.astype(str).str.startswith(value, na=False)
                         else:
                             msg = f"ステップ {step_num} (filter): 条件 'starts_with' の value は文字列である必要があります。スキップします。"
                             logger.warning(msg)
                             step_messages.append({"type": "warning", "message": msg})
                             # continueする前にメッセージを保存
                             operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                             continue
                    elif condition == "ends_with":
                         if isinstance(value, str):
                             mask = target_series.astype(str).str.endswith(value, na=False)
                         else:
                             msg = f"ステップ {step_num} (filter): 条件 'ends_with' の value は文字列である必要があります。スキップします。"
                             logger.warning(msg)
                             step_messages.append({"type": "warning", "message": msg})
                             # continueする前にメッセージを保存
                             operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                             continue
                    elif condition == "isna":
                        mask = target_series.isna()
                    elif condition == "notna":
                        mask = target_series.notna()
                    else:
                        msg = f"ステップ {step_num} (filter): 未知または無効な条件演算子 '{condition}' です。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    
                    if mask is not None:
                         filtered_df = current_df[mask]
                         msg = f"フィルター適用: '{column}' {condition} '{value if value is not None else ''}'. {len(filtered_df)}/{len(current_df)} 行が残りました。"
                         logger.info(msg)
                         step_messages.append({"type": "info", "message": msg})
                         # フィルターによってNaNを含む行がどう扱われたかを確認 (デバッグ用)
                         remaining_non_na = filtered_df[column].notna().sum()
                         logger.debug(f"フィルター後の列 '{column}' の非NaN数: {remaining_non_na} (元: {original_non_na})")
                         current_df = filtered_df
                    else:
                         msg = f"ステップ {step_num} (filter): フィルターマスクが生成されませんでした。データフレームは変更されません。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})

                elif operation_type == "select":
                    columns = op.get("columns")
                    if not columns or not isinstance(columns, list):
                        msg = f"ステップ {step_num} (select): 'columns' がリスト形式で指定されていません。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue

                    # 存在する列のみを選択
                    valid_columns = [col for col in columns if col in current_df.columns]
                    missing_columns = [col for col in columns if col not in valid_columns]
                    if not valid_columns:
                        msg = f"ステップ {step_num} (select): 選択された列 {columns} は全て存在しません。操作を中止します。"
                        logger.error(msg)
                        step_messages.append({"type": "error", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        return None, intermediate_results # 致命的なエラーとしてNoneを返す
                    if missing_columns:
                         msg = f"ステップ {step_num} (select): 列 {missing_columns} は存在しないため、選択から除外します。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})

                    current_df = current_df[valid_columns]
                    msg = f"列を選択しました: {valid_columns}。 残り {len(current_df.columns)} 列。"
                    logger.info(msg)
                    step_messages.append({"type": "info", "message": msg})

                elif operation_type in ["sum", "mean", "max", "min", "count"]:
                    column = op.get("column") # count の場合は None の可能性がある
                    output_name = op.get("output") # 指定されれば使う

                    # count 以外で column が指定されていない、または存在しない場合はエラー
                    if operation_type != "count" and (not column or column not in current_df.columns):
                        msg = f"ステップ {step_num} ({operation_type}): 列 '{column}' が存在しないか指定されていません。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    # count で column が指定されているが存在しない場合も警告
                    if operation_type == "count" and column and column not in current_df.columns:
                         msg = f"ステップ {step_num} (count): 列 '{column}' が存在しません。行数をカウントします。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         column = None # 行数カウントにフォールバック
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                         continue

                    # 集計対象のシリーズ
                    target_series = current_df[column] if column else current_df # countで行全体の場合

                    # 集計実行
                    result_value = None
                    if operation_type == "sum":
                        if pd.api.types.is_numeric_dtype(target_series):
                            result_value = target_series.sum()
                        else: logger.warning(f"ステップ {step_num} (sum): 列 '{column}' は数値型でないため合計できません。スキップ。")
                    elif operation_type == "mean":
                        if pd.api.types.is_numeric_dtype(target_series):
                            result_value = target_series.mean()
                        else: logger.warning(f"ステップ {step_num} (mean): 列 '{column}' は数値型でないため平均を計算できません。スキップ。")
                    elif operation_type == "max":
                        try: result_value = target_series.max() # 数値、文字列、日付などで動作
                        except TypeError: logger.warning(f"ステップ {step_num} (max): 列 '{column}' の型では最大値を計算できません。スキップ。")
                    elif operation_type == "min":
                        try: result_value = target_series.min()
                        except TypeError: logger.warning(f"ステップ {step_num} (min): 列 '{column}' の型では最小値を計算できません。スキップ。")
                elif operation_type == "count":
                    result_value = target_series.count() if column else len(target_series) # 列指定なら非NaN数、なければ行数

                    if result_value is not None:
                        # output_name がなければデフォルト名を生成
                        if not output_name:
                            output_name = f"{operation_type}_{column}" if column else f"{operation_type}_rows"

                        # 中間結果に保存 (標準型に変換)
                        std_result_value = convert_to_standard_type(result_value)
                        intermediate_results[output_name] = std_result_value
                        logger.info(f"集計実行: {output_name} = {std_result_value}")

                        # 集計操作後は結果を1行のDataFrameにするのが一般的だが、
                        # 複数の集計を組み合わせるシナリオも考え、ここではDataFrame構造は維持しない
                        # 必要であれば、結果を1行のDataFrameに変換する処理を追加
                        # current_df = pd.DataFrame([{output_name: std_result_value}])
                    else:
                        # result_value が None の場合 (スキップされたなど)
                        logger.warning(f"ステップ {step_num} ({operation_type}): 集計結果が得られませんでした。")
                    
                    
                elif operation_type == "sort":
                    column = op.get("column")
                    ascending = op.get("ascending", True) # デフォルト昇順

                    if not column or column not in current_df.columns:
                        msg = f"ステップ {step_num} (sort): 列 '{column}' が存在しないか指定されていません。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue

                    current_df = current_df.sort_values(by=column, ascending=ascending, na_position='last') # NaNは最後に
                    msg = f"列 '{column}' でソートしました (ascending={ascending})。"
                    logger.info(msg)
                    step_messages.append({"type": "info", "message": msg})
                    
                elif operation_type == "head":
                    rows = op.get("rows", 5)
                    if not isinstance(rows, int) or rows < 0:
                         msg = f"ステップ {step_num} (head): 'rows' は正の整数である必要があります。デフォルトの5を使用します。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         rows = 5
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                         continue
                    current_df = current_df.head(rows)
                    msg = f"先頭 {rows} 行を取得しました。"
                    logger.info(msg)
                    step_messages.append({"type": "info", "message": msg})
                    
                elif operation_type == "tail":
                    rows = op.get("rows", 5)
                    if not isinstance(rows, int) or rows < 0:
                         msg = f"ステップ {step_num} (tail): 'rows' は正の整数である必要があります。デフォルトの5を使用します。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         rows = 5
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                         continue
                    current_df = current_df.tail(rows)
                    msg = f"末尾 {rows} 行を取得しました。"
                    logger.info(msg)
                    step_messages.append({"type": "info", "message": msg})
                    
                elif operation_type == "group_by":
                    group_column = op.get("column") # 単一列または列リスト
                    target_column = op.get("target") # 集計対象列
                    aggregation = op.get("aggregation") # sum, mean, count, max, min
                    output_name = op.get("output") # オプション

                    # パラメータ検証
                    if not group_column or not target_column or not aggregation:
                         msg = f"ステップ {step_num} (group_by): 'column', 'target', 'aggregation' は必須です。スキップします。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                         continue
                    if aggregation not in ["sum", "mean", "count", "max", "min"]:
                        msg = f"ステップ {step_num} (group_by): サポートされていない集計方法 '{aggregation}' です。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    
                    # グループ化列の存在確認
                    group_columns = [group_column] if isinstance(group_column, str) else group_column
                    valid_group_cols = [col for col in group_columns if col in current_df.columns]
                    if not valid_group_cols:
                        msg = f"ステップ {step_num} (group_by): グループ化列 {group_columns} が存在しません。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    if len(valid_group_cols) < len(group_columns):
                         msg = f"ステップ {step_num} (group_by): グループ化列 {list(set(group_columns) - set(valid_group_cols))} は存在しません。存在する列 {valid_group_cols} で実行します。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                         continue
                    if target_column not in current_df.columns:
                        msg = f"ステップ {step_num} (group_by): 集計対象列 '{target_column}' が存在しません。スキップします。"
                        logger.warning(msg)
                        step_messages.append({"type": "warning", "message": msg})
                        # continueする前にメッセージを保存
                        operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                        continue
                    
                    # 集計実行
                    grouped = current_df.groupby(valid_group_cols)[target_column]
                    result_series = None
                    if aggregation == "sum":
                        if pd.api.types.is_numeric_dtype(current_df[target_column]): result_series = grouped.sum()
                        else: logger.warning(f"ステップ {step_num} (group_by/sum): 列 '{target_column}' は数値型でないため合計できません。スキップ。")
                    elif aggregation == "mean":
                         if pd.api.types.is_numeric_dtype(current_df[target_column]): result_series = grouped.mean()
                         else: logger.warning(f"ステップ {step_num} (group_by/mean): 列 '{target_column}' は数値型でないため平均を計算できません。スキップ。")
                    elif aggregation == "count":
                         result_series = grouped.count() # NaNを除外してカウント
                    elif aggregation == "max":
                         try: result_series = grouped.max()
                         except TypeError: logger.warning(f"ステップ {step_num} (group_by/max): 列 '{target_column}' の型では最大値を計算できません。スキップ。")
                    elif aggregation == "min":
                         try: result_series = grouped.min()
                         except TypeError: logger.warning(f"ステップ {step_num} (group_by/min): 列 '{target_column}' の型では最小値を計算できません。スキップ。")

                    if result_series is not None:
                        # 結果をDataFrameに変換
                        result_df_grouped = result_series.reset_index()
                        # 出力列名を決定
                        if not output_name:
                            output_name = f"{aggregation}_{target_column}"
                        # カラム名をリネーム (グループ化列 + 集計結果列)
                        result_df_grouped.columns = valid_group_cols + [output_name]
                        current_df = result_df_grouped
                        msg = f"グループ化と集計 ({aggregation}) を実行しました。結果: {len(current_df)}行"
                        logger.info(msg)
                        step_messages.append({"type": "info", "message": msg})
                        # 中間結果にもグループごとの値を保存（必要に応じて）
                        # intermediate_results[f"grouped_{output_name}"] = result_df_grouped.to_dict('records')
                    else:
                         msg = f"ステップ {step_num} (group_by): グループ化集計の結果が得られませんでした。"
                         logger.warning(msg)
                         step_messages.append({"type": "warning", "message": msg})
                         # continueする前にメッセージを保存
                         operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})

                else:
                    msg = f"ステップ {step_num}: 未知の操作タイプ '{operation_type}' です。スキップします。"
                    logger.warning(msg)
                    step_messages.append({"type": "warning", "message": msg})
                    # continueする前にメッセージを保存
                    operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                    continue # 次の操作へ

                # メッセージがあれば、必ずステップ終了時に記録
                # ※ continueの前にも記録するようにコードを修正済み
                if step_messages:
                    operation_messages.append({
                        "step": step_num, 
                        "operation": operation_type, 
                        "messages": step_messages
                    })
            
            except Exception as e:
                msg = f"ステップ {step_num} ({operation_type}) の実行中に予期せぬエラーが発生しました: {e}"
                logger.error(msg, exc_info=True)
                step_messages.append({"type": "error", "message": msg})
                operation_messages.append({"step": step_num, "operation": operation_type, "messages": step_messages})
                # エラーが発生した場合の処理
                intermediate_results["operation_messages"] = operation_messages  # メッセージを中間結果に保存
                return None, intermediate_results

            logger.debug(f"ステップ {step_num} ({operation_type}) 完了後のデータ形状: {current_df.shape if current_df is not None else 'None'}")
            logger.debug(f"ステップ {step_num} 完了後のカラム: {list(current_df.columns) if current_df is not None else 'None'}")
            logger.info(f"--- ステップ {step_num}: {operation_type} 実行完了 ---")

        # 操作メッセージを中間結果に保存 (常に保存する)
        intermediate_results["operation_messages"] = operation_messages
        logger.info(f"操作実行完了。記録されたメッセージ: {len(operation_messages)}件")
        
        # デバッグログを追加
        for i, msg_info in enumerate(operation_messages):
            step = msg_info.get("step", "不明")
            op = msg_info.get("operation", "不明")
            msgs = msg_info.get("messages", [])
            logger.debug(f"記録されたメッセージ {i+1}/{len(operation_messages)}: ステップ{step} ({op}) - {len(msgs)}件")
            for j, msg in enumerate(msgs):
                msg_type = msg.get("type", "不明")
                msg_content = msg.get("message", "")
                if msg_type in ["warning", "error"]:
                    logger.debug(f"  重要なメッセージ: [{msg_type}] {msg_content}")

        # 最終結果を返す
        logger.info(f"全 {len(operations)} ステップの操作実行完了。最終結果: {current_df.shape if current_df is not None else 'None'}")
        if current_df is not None:
             logger.debug(f"最終結果サンプル:{current_df.head().to_string()}")
        logger.debug(f"中間結果キー一覧:{list(intermediate_results.keys())}")

        return current_df, intermediate_results


    async def _format_results_as_text(self, query: str, operations: List[Dict], result_df: pd.DataFrame, intermediate_results: Dict[str, Any]) -> str:
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

        # JSON変換用のヘルパー関数 (self._execute_operationsから移動・再利用)
        def default_converter(obj):
            if pd.isna(obj): return None
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
            if isinstance(obj, (np.float_, np.float16, np.float32, np.float64, np.longdouble)): return None if np.isinf(obj) or np.isnan(obj) else float(obj)
            if isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
            if isinstance(obj, (np.bool_)): return bool(obj)
            if isinstance(obj, (np.void)): return None
            if isinstance(obj, pd.Timestamp): return obj.isoformat()
            if isinstance(obj, pd.Timedelta): return obj.total_seconds()
            if hasattr(obj, 'tolist'): return obj.tolist()
            if isinstance(obj, (pd.Series, pd.Index)): return obj.tolist()
            if isinstance(obj, pd.DataFrame): return obj.to_dict(orient='records') # DataFrameも変換可能に
            # UUIDなどの他の型も必要に応じて追加
            # if isinstance(obj, uuid.UUID): return str(obj)
            try:
                # デフォルトのJSONエンコーダで扱えない場合は文字列に変換
                json.dumps(obj) # テストエンコード
                return obj
            except TypeError:
                return str(obj)
                
        logger.info("結果の自然言語フォーマットを開始します。")

        # 操作メッセージの存在を確認するデバッグログを追加
        if "operation_messages" in intermediate_results:
            operation_messages_count = len(intermediate_results["operation_messages"])
            logger.info(f"操作メッセージが {operation_messages_count} 件見つかりました")
            # サンプルとして最初の数件を詳細ログ
            for i, step_info in enumerate(intermediate_results["operation_messages"][:2]):
                step_num = step_info.get("step", "不明")
                operation = step_info.get("operation", "不明")
                messages = step_info.get("messages", [])
                logger.info(f"ステップ{step_num} ({operation}) のメッセージ数: {len(messages)}")
                for j, msg in enumerate(messages[:3]):  # 最初の3件までを表示
                    logger.info(f"  メッセージ{j+1}: タイプ={msg.get('type', '不明')}, 内容={msg.get('message', '内容なし')}")
        else:
            logger.warning("操作メッセージが intermediate_results に見つかりません")
        
        # 結果のサマリーを作成 (表示行数を制限)
        max_rows_display = 10
        result_summary = ""
        num_rows = len(result_df)
        num_cols = len(result_df.columns)
        
        if num_rows == 0:
            result_summary = "(データなし)"
        elif num_rows == 1 and num_cols == 1:
             # 1行1列の結果 (集計値など) は値のみ表示
             result_value = result_df.iloc[0, 0]
             result_summary = f"{result_df.columns[0]}: {default_converter(result_value)}"
        else:
            # 表形式で表示 (最大行数を制限)
            result_summary = result_df.head(max_rows_display).to_string(index=False)
            if num_rows > max_rows_display:
                result_summary += f"\n... ({num_rows - max_rows_display} 行省略)"
        logger.debug(f"最終結果サマリー: {result_summary}")

        # 中間結果もJSONシリアライズ可能な形式に変換
        processed_intermediate_results = {}
        for key, value in intermediate_results.items():
            if key != "operation_messages":  # 操作メッセージは別途処理
                try:
                    processed_intermediate_results[key] = default_converter(value)
                except Exception as e:
                     logger.warning(f"中間結果 '{key}' の変換中にエラー: {e}。文字列として格納します。")
                     processed_intermediate_results[key] = str(value)
        logger.debug(f"処理済み中間結果:{json.dumps(processed_intermediate_results, indent=2, ensure_ascii=False)}")

        # 操作メッセージを処理して重要な警告やエラーを抽出
        operation_messages = intermediate_results.get("operation_messages", [])
        important_messages = []
        has_critical_warnings = False
        
        for step_info in operation_messages:
            step_num = step_info["step"]
            operation = step_info["operation"]
            logger.info(f"ステップ{step_num} ({operation}) のメッセージ: {step_info['messages']}")
            for msg in step_info["messages"]:
                # warningとerrorのみをLLMへの入力に含める（重要なメッセージのみ）
                if msg["type"] in ["warning", "error"]:
                    message_text = msg["message"]
                    # クエリの条件が処理されなかった可能性がある重要なワーニングを識別
                    if ("スキップ" in message_text and 
                        ("列" in message_text and "型" in message_text) or
                        ("変換できません" in message_text)):
                        has_critical_warnings = True
                    
                    important_messages.append(f"ステップ{step_num} ({operation}): {message_text}")
        
        # ワーニングメッセージを目立たせるためのフォーマット
        warnings_section = ""
        if important_messages:
            warnings_section = "処理中に発生した重要な警告やエラー:\n"
            for idx, msg in enumerate(important_messages, 1):
                warnings_section += f"{idx}. {msg}\n"
        else:
            warnings_section = "処理中に発生した警告やエラー: 特になし"
        
        # LLMに渡すプロンプト
        prompt = f"""
以下の情報を使用して、元のクエリに対する回答を自然で読みやすい日本語で生成してください。
        
元のクエリ: {query}
        
実行された操作 (概要): {json.dumps([op['operation'] for op in operations], ensure_ascii=False)}

{warnings_section}

中間結果 (集計値など、もしあれば): {json.dumps(processed_intermediate_results, indent=2, ensure_ascii=False)}

最終結果 ({num_rows}行 x {num_cols}列、最大{max_rows_display}行表示):
{result_summary}

回答のポイント:
1. 【最重要】警告やエラーがある場合は、必ずそれらについて最初に言及してください。特に、クエリの一部が実行できなかった場合は、その理由と影響を明確に説明する必要があります。
2. 例えば「列が数値型でないため比較ができなかった」などの警告があれば、「〇〇列は日付型のため数値比較ができず、その条件は無視されました」のように具体的に説明してください。
3. 回答は元のクエリに対する直接的な答えを含み、クエリのうち満たせていない事項があれば、その旨を明記してください。
4. 重要な数値や結論は明確に述べてください (例: 合計はX個、平均価格はY円、条件に合うものはZ件)。
5. 結果が表形式の場合、「以下の通りです」のように導入し、表が見やすいように提示してください。
6. 結果がない場合は、「条件に一致するデータは見つかりませんでした。」のように明確に伝えてください。
7. 専門用語を避け、平易な言葉で説明してください。
8. 回答は簡潔に、最大でも300文字程度にまとめてください。

${'' if not has_critical_warnings else '【特記】重要な警告が検出されました。クエリの一部が実行されていない可能性が高いです。この点について必ず言及してください。'}

回答例:
「警告: 発注日列が日付型のため、数値比較（>= 10000）ができませんでした。この条件は無視されています。
その上で、製品XYZの合計数量は150個です。」

「在庫数が最も多い製品は「高性能ノートパソコン」で、在庫数は50個です。」
"""
        logger.info(f"LLMプロンプト生成完了 (_format_results_as_text):{prompt[:500]}...")

        if not self.llm:
             logger.error("LLMクライアントが利用できません。結果フォーマットをスキップします。")
             return "エラー: 結果をフォーマットできませんでした。"

        try:
            response = await self.llm.ainvoke([
                    {"role": "system", "content": "あなたはデータ分析結果を自然な日本語で説明するエキスパートです。与えられた情報に基づいて、ユーザーの質問に対する回答を簡潔かつ正確に生成してください。特に、処理中に発生した警告やエラーについては必ず言及し、それが結果にどのような影響を与えたかを明確に説明してください。"},
                {"role": "user", "content": prompt}
            ])
            formatted_text = response.content.strip()
            logger.info("LLMによる結果フォーマット成功。")
            logger.debug(f"LLM Raw Response (_format_results_as_text):{formatted_text}")
            return formatted_text
        except Exception as e:
            logger.error(f"LLM呼び出し中にエラーが発生しました (_format_results_as_text): {e}", exc_info=True)
            # LLMフォーマット失敗時のフォールバック
            if important_messages:
                warning_text = "\n".join(important_messages[:3])  # 重要なメッセージを最大3つ表示
                return f"警告: 処理中に問題が発生しました。\n{warning_text}\n\n生データ:{result_summary}"
            else:
                return f"結果のフォーマット中にエラーが発生しました。生データ:{result_summary}"

# (test_query_agent 関数は __main__.py や test_run_agent.py に移管するため削除)

# モジュールが直接実行された場合のエントリーポイント
if __name__ == "__main__":
    asyncio.run(test_query_agent()) 