from A2A_risk.samples.python.agents.data_agent.core.models import AnalysisPlan, GraphState
from A2A_risk.samples.python.agents.data_agent.core.logging_config import setup_logger
from A2A_risk.samples.python.agents.data_agent.embeddings.manager import EmbeddingManager
from langchain_openai import ChatOpenAI
import pandas as pd
import os
import json
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Literal, Union, Type, Annotated

logger = setup_logger(__name__)

class QueryAgent:
    """
    自然言語クエリをデータフレーム操作に変換し、実行する汎用エージェント。
    内部でLangGraphとPydanticモデルを使用して、クエリを構造化された操作に変換して実行します。
    """
    def __init__(self, model="gpt-4o-mini"):
        """
        エージェントの初期化
        
        Args:
            model (str): 使用するLLMモデル
        """
        self.model = model
        self.data = None  # 初期状態は空のDataFrame
        self.column_info = {}  # 列情報を格納する辞書
        self.data_source_info = None  # データソース情報を保持 (オプション)
        
        # LangGraphワークフローの設定
        self.workflow = app  # sample.pyで定義されたワークフロー

        # ロガー設定
        self.logger = logging.getLogger(__name__)
        
        # EmbeddingManager の初期化（常に生成しておく）
        self.embedding_mgr = EmbeddingManager()

        # LLM設定
        try:
            self.llm = ChatOpenAI(model=model, temperature=0)
            self.logger.info(f"LLMクライアントを初期化しました: model={model}")
        except Exception as e:
            self.logger.error(f"LLMクライアントの初期化に失敗しました: {e}")
            self.llm = None  # 初期化失敗

    def load_data(self, data_source: Any, build_embedding: bool = False, embedding_persist_path: Optional[str] = None, text_column: Optional[str] = None, metadata_columns: Optional[List[str]] = None):
        """
        指定されたデータソースからデータを読み込み、データフレームと列情報を設定する

        Args:
            data_source (Any): データソース。ファイルパス(str)、pandas DataFrame、または辞書(DB接続情報など)。
            build_embedding (bool): テキスト列をベクトル化して検索可能にするかどうか
            embedding_persist_path (Optional[str]): ベクトルストアの永続化パス
            text_column (Optional[str]): ベクトル化対象のテキスト列名
            metadata_columns (Optional[List[str]]): メタデータとして保持する列名リスト
        """
        self.data_source_info = data_source  # データソース情報を保存
        self.logger.info(f"データソースの読み込みを開始します: {type(data_source)}")
        
        try:
            if isinstance(data_source, str):
                # ファイルパスの場合
                file_path = data_source
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"指定されたファイル・フォルダが見つかりません: {file_path}")

                if file_path.lower().endswith('.csv'):
                    self.logger.info(f"ファイルを読み込みます: {file_path}")
                    # ファイル拡張子に基づいて読み込み方法を選択
                    if file_path.lower().endswith('.csv'):
                        self.data = pd.read_csv(file_path)
                    elif file_path.lower().endswith(('.xlsx', '.xls')):
                        self.data = pd.read_excel(file_path)
                    else:
                        raise ValueError(f"サポートされていないファイル形式です: {file_path}")

                #フォルダパスの場合、フォルダ配下の階層構造を再帰的に読み込み、拡張子によって読み込み方法を選択
                #読み込んだデータをメタデータとともにデータフレームとして管理
                if os.path.isdir(file_path):
                    data_list = []
                    for root, dirs, files in os.walk(file_path):
                        for file in files:
                            file_full_path = os.path.join(root, file)
                            try:
                                if file.lower().endswith('.pdf'):
                                    # PDFファイルをテキスト化してDataFrameに格納
                                    try:
                                        import PyPDF2
                                        import re
                                        with open(file_full_path, 'rb') as pdf_file:
                                            reader = PyPDF2.PdfReader(pdf_file)
                                            paragraphs = []
                                            page_numbers = []
                                            for i, page in enumerate(reader.pages):
                                                text = page.extract_text() or ""
                                                # 段落ごとに分割し、改行はスペースに
                                                page_paragraphs = [p.strip().replace('\n', ' ') for p in re.split(r'\n\s*\n', text) if p.strip()]
                                                paragraphs.extend(page_paragraphs)
                                                page_numbers.extend([i+1]*len(page_paragraphs))
                                        if paragraphs:
                                            df = pd.DataFrame({
                                                'folder': [os.path.basename(root)] * len(paragraphs),
                                                'file': [file] * len(paragraphs),
                                                'path': [file_full_path] * len(paragraphs),
                                                'page': page_numbers,
                                                'paragraph': list(range(1, len(paragraphs)+1)),
                                                'content': paragraphs
                                            })
                                            df.to_csv(file_full_path.replace('.pdf', '.csv'), index=False)
                                            data_list.append(df)
                                            self.logger.info(f"PDFファイルを段落ごとに分割して読み込みました: {file_full_path}")
                                        else:
                                            self.logger.warning(f"PDFファイルからテキストを抽出できませんでした: {file_full_path}")
                                    except Exception as e:
                                        self.logger.error(f"PDFファイルのテキスト化に失敗しました: {file_full_path}, エラー: {e}")
                                elif file.lower().endswith(('.txt', '.md')):
                                    # テキストファイルを段落ごとにDataFrameに格納
                                    try:
                                        import re
                                        with open(file_full_path, 'r', encoding='utf-8') as f:
                                            text = f.read()
                                        paragraphs = [p.strip().replace('\n', ' ') for p in re.split(r'\n\s*\n', text) if p.strip()]
                                        df = pd.DataFrame({
                                            'folder': [os.path.basename(root)] * len(paragraphs),
                                            'file': [file] * len(paragraphs),
                                            'path': [file_full_path] * len(paragraphs),
                                            'page': [1] * len(paragraphs),
                                            'paragraph': list(range(1, len(paragraphs)+1)),
                                            'content': paragraphs
                                        })
                                        data_list.append(df)
                                        self.logger.info(f"テキストファイルを段落ごとに分割して読み込みました: {file_full_path}")
                                    except Exception as e:
                                        self.logger.error(f"テキストファイルの段落分割に失敗しました: {file_full_path}, エラー: {e}")
                                elif file.lower().endswith(('.xlsx', '.xls')):
                                    df = pd.read_excel(file_full_path)
                                    df['file_path'] = file_full_path
                                    data_list.append(df)
                                elif file.lower().endswith('.csv'):
                                    df = pd.read_csv(file_full_path)
                                    df['file_path'] = file_full_path
                                    data_list.append(df)
                                else:
                                    self.logger.warning(f"サポートされていないファイル形式のためスキップします: {file_full_path}")
                                    continue
                            except Exception as e:
                                self.logger.error(f"ファイルの読み込みに失敗しました: {file_full_path}, エラー: {e}")
                                continue
                    if data_list:
                        self.data = pd.concat(data_list, ignore_index=True)
                        self.data.to_csv(os.path.join(file_path, 'data.csv'), index=False)
                    else:
                        self.data = pd.DataFrame()

            elif isinstance(data_source, pd.DataFrame):
                # DataFrameオブジェクトの場合
                self.logger.info("pandas DataFrameを直接読み込みます")
                self.data = data_source.copy()  # 念のためコピー
            else:
                raise TypeError("サポートされていないデータソース形式です。ファイルパス(str)、pandas DataFrame、または対応する辞書を指定してください。")

            if self.data.empty:
                self.logger.warning("読み込んだデータが空です。処理を続行しますが、結果は得られない可能性があります。")
                self.column_info = {}
                return  # 空の場合は以降の処理をスキップ

            self.logger.info(f"データ読み込み成功: {len(self.data)}行 x {len(self.data.columns)}列")
            self.logger.info(f"カラム一覧: {list(self.data.columns)}")

            # データ型の推定と数値データの前処理を試みる
            self._preprocess_data()

            # 列情報を動的に生成
            self.column_info = self._generate_column_info()
            self.logger.info("列情報の生成完了")

            # データの最初の数行を表示して確認
            self.logger.info(f"データサンプル:{self.data.head(3).to_string()}")

            # --------------------------------------------------
            # Embedding の構築
            # --------------------------------------------------
            if build_embedding:
                self.logger.info("Embedding: 構築処理を開始します ...")
                # text_column が指定されていない場合は推測する
                target_text_col = text_column
                if target_text_col is None:
                    if "content" in self.data.columns:
                        target_text_col = "content"
                    else:
                        # object 型の先頭列を候補にする
                        cand_cols = [c for c in self.data.columns if self.data[c].dtype == "object"]
                        target_text_col = cand_cols[0] if cand_cols else None

                if target_text_col:
                    # 永続化パスの設定
                    if embedding_persist_path:
                        self.embedding_mgr.persist_path = embedding_persist_path

                    try:
                        self.logger.info(f"Embedding: build_indexを呼び出します (text_column={target_text_col})")
                        self.embedding_mgr.build_index(
                            self.data,
                            text_column=target_text_col,
                            metadata_cols=metadata_columns,
                        )
                        self.logger.info(f"Embedding: build_indexの呼び出しが完了しました (text_column={target_text_col})")
                    except Exception as emb_err:
                        self.logger.error(f"Embedding: インデックス構築に失敗: {emb_err}", exc_info=True)
                else:
                    self.logger.warning("Embedding: テキスト列が見つからなかったため、Embedding 構築をスキップしました。")

        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}", exc_info=True)
            self.data = pd.DataFrame()
            self.column_info = {}
            raise  # エラーを再送出して呼び出し元に通知

    def _preprocess_data(self):
        """データフレームのデータ型を推定し、数値データのクリーンアップを試みる"""
        self.logger.info("データ型の前処理を開始します...")
        for col in self.data.columns:
            # 全てがNaNまたはNoneのカラムはスキップ
            if self.data[col].isnull().all():
                self.logger.info(f"列 '{col}' は全て欠損値のため、前処理をスキップします。")
                continue

            # object型で、数値に変換できそうな列を試す
            if self.data[col].dtype == 'object':
                try:
                    # 通貨記号やカンマを除去
                    series_str = self.data[col].astype(str)
                    cleaned_series = series_str.replace(r'[$,¥€]', '', regex=True)

                    # 数値に変換可能かチェック
                    numeric_check = pd.to_numeric(cleaned_series, errors='coerce')

                    # NaNでない割合を計算
                    notna_ratio = numeric_check.notna().sum() / self.data[col].notna().sum() if self.data[col].notna().sum() > 0 else 0

                    if notna_ratio > 0.8:  # 非NaNだった値の8割以上が数値に変換できれば変換
                        self.data[col] = pd.to_numeric(cleaned_series, errors='coerce')
                        self.logger.info(f"列 '{col}' を数値型に変換しました。({notna_ratio*100:.1f}% 変換成功)")
                    # 日付っぽい文字列を日付型に変換
                    elif pd.to_datetime(self.data[col], errors='coerce').notna().mean() > 0.8:
                        try:
                            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                            self.logger.info(f"列 '{col}' を日付型に変換しました。")
                        except Exception as date_err:
                            self.logger.warning(f"列 '{col}' の日付型への変換中にエラーが発生しました: {date_err}")

                except Exception as e:
                    # 変換に失敗しても元の型のままにする
                    self.logger.warning(f"列 '{col}' の型変換中にエラーが発生しましたが、処理を続行します: {e}")

        self.logger.info("データ型の前処理が完了しました。")
        self.logger.info(f"処理後のデータ型:{self.data.dtypes}")

    def _generate_column_info(self) -> Dict[str, Dict[str, Any]]:
        """データフレームから列名とデータ型の情報を生成する"""
        if self.data.empty:
            return {}

        col_info = {}
        self.logger.info("カラム情報の生成を開始します...")
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            dtype_str = str(dtype)
            
            # データ型に応じた説明
            if pd.api.types.is_numeric_dtype(dtype):
                description = "数値"
                # 数値データの場合、最小値と最大値を取得
                not_null = self.data[col].dropna()
                if len(not_null) > 0:
                    min_val = not_null.min()
                    max_val = not_null.max()
                    mean_val = not_null.mean()
                    unique_count = not_null.nunique()
                    col_info[col] = {
                        "type": description,
                        "dtype": dtype_str,
                        "min": min_val,
                        "max": max_val,
                        "mean": mean_val,
                        "unique_count": unique_count
                    }
                else:
                    col_info[col] = {"type": description, "dtype": dtype_str}
            
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                description = "日付"
                not_null = self.data[col].dropna()
                if len(not_null) > 0:
                    min_date = not_null.min()
                    max_date = not_null.max()
                    col_info[col] = {
                        "type": description,
                        "dtype": dtype_str,
                        "min_date": min_date,
                        "max_date": max_date
                    }
                else:
                    col_info[col] = {"type": description, "dtype": dtype_str}
            
            elif dtype == 'object' or pd.api.types.is_string_dtype(dtype):
                description = "文字列"
                not_null = self.data[col].dropna()
                unique_count = not_null.nunique()
                
                col_info[col] = {
                    "type": description,
                    "dtype": dtype_str,
                    "unique_count": unique_count
                }
                
                # ユニークな値が少ない場合はカテゴリカルデータとして扱い、値のリストを追加
                if 0 < unique_count <= 20:  # 20個以下のユニークな値がある場合
                    unique_values = not_null.unique().tolist()
                    col_info[col]["unique_values"] = unique_values
                    col_info[col]["type"] = "カテゴリ"
            
            elif pd.api.types.is_categorical_dtype(dtype):
                description = "カテゴリ"
                categories = self.data[col].cat.categories.tolist()
                col_info[col] = {
                    "type": description,
                    "dtype": dtype_str,
                    "categories": categories
                }
            
            elif pd.api.types.is_bool_dtype(dtype):
                description = "真偽値"
                col_info[col] = {"type": description, "dtype": dtype_str}
            
            else:
                description = "その他"
                col_info[col] = {"type": description, "dtype": dtype_str}
        
        self.logger.info(f"カラム情報の生成が完了しました。{len(col_info)}列の情報を生成しました。")
        return col_info

    async def process_query(self, query: str, return_type: str = "text", **kwargs) -> dict:
        """
        自然言語クエリを処理し、結果を返す
        
        Args:
            query (str): 自然言語クエリ
            return_type (str): 'text'（デフォルト）または 'df'。
            **kwargs: 追加パラメータ
        Returns:
            dict: {text, data, data_type, intermediate_results}
        """
        if self.data is None or self.data.empty:
            return {"text": "データが読み込まれていないか空です。load_data()メソッドで有効なデータを読み込んでください。", "data": None, "data_type": return_type, "intermediate_results": {}}
        
        self.logger.info(f"クエリを処理します: {query}")
        
        # LangGraphの初期状態を設定
        initial_state = {
            "query": query,
            "dataframes": {"df_main": self.data},
            "intermediate_results": {},
            "execution_log": [],
            "error_message": None,
            "messages": [], # Initialize messages
            "embedding_manager": self.embedding_mgr # <-- Pass manager to state
        }
        
        # ワークフロー実行
        try:
            result = await self.workflow.ainvoke(initial_state)
            if result.get("error_message"):
                self.logger.error(f"クエリ処理中にエラーが発生しました: {result['error_message']}")
                return {"text": "問い合わせの結果を取得できませんでした。クエリを具体化・細分化して再度問い合わせください。", "data": None, "data_type": return_type, "intermediate_results": {}}
            
            all_dfs = result.get("dataframes", {})
            final_df = all_dfs.get("df_final_output")
            execution_log = result.get("execution_log", [])
            intermediate_results = result.get("intermediate_results", {})
            logger.info(f"result: {result}")

            if (final_df is not None) or (intermediate_results is not None):
                formatted_result = await self._format_results_as_text(query, execution_log, final_df, intermediate_results)
                
                data = all_dfs  # すべてのDataFrameを辞書で返す
                
                return {
                    "text": formatted_result,
                    "data": data,
                    "data_type": return_type,
                    "intermediate_results": intermediate_results
                }
            else:
                return {"text": "問い合わせの結果を取得できませんでした。クエリを具体化・細分化して再度問い合わせください。", "data": None, "data_type": return_type, "intermediate_results": {}}
        except Exception as e:
            self.logger.error(f"クエリ処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return {"text": "問い合わせの結果を取得できませんでした。クエリを具体化・細分化して再度問い合わせください。", "data": None, "data_type": return_type, "intermediate_results": {}}

    async def _format_results_as_text(self, query: str, execution_log: List[Dict], result_df: pd.DataFrame, intermediate_results: Dict[str, Any], error_message: Optional[str] = None) -> str:
        """LLMを使用して結果をテキスト形式にフォーマットする"""
        logger.info(f"結果：{result_df}")
        
        final_prompt = f"""
以下の情報を使用して、元のクエリに対する回答を自然で読みやすい日本語で生成してください。

元のクエリ: {query}

実行された処理のログ:
{execution_log}

中間結果 (集計値など、もしあれば): {json.dumps(intermediate_results, indent=2, ensure_ascii=False)}

発生したエラー: {error_message}

最終結果
{result_df}

回答のポイント:
1. 【重要】「実行された処理の概要」で言及されている警告やエラー、スキップされた処理などを踏まえて、結果がクエリの意図通りか、どの部分が実行できなかったかを明確に説明してください。
2. 回答は元のクエリに対する直接的な答えを含み、クエリのうち満たせていない事項があれば、その旨を明記してください。
3. 結果が表形式の場合、「以下の通りです」のように導入し、表が見やすいように提示してください。
4. 結果がない場合は、「条件に一致するデータは見つかりませんでした。」のように明確に伝えてください。

回答例:
「ご指定の通り、まず平均単価を計算しました。しかし、次のステップで単価と平均単価を比較しようとした際に、型が異なっていたため比較できず、このフィルター処理はスキップされました。そのため、フィルターは適用されず、全データの単価の合計はXXX円となりました。」
"""
        logger.info(f"LLMプロンプト生成完了 (最終回答):{final_prompt[:500]}...")
        
        try:
            # self.llmが設定されていない場合は、新しく作成
            if not hasattr(self, 'llm') or self.llm is None:
                self.logger.warning("LLMが初期化されていないため、新しく作成します")
                self.llm = ChatOpenAI(model=self.model, temperature=0)
            
            response = await self.llm.ainvoke([
                {"role": "system", "content": "あなたはデータ分析の処理要約、中間結果、最終データを基に、ユーザーの元の質問に対する最終的な回答を自然な日本語で生成するエキスパートです。処理要約で言及されている問題点を踏まえて、結果を正確に説明してください。"},
                {"role": "user", "content": final_prompt}
            ])
            formatted_text = response.content.strip()
            
            return f"{formatted_text}"
        except Exception as e:
            self.logger.error(f"結果のフォーマット中にエラーが発生しました: {e}", exc_info=True)
            # エラー時はシンプルな出力を返す
            if len(result_df) <= 20:
                df_str = result_df.to_string(index=False)
                return f"クエリの結果は以下の通りです：\n\n{df_str}"
            
            # 大きなデータフレームの場合は先頭10行と概要を表示
            head_str = result_df.head(10).to_string(index=False)
            return f"クエリの結果（全{len(result_df)}行中の先頭10行）：\n\n{head_str}\n\n合計 {len(result_df)} 行のデータが見つかりました。" 