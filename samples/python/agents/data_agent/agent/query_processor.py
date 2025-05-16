# from A2A_risk.samples.python.agents.data_agent.core.models import AnalysisPlan, GraphState
from A2A_risk.samples.python.agents.data_agent.core.logging_config import setup_logger
from A2A_risk.samples.python.agents.data_agent.embeddings.manager import EmbeddingManager
# from A2A_risk.samples.python.agents.data_agent.workflow_definition import app
from langchain_openai import ChatOpenAI
import pandas as pd
import os
import json
import logging
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Literal, Union, Type, Annotated
# tiktoken をインポート
import tiktoken

# create_pandas_dataframe_agent用
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


logger = setup_logger(__name__)

class QueryAgent:
    """
    自然言語クエリをデータフレーム操作に変換し、実行する汎用エージェント。
    内部でLangGraphとPydanticモデルを使用して、クエリを構造化された操作に変換して実行します。
    """
    def __init__(self, model="gpt-4.1-nano"):
        """
        エージェントの初期化
        
        Args:
            model (str): 使用するLLMモデル
        """
        self.model = model
        self.data = None  # 初期状態は空のDataFrame
        self.column_info = {}  # 列情報を格納する辞書
        self.data_source_info = None  # データソース情報を保持 (オプション)
        
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
        self.logger.info(f"データソースの読み込みを開始します: {data_source}")
        try:
            if isinstance(data_source, str):
                # ファイルパスの場合
                file_path = data_source
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"指定されたファイル・フォルダが見つかりません: {file_path}")

                if file_path.lower().endswith('.csv'):
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
                                            page_numbers_for_paragraphs = [] # 段落ごとのページ番号を保持
                                            for i, page in enumerate(reader.pages):
                                                text = page.extract_text() or ""
                                                page_paragraphs = [p.strip().replace('\\n', ' ') for p in re.split(r'\\n\\s*\\n', text) if p.strip()]
                                                paragraphs.extend(page_paragraphs)
                                                page_numbers_for_paragraphs.extend([i+1]*len(page_paragraphs)) # 各段落に対応するページ番号

                                        if paragraphs:
                                            try:
                                                # エンコーディングを取得 (gpt-4o-miniやgpt-4で使われることが多いもの)
                                                encoding = tiktoken.get_encoding("cl100k_base")
                                                MAX_TOKENS = 1000
                                                OVERLAP_RATIO = 0.1

                                                chunks = []
                                                current_chunk_paragraphs = []
                                                current_chunk_tokens_ids = []
                                                start_page = 1 # チャンクの開始ページ

                                                for idx, paragraph in enumerate(paragraphs):
                                                    paragraph_token_ids = encoding.encode(paragraph)
                                                    paragraph_token_count = len(paragraph_token_ids)
                                                    current_page = page_numbers_for_paragraphs[idx]

                                                    # 現在の段落を追加するとMAX_TOKENSを超えるか？
                                                    if len(current_chunk_tokens_ids) + paragraph_token_count > MAX_TOKENS and current_chunk_paragraphs:
                                                        # 現在のチャンクを完成させる
                                                        chunk_text = "\\n\\n".join(current_chunk_paragraphs)
                                                        chunks.append({
                                                            'folder': os.path.basename(root),
                                                            'file': file,
                                                            'path': file_full_path,
                                                            'start_page': start_page, # チャンクの開始ページ
                                                            'content': chunk_text,
                                                            'token_count': len(current_chunk_tokens_ids)
                                                        })

                                                        # 重複部分を計算 (トークンIDリストから)
                                                        overlap_token_count = int(len(current_chunk_tokens_ids) * OVERLAP_RATIO)
                                                        overlap_token_ids = current_chunk_tokens_ids[-overlap_token_count:]

                                                        # 重複部分の段落を特定するのは複雑なので、重複トークンIDと新しい段落から次のチャンクを開始
                                                        current_chunk_paragraphs = [paragraph] # 新しい段落から開始
                                                        current_chunk_tokens_ids = overlap_token_ids + paragraph_token_ids
                                                        start_page = current_page # 新しいチャンクの開始ページ

                                                    else:
                                                        # チャンクに段落を追加
                                                        current_chunk_paragraphs.append(paragraph)
                                                        current_chunk_tokens_ids.extend(paragraph_token_ids)
                                                        if not current_chunk_paragraphs: # 最初の段落なら開始ページを設定
                                                            start_page = current_page


                                                # 最後のチャンクを追加
                                                if current_chunk_paragraphs:
                                                    chunk_text = "\\n\\n".join(current_chunk_paragraphs)
                                                    chunks.append({
                                                            'folder': os.path.basename(root),
                                                        'file': file,
                                                        'path': file_full_path,
                                                        'start_page': start_page,
                                                        'content': chunk_text,
                                                        'token_count': len(current_chunk_tokens_ids)
                                                    })

                                                if chunks:
                                                    df = pd.DataFrame(chunks)
                                                    # chunk_id を追加
                                                    df['chunk_id'] = range(1, len(df) + 1)
                                                    data_list.append(df)
                                                    self.logger.info(f"PDFファイルをトークンベースでチャンク化して読み込みました ({len(chunks)} chunks): {file_full_path}")
                                                else:
                                                        self.logger.warning(f"PDFファイルからチャンクを生成できませんでした: {file_full_path}")


                                            except Exception as chunk_err:
                                                self.logger.error(f"PDFのチャンク化中にエラーが発生しました: {file_full_path}, エラー: {chunk_err}. 段落ごとの分割にフォールバックします。")
                                                # エラー時は元の段落分割ロジックにフォールバック (オプション)
                                                df = pd.DataFrame({
                                                    'folder': [os.path.basename(root)] * len(paragraphs),
                                                    'file': [file] * len(paragraphs),
                                                    'path': [file_full_path] * len(paragraphs),
                                                    'page': page_numbers_for_paragraphs, # 元のページ番号を使用
                                                    'paragraph': list(range(1, len(paragraphs)+1)),
                                                    'content': paragraphs
                                                })
                                                data_list.append(df)
                                    except Exception as e:
                                        self.logger.error(f"PDFファイルの段落分割に失敗しました: {file_full_path}, エラー: {e}")

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
                                # elif file.lower().endswith('.csv'):
                                #     df = pd.read_csv(file_full_path)
                                #     df['file_path'] = file_full_path
                                #     data_list.append(df)
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
            # create_pandas_dataframe_agent用
            agent = create_pandas_dataframe_agent(
                self.llm,
                self.data.copy(),  # DataFrameのコピーを渡す
                verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                return_intermediate_steps=True,
                allow_dangerous_code=True,
            )
            result = agent.invoke(query)

            logger.info(f"result: {result}")
            last_observation = ""
            for step_idx, (action_log, observation) in enumerate(result["intermediate_steps"]):
                if action_log.tool == "python_repl_ast":
                    print(f"Step {step_idx}:")
                    print("  実行コード :", action_log.tool_input)
                    print(f"  実行結果({type(observation)})   :{observation}")
                    last_observation = observation

            return {"text": result["output"], "data": last_observation}

        except Exception as e:
            self.logger.error(f"クエリ処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return {"text": "問い合わせの結果を取得できませんでした。クエリを具体化・細分化して再度問い合わせください。", "data": None, "data_type": return_type, "intermediate_results": {}}

logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# 設定ファイルとデータファイルのデフォルトパス
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(CURRENT_DIR, "data", "dummy_data.csv")
DEFAULT_LLM_MODEL = 'gpt-4.1-nano' # デフォルトLLMモデルを定数化

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
    _config = config_data
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
            llm_model = _config.get('llm_model')
            data_source = _config.get('data_source')

            logger.info(f"使用するLLMモデル: {llm_model}")

            # インスタンスを作成
            _agent_instance = QueryAgent(model=llm_model)

            # データをロード
            logger.info(f"データソースをロードします: {data_source}")
            _agent_instance.load_data(data_source)
            logger.info(f"データロード完了 (ソース: {data_source})。")

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

    if not input_query.strip():
        logger.warning("空のクエリを受け取りました。")
        return {"output": "クエリが空です。", "error": "Empty query"}

    try:
        # QueryAgentのprocess_queryメソッドを呼び出す
        logger.info(f"クエリ '{input_query}' を処理します。パラメータ: {additional_params}")
        result = await agent.process_query(input_query, **additional_params)
        return {"output": result}
    except Exception as e:
        # エラーハンドリング
        logger.error(f"クエリ処理中にエラーが発生しました: {e}", exc_info=True)
        error_message = f"クエリ処理中にエラーが発生しました: {str(e)}"
        return {"output": error_message, "error": str(e)}
