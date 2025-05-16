from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from typing import Optional, List
import logging
from ..core.logging_config import setup_logger

logger = setup_logger(__name__)

class EmbeddingManager:
    def __init__(self, model: str = "text-embedding-3-small", persist_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.embedder = OpenAIEmbeddings(model=model)
        self.persist_path = persist_path
        self.vectorstore: Optional[FAISS] = None

    def build_index(self, df: "pd.DataFrame", *, text_column: str, metadata_cols: Optional[List[str]] = None, id_column: Optional[str] = None) -> None:
        if df.empty:
            self.logger.warning("EmbeddingManager: 空のDataFrameが渡されたため、ベクトル化をスキップします。")
            return
        if text_column not in df.columns:
            self.logger.error(f"EmbeddingManager: text_column '{text_column}' が DataFrame に存在しません。ベクトル化中止。")
            raise ValueError(f"text_column '{text_column}' が DataFrame に存在しません。")
        metadata_cols = metadata_cols or []
        docs: List[Document] = []
        self.logger.info(f"EmbeddingManager: ベクトル化を開始します (text_column={text_column}, 件数={len(df)})")
        for idx, row in df.iterrows():
            content = str(row[text_column])
            metadata = {col: row[col] for col in metadata_cols if col in row}
            metadata["row_index"] = int(idx)
            if id_column and id_column in row:
                metadata["id"] = row[id_column]
            docs.append(Document(page_content=content, metadata=metadata))
        try:
            self.vectorstore = FAISS.from_documents(docs, self.embedder)
            self.logger.info(f"EmbeddingManager: ベクトルインデックスの構築が完了しました (ドキュメント数={len(docs)})")
            if self.persist_path:
                self.vectorstore.save_local(self.persist_path)
                self.logger.info(f"EmbeddingManager: FAISSベクトルストアを保存しました: {self.persist_path}")
        except Exception as e:
            self.logger.error(f"EmbeddingManager: ベクトルインデックス構築中にエラー: {e}", exc_info=True)

    def search(self, query: str, k: int = 5):
        if self.vectorstore is None:
            self.logger.warning("EmbeddingManager: ベクトルストアが未構築のため、検索できません。")
            return []
        return self.vectorstore.similarity_search(query, k=k) 