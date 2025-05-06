from typing import List, Dict, Any, Optional, Literal, Union, Type, Annotated
from pydantic import BaseModel, Field, validator, ValidationError, model_validator
from pydantic_core import PydanticUndefined
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
import operator
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import logging
from A2A_risk.samples.python.agents.data_agent.embeddings import EmbeddingManager
from A2A_risk.samples.python.agents.data_agent.core.models import (
    InputSource, OutputDestination, AggregationSpec, FilterCondition,
    LoadParams, SelectParams, FilterParams, GroupByParams, MergeParams,
    CalculateScalarParams, ScalarArithmeticParams, CalculateColumnParams,
    DropDuplicatesParams, SortParams, RenameParams, UseLLMParams, RAGParams,
    OperationStep, AnalysisPlan, GraphState
)
from A2A_risk.samples.python.agents.data_agent.core.utils import safe_compare
from A2A_risk.samples.python.agents.data_agent.nodes.plan_generator import plan_generator_node
from A2A_risk.samples.python.agents.data_agent.nodes.execute_plan import execute_plan_node

# --- プラン全体 ---
class AnalysisPlan(BaseModel):
    plan: List[OperationStep]

# --- 操作タイプと対応するパラメータクラスのマッピング ---

OPERATION_PARAM_MAP: Dict[str, Type[BaseModel]] = {
    'load': LoadParams,
    'select': SelectParams,
    'filter': FilterParams,
    'group_by': GroupByParams,
    'merge': MergeParams,
    'calculate_scalar': CalculateScalarParams,
    'scalar_arithmetic': ScalarArithmeticParams,
    'calculate_column': CalculateColumnParams,
    'drop_duplicates': DropDuplicatesParams,
    'sort': SortParams,
    'rename': RenameParams,
    'use_llm': UseLLMParams,
    'rag': RAGParams,
}

query = str({
    "description": "発注情報のデータから、担当者ごとに過去1年間の取引の平均単価と最新の取引単価を抽出してください。対象とするデータは、担当者ID、担当者名、取引先名、品目名、過去平均単価、最新取引単価です。",
    "query": "担当者ごとの過去1年間の平均単価と最新の取引単価を抽出してください。対象期間は過去1年間です。出力には担当者ID、担当者名、取引先名、品目名、過去平均単価、最新取引単価のカラムを含めてください。",
    "expected_output": "担当者ID、担当者名、取引先名、品目名、過去平均単価、最新取引単価の表データ。"
  })

import numpy as np
from typing import Dict, Any, List, Optional, TypedDict
import logging


# --- ロガー設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# --- LangGraph---
from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)
workflow.add_node("plan_generator", plan_generator_node)
workflow.add_node("execute_plan", execute_plan_node)

workflow.set_entry_point("plan_generator")
workflow.add_edge("plan_generator", "execute_plan")
workflow.add_edge("execute_plan", END)

app = workflow.compile()

# 最後にQueryAgentクラスを追加（sample.pyの末尾）
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import os
import logging
import json
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
