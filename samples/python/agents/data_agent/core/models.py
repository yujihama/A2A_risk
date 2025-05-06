from typing import List, Dict, Any, Optional, Literal, Union, Type, Annotated
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import AnyMessage
import operator
import pandas as pd

# --- 共通コンポーネント ---
class InputSource(BaseModel):
    dataframe: Optional[str] = None
    left_dataframe: Optional[str] = None
    right_dataframe: Optional[str] = None
    scalars: Optional[List[str]] = None

class OutputDestination(BaseModel):
    dataframe: Optional[str] = None
    scalar: Optional[str] = None

class AggregationSpec(BaseModel):
    column: str
    func: Literal['mean', 'sum', 'count', 'min', 'max', 'first', 'last', 'nunique']

class FilterCondition(BaseModel):
    column: str
    operator: Literal[
        '==', '!=', '>', '>=', '<', '<=',
        'in', 'not in',
        'contains', 'not contains',
        'startswith', 'endswith',
        'isna', 'notna'
    ]
    value: Optional[Any] = None
    value_ref: Optional[str] = None

    @model_validator(mode='after')
    def check_value_logic(self) -> 'FilterCondition':
        op = self.operator
        value_is_set = self.value is not None
        ref_is_set = self.value_ref is not None
        if op not in ['isna', 'notna']:
            if not (value_is_set ^ ref_is_set):
                raise ValueError(f"演算子 '{op}' では 'value' または 'value_ref' のどちらか一方のみを指定してください。")
        elif value_is_set or ref_is_set:
            raise ValueError(f"演算子 '{op}' では 'value' や 'value_ref' は不要です。")
        return self

# --- 各操作のパラメータ定義 ---
class LoadParams(BaseModel):
    source_path: str

class SelectParams(BaseModel):
    columns: List[str]
    drop_duplicates: Optional[bool] = False
    subset_for_duplicates: Optional[List[str]] = None

class FilterParams(BaseModel):
    conditions: List[FilterCondition]
    condition_logic: Optional[Literal['AND', 'OR']] = 'AND'

class GroupByParams(BaseModel):
    group_keys: List[str]
    aggregations: Dict[str, AggregationSpec]
    @model_validator(mode='after')
    def check_aggregations_not_empty(self) -> 'GroupByParams':
        if not self.aggregations:
            raise ValueError("aggregationsは空にできません。少なくとも1つ以上の集計指定が必要です。")
        return self

class MergeParams(BaseModel):
    on: List[str]
    how: Literal['left', 'right', 'inner', 'outer'] = 'inner'
    suffixes: Optional[List[str]] = None

class CalculateScalarParams(BaseModel):
    column: str
    aggregation: Literal['mean', 'sum', 'count', 'min', 'max', 'first', 'last', 'nunique']

class ScalarArithmeticParams(BaseModel):
    expression: Optional[str] = None
    operator: Optional[Literal['add', 'subtract', 'multiply', 'divide']] = None
    operand1_ref: Optional[str] = None
    operand1_literal: Optional[Any] = None
    operand2_ref: Optional[str] = None
    operand2_literal: Optional[Any] = None

class CalculateColumnParams(BaseModel):
    new_column_name: str
    operation: Literal['add', 'subtract', 'multiply', 'divide', 'compare', 'if_else']
    left_column: Optional[str] = None
    right_column: Optional[str] = None
    left_value: Optional[Any] = None
    right_value: Optional[Any] = None
    left_scalar_ref: Optional[str] = None
    right_scalar_ref: Optional[str] = None
    compare_operator: Optional[Literal['==', '!=', '>', '>=', '<', '<=']] = None
    condition_column: Optional[str] = None
    true_value: Optional[Any] = None
    false_value: Optional[Any] = None
    @model_validator(mode='after')
    def validate_operands(self) -> 'CalculateColumnParams':
        left_options = [self.left_column is not None, self.left_value is not None, self.left_scalar_ref is not None]
        if sum(left_options) != 1:
            raise ValueError("左側オペランドは列名、固定値、スカラー参照のいずれか1つのみを指定してください")
        if self.operation != 'if_else':
            right_options = [self.right_column is not None, self.right_value is not None, self.right_scalar_ref is not None]
            if sum(right_options) != 1:
                raise ValueError("右側オペランドは列名、固定値、スカラー参照のいずれか1つのみを指定してください")
        if self.operation == 'compare' and not self.compare_operator:
            raise ValueError("compare操作では比較演算子を指定してください")
        if self.operation == 'if_else':
            if not self.condition_column:
                raise ValueError("if_else操作では条件列を指定してください")
            if self.true_value is None and self.false_value is None:
                raise ValueError("if_else操作では真または偽の場合の値のいずれかを指定してください")
        return self

class DropDuplicatesParams(BaseModel):
    subset: Optional[List[str]] = None
    keep: Optional[Literal['first', 'last', False]] = 'first'

class SortParams(BaseModel):
    by: List[str]
    ascending: Union[bool, List[bool]] = True

class RenameParams(BaseModel):
    columns: Dict[str, str]

class UseLLMParams(BaseModel):
    use_column_name: str
    output_column_name: str
    prompt: str

class RAGParams(BaseModel):
    retrieval_query: str = Field(..., description="ベクトル検索に使用するクエリ")
    prompt_template: Optional[str] = Field(
        default="以下のコンテキスト情報のみを使用して、質問に答えてください。\n\nコンテキスト:\n{context}\n\n質問:\n{question}",
        description="LLMに渡すプロンプトテンプレート。{context}と{question}を含める必要があります。"
    )
    top_k: int = Field(default=5, description="検索する類似ドキュメントの数")

class OperationStep(BaseModel):
    step_id: str = Field(..., description="各ステップの一意な識別子")
    operation: Literal[
        'load', 'select', 'filter', 'group_by', 'merge',
        'calculate_scalar', 'scalar_arithmetic', 'drop_duplicates', 'sort', 'rename',
        'calculate_column', 'use_llm', 'rag'
    ] = Field(..., description="実行する操作の種類")
    inputs: InputSource = Field(..., description="操作の入力元")
    params: Union[
        LoadParams, SelectParams, FilterParams, GroupByParams, MergeParams,
        CalculateScalarParams, ScalarArithmeticParams, CalculateColumnParams,
        DropDuplicatesParams, SortParams, RenameParams, UseLLMParams,
        RAGParams
    ] = Field(..., description="操作固有のパラメータ")
    outputs: OutputDestination = Field(..., description="操作の出力先")
    description: Optional[str] = None

class AnalysisPlan(BaseModel):
    plan: List[OperationStep]

from typing import TypedDict
class GraphState(TypedDict):
    query: str
    plan_json: Optional[Dict]
    validated_plan: Optional[AnalysisPlan]
    dataframes: Dict[str, pd.DataFrame]
    intermediate_results: Dict[str, Any]
    execution_log: List[Dict[str, Any]]
    error_message: Optional[str]
    messages: Annotated[List[AnyMessage], operator.add]
    embedding_manager: Optional[Any] 