from .models import (
    LoadParams, SelectParams, FilterParams, GroupByParams, MergeParams,
    CalculateScalarParams, ScalarArithmeticParams, CalculateColumnParams,
    DropDuplicatesParams, SortParams, RenameParams, UseLLMParams, RAGParams, BaseModel
)

OPERATION_PARAM_MAP = {
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