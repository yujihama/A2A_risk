from typing import List, Dict, Any, Optional, Literal, Union, Type
from pydantic import BaseModel, Field, validator, ValidationError, model_validator
from pydantic_core import PydanticUndefined
from langchain_openai import ChatOpenAI

# --- 共通コンポーネント ---

class InputSource(BaseModel):
    """操作の入力元を指定"""
    dataframe: Optional[str] = None
    left_dataframe: Optional[str] = None
    right_dataframe: Optional[str] = None
    scalars: Optional[List[str]] = None # 使用する中間スカラー結果のキー名リスト

class OutputDestination(BaseModel):
    """操作の出力先を指定"""
    dataframe: Optional[str] = None # 結果を格納するDF名 (上書き or 新規)
    scalar: Optional[str] = None    # 結果を格納するスカラー名

class AggregationSpec(BaseModel):
    """単一の集計指定"""
    column: str # 集計対象の列名
    func: Literal['mean', 'sum', 'count', 'min', 'max', 'first', 'last', 'nunique'] # 集計関数

class FilterCondition(BaseModel):
    """単一のフィルター条件"""
    column: str
    operator: Literal[
        '==', '!=', '>', '>=', '<', '<=',
        'in', 'not in',
        'contains', 'not contains',
        'startswith', 'endswith',
        'isna', 'notna'
    ]
    value: Optional[Any] = None       # リテラル値
    value_ref: Optional[str] = None   # 中間スカラー結果のキー名を参照

    # Pydantic V1の @validator の代わりに V2の @model_validator を使用
    @model_validator(mode='after') # モデル全体のフィールドが揃った後に実行
    def check_value_logic(self) -> 'FilterCondition':
        # self を使って各フィールドにアクセス
        op = self.operator
        value_is_set = self.value is not None
        ref_is_set = self.value_ref is not None

        if op not in ['isna', 'notna']:
            # isna/notna 以外: value か value_ref のどちらか一方だけが必要
            if not (value_is_set ^ ref_is_set): # XOR: どちらか一方だけTrueの場合True
                raise ValueError(f"演算子 '{op}' では 'value' または 'value_ref' のどちらか一方のみを指定してください。")
        elif value_is_set or ref_is_set:
            # isna/notna: value も value_ref も指定してはいけない
            raise ValueError(f"演算子 '{op}' では 'value' や 'value_ref' は不要です。")

        # 検証が成功したら self を返す
        return self


# --- 各操作のパラメータ定義 ---

class LoadParams(BaseModel):
    source_path: str # ファイルパスなど
    # 必要に応じて他のパラメータ (シート名、区切り文字など) を追加

class SelectParams(BaseModel):
    columns: List[str]
    drop_duplicates: Optional[bool] = False
    subset_for_duplicates: Optional[List[str]] = None # 重複判定の対象列

class FilterParams(BaseModel):
    conditions: List[FilterCondition]
    condition_logic: Optional[Literal['AND', 'OR']] = 'AND'

class GroupByParams(BaseModel):
    group_keys: List[str]
    # 出力列名をキー、集計仕様を値とする辞書
    aggregations: Dict[str, AggregationSpec]

    @model_validator(mode='after')
    def check_aggregations_not_empty(self) -> 'GroupByParams':
        if not self.aggregations:
            raise ValueError("aggregationsは空にできません。少なくとも1つ以上の集計指定が必要です。")
        return self

class MergeParams(BaseModel):
    on: List[str] # 結合キー (単一でもリストで渡す)
    how: Literal['left', 'right', 'inner', 'outer'] = 'inner'
    suffixes: Optional[List[str]] = None # 例: ["_left", "_right"]

class CalculateScalarParams(BaseModel):
    column: str
    aggregation: Literal['mean', 'sum', 'count', 'min', 'max', 'first', 'last', 'nunique']
    # 必要に応じてフィルター条件などを追加可能

class ScalarArithmeticParams(BaseModel):
    # 例: expression: "scalar1 * 2 + scalar2"
    # または operator/operand形式
    expression: Optional[str] = None
    operator: Optional[Literal['add', 'subtract', 'multiply', 'divide']] = None
    operand1_ref: Optional[str] = None # スカラー参照
    operand1_literal: Optional[Any] = None
    operand2_ref: Optional[str] = None # スカラー参照
    operand2_literal: Optional[Any] = None
    # ここはより厳密な定義が必要 (演算によって必要なオペランドが変わる)

class CalculateColumnParams(BaseModel):
    new_column_name: str
    # 構造化された演算定義
    operation: Literal['add', 'subtract', 'multiply', 'divide', 'compare', 'if_else']
    left_column: Optional[str] = None  # 左側のカラム名
    right_column: Optional[str] = None  # 右側のカラム名
    left_value: Optional[Any] = None  # 左側の固定値（カラム名の代わり）
    right_value: Optional[Any] = None  # 右側の固定値（カラム名の代わり）
    left_scalar_ref: Optional[str] = None  # 左側の中間結果スカラー参照
    right_scalar_ref: Optional[str] = None  # 右側の中間結果スカラー参照
    # 比較演算子（compare操作時に使用）
    compare_operator: Optional[Literal['==', '!=', '>', '>=', '<', '<=']] = None
    # 条件演算（if_else操作時に使用）
    condition_column: Optional[str] = None  # 条件となる列
    true_value: Optional[Any] = None  # 条件が真の場合の値
    false_value: Optional[Any] = None  # 条件が偽の場合の値

    @model_validator(mode='after')
    def validate_operands(self) -> 'CalculateColumnParams':
        """左右のオペランドの指定が適切かチェック"""
        # 左側オペランドのチェック（column, value, scalar_refのいずれか1つのみ指定可能）
        left_options = [self.left_column is not None, 
                        self.left_value is not None, 
                        self.left_scalar_ref is not None]
        if sum(left_options) != 1:
            raise ValueError("左側オペランドは列名、固定値、スカラー参照のいずれか1つのみを指定してください")
            
        # if_else操作以外の場合の右側オペランドのチェック
        if self.operation != 'if_else':
            right_options = [self.right_column is not None, 
                            self.right_value is not None, 
                            self.right_scalar_ref is not None]
            if sum(right_options) != 1:
                raise ValueError("右側オペランドは列名、固定値、スカラー参照のいずれか1つのみを指定してください")
                
        # compare操作の場合は比較演算子が必要
        if self.operation == 'compare' and not self.compare_operator:
            raise ValueError("compare操作では比較演算子を指定してください")
            
        # if_else操作の場合は条件列と真偽値に対応する値が必要
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
    columns: Dict[str, str] # {"古い列名": "新しい列名"}

# --- 操作ステップ本体 ---

class OperationStep(BaseModel):
    step_id: str = Field(..., description="各ステップの一意な識別子")
    operation: Literal[
        'load', 'select', 'filter', 'group_by', 'merge',
        'calculate_scalar', 'scalar_arithmetic', 'drop_duplicates', 'sort', 'rename',
        'calculate_column'
        # 必要に応じて操作タイプを追加
    ] = Field(..., description="実行する操作の種類")
    inputs: InputSource = Field(..., description="操作の入力元")
    params: Union[
        LoadParams, SelectParams, FilterParams, GroupByParams, MergeParams,
        CalculateScalarParams, ScalarArithmeticParams, CalculateColumnParams,
        DropDuplicatesParams, SortParams, RenameParams
        # 対応するパラメータクラスを追加
    ] = Field(..., description="操作固有のパラメータ")
    outputs: OutputDestination = Field(..., description="操作の出力先")
    description: Optional[str] = None # LLMが生成する説明

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
}

query = str({
    "description": "発注情報のデータから、担当者ごとに過去1年間の取引の平均単価と最新の取引単価を抽出してください。対象とするデータは、担当者ID、担当者名、取引先名、品目名、過去平均単価、最新取引単価です。",
    "query": "担当者ごとの過去1年間の平均単価と最新の取引単価を抽出してください。対象期間は過去1年間です。出力には担当者ID、担当者名、取引先名、品目名、過去平均単価、最新取引単価のカラムを含めてください。",
    "expected_output": "担当者ID、担当者名、取引先名、品目名、過去平均単価、最新取引単価の表データ。"
  })

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, TypedDict
import logging

# --- LangGraph State ---
class GraphState(TypedDict):
    query: str
    plan_json: Optional[Dict] # LLMが生成した生のJSON
    validated_plan: Optional[AnalysisPlan] # Pydanticで検証済みのプラン
    dataframes: Dict[str, pd.DataFrame] # DF名 -> DataFrameオブジェクト
    intermediate_results: Dict[str, Any] # スカラー名 -> 値
    execution_log: List[Dict[str, Any]] # 実行ログ
    error_message: Optional[str]

# --- ロガー設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- plan_generator_node ---
async def plan_generator_node(state: GraphState) -> Dict[str, Any]: # async def に変更
    """
    ユーザーの自然言語クエリとデータフレーム情報から、LLMを用いてAnalysisPlan(JSON)を生成し、Pydanticで検証するLangGraphノード。
    """
    import json

    query = state.get("query")
    dataframes = state.get("dataframes", {})
    error_message = None

    # データフレームのスキーマ情報を抽出
    df_schema_info = {}
    for df_name, df in dataframes.items():
        df_schema_info[df_name] = {
            "columns": list(df.columns),
            "description": "自動抽出"
        }
    if not df_schema_info:
        error_message = "データフレーム情報がありません。"
        return {**state, "error_message": error_message}

    # AnalysisPlanのスキーマ
    analysis_plan_schema_json = json.dumps(AnalysisPlan.model_json_schema(), ensure_ascii=False, indent=2)

    # プロンプト組み立て
    prompt = f"""
# 指示
以下の自然言語クエリとデータフレーム構造に基づき、クエリの目的を達成するための段階的なデータ操作手順リスト（日本語）に分解してください。

# ゴールと要件
生成される手順リストは、以下の条件を満たす必要があります。
- 各ステップは、データ分析に詳しくない人でも理解できるレベルまで、明確かつ具体的に記述する。
- 各ステップは論理的な順序で並んでいる。
- **中間データの命名と参照:** 各ステップで生成される主要な結果（新しいデータフレームや計算された重要な集計値/スカラー値）には、括弧書きなどで分かりやすい一時的な名前（例: `(df_selected)`, `(overall_avg_price)`）を付けてください。後続のステップでその中間データや値を参照する場合は、その名前を使って明確に指示してください。
- **データフロー:** 基本的な処理の流れは、前のステップの結果を入力として使いますが、必要に応じて**元のデータフレーム**や途中で計算・保存した**集計値（スカラー値）**を再度参照する場合があることを明確に記述してください。
- **計算対象の明確化:** 特に、全体の集計値（例: 全体の平均、全体の合計など）を計算する場合は、**どのデータフレームのどの列**を対象とするかを明確に記述してください（例: `df_main`の`単価`列全体の平均）。グループ化後のデータを使うのか、元のデータを使うのかを区別してください。
- **操作の粒度:** 原則として、一つのステップでは一つの主要なデータフレーム操作（例: filter, select, group_by, calculate_scalar, merge）を行うように、操作を細分化してください。

# 入力情報
## 自然言語クエリ:
{query}

## 利用可能なデータフレーム
{json.dumps(df_schema_info, ensure_ascii=False, indent=2)}

## 操作の候補
{list(OPERATION_PARAM_MAP.keys())}

# 出力フォーマット
以下のJSON形式（ステップ番号と日本語の説明を持つオブジェクトのリスト）で回答してください。説明文の中に、入力データ名と出力データ名を明記してください。

```json
[
  {{
    "step": "1_XXX",
    "description": "（入力データ名）を使って、（操作内容）。結果を（出力データ名）として保存する。"
  }},
  {{
    "step": "2_XXX",
    "description": "（前のステップの出力データ名）を使って、（操作内容）。結果を（出力データ名）として保存する。"
  }},
  {{
    "step": "3_XXX",
    "description": "..."
  }}
]
    """

    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        # LLM呼び出し (async) - この関数はasyncではないので、普通にinvokeを使用する
        # response = llm.invoke([
        response = await llm.ainvoke([
            {"role": "system", "content": "あなたは自然言語クエリをpandasデータフレーム操作に正確に変換するエキスパートです。"},
            {"role": "user", "content": prompt},
        ])
        user_query = response.content.strip()
        logger.info(f"user_query: {user_query}")

        # プロンプト組み立て    
        analysis_plan_schema_json = json.dumps(AnalysisPlan.model_json_schema(), indent=2, ensure_ascii=False)

        prompt = f"""
        # ROLE: データ分析プランナー
        あなたは、ユーザーの自然言語による分析リクエストを理解し、それを段階的なデータ操作プランに変換する専門家です。
        生成するプランは、後続のシステムが自動的に実行できるような、厳密に定義されたJSON形式に従う必要があります。

        # INPUT CONTEXT:
        ## Available DataFrames:
        以下は現在利用可能なデータフレームとそのスキーマ情報です。プランニング時にはこれらの情報を使用してください。
        ```json
        {{
        "df_main": {{
            "columns": {{
            '発注ID', '発注日', '担当者ID', '担当者名', '取引先ID', '取引先名', '品目名', '単価', '数量','発注金額', '稟議ID'
            }},
            "description": "主要な売上トランザクションデータ"
        }},
        }}

        基準日：2025-01-01
        ```

        # User Query:
        {str(user_query)}

        # Available Operations & Output Schema:
        生成するプランは、以下のJSONスキーマに厳密に従った List[OperationStep] 形式でなければなりません。
        各ステップは OperationStep オブジェクトとし、利用可能な operation タイプとその params 構造を守ってください。
        中間的なデータフレームや計算結果（スカラー値）には、後で参照できるように outputs で明確な名前 (dataframe または scalar) を付けてください。
        後続のステップで中間結果を使用する場合は、inputs の dataframe や scalars でその名前を指定し、params 内で参照する場合は value_ref を使用してください。
        各ステップの description フィールドには、そのステップが何を行っているかを簡潔に記述してください。

        ## JSONスキーマ
        {analysis_plan_schema_json}

        ## operationとparamsの対応 ※異なるoperationとparamsを組み合わせないようにしてください。
        {OPERATION_PARAM_MAP}

        # OUTPUT INSTRUCTIONS:
        上記の指示に従い、ユーザーのクエリを実行するためのJSONプランを生成してください。JSON以外のテキストは含めないでください。

        """

        # LLM呼び出し - 普通にinvokeを使用し、structured_output機能を使う
        llm = ChatOpenAI(model="o3-mini")
        # response = llm.with_structured_output(
        response = await llm.with_structured_output(
            AnalysisPlan,
            method="json_mode"
        # ).invoke([
        ).ainvoke([
            {"role": "system", "content": "あなたは自然言語のクエリをデータフレーム操作に変換する専門家です。"},
            {"role": "user", "content": prompt}
        ])

        # 生成されたJSONをOPERATION_PARAM_MAPに基づいてチェック
        logger.info("生成されたJSONの簡易チェックを実行します...")
        
        # 検証統計用の変数
        error_count = 0
        warning_count = 0
        created_dataframes = set()
        referenced_dataframes = set()
        created_scalars = set()
        referenced_scalars = set()
        
        for i, step in enumerate(response.plan):
            op_type = step.operation
            if op_type not in OPERATION_PARAM_MAP:
                logger.warning(f"ステップ {step.step_id}: 未定義の操作タイプ '{op_type}' が使用されています")
                warning_count += 1
                continue
                
            expected_param_class = OPERATION_PARAM_MAP[op_type]
            actual_params = step.params
            
            # 入力データフレームの参照チェック
            if step.inputs.dataframe:
                referenced_dataframes.add(step.inputs.dataframe)
            if step.inputs.left_dataframe:
                referenced_dataframes.add(step.inputs.left_dataframe)
            if step.inputs.right_dataframe:
                referenced_dataframes.add(step.inputs.right_dataframe)
            
            # スカラー値の参照チェック
            if step.inputs.scalars:
                for scalar in step.inputs.scalars:
                    referenced_scalars.add(scalar)
            
            # 出力データフレームとスカラーの作成チェック
            if step.outputs.dataframe:
                created_dataframes.add(step.outputs.dataframe)
            if step.outputs.scalar:
                created_scalars.add(step.outputs.scalar)
            
            # operationがfilterの場合、value_refによるスカラー参照もチェック
            if op_type == 'filter' and hasattr(actual_params, 'conditions'):
                for condition in actual_params.conditions:
                    if hasattr(condition, 'value_ref') and condition.value_ref:
                        referenced_scalars.add(condition.value_ref)
            
            # operationがscalar_arithmeticの場合のスカラー参照もチェック
            if op_type == 'scalar_arithmetic':
                if hasattr(actual_params, 'operand1_ref') and actual_params.operand1_ref:
                    referenced_scalars.add(actual_params.operand1_ref)
                if hasattr(actual_params, 'operand2_ref') and actual_params.operand2_ref:
                    referenced_scalars.add(actual_params.operand2_ref)
            
            # operationがcalculate_columnの場合のスカラー参照もチェック
            if op_type == 'calculate_column':
                if hasattr(actual_params, 'left_scalar_ref') and actual_params.left_scalar_ref:
                    referenced_scalars.add(actual_params.left_scalar_ref)
                if hasattr(actual_params, 'right_scalar_ref') and actual_params.right_scalar_ref:
                    referenced_scalars.add(actual_params.right_scalar_ref)
            
            # 実際のパラメータが期待されるクラスのインスタンスかチェック
            if not isinstance(actual_params, expected_param_class):
                logger.warning(f"ステップ {step.step_id}: 操作タイプ '{op_type}' に対して、期待されるパラメータクラス '{expected_param_class.__name__}' ではありません")
                error_count += 1
            else:
                logger.info(f"ステップ {step.step_id}: 操作タイプ '{op_type}' のパラメータ検証OK")
                
                # 詳細なパラメータチェック（必須フィールドの存在確認）
                param_schema = expected_param_class.model_json_schema()
                required_fields = param_schema.get('required', [])
                missing_fields = []
                
                for field in required_fields:
                    if not hasattr(actual_params, field) or getattr(actual_params, field) is None:
                        missing_fields.append(field)
                
                if missing_fields:
                    logger.warning(f"ステップ {step.step_id}: 必須パラメータが不足しています: {', '.join(missing_fields)}")
                    warning_count += 1
                else:
                    logger.info(f"ステップ {step.step_id}: 必須パラメータはすべて設定されています")
                
                # 入出力の存在確認
                if not step.inputs.dataframe and not step.inputs.left_dataframe and not step.inputs.right_dataframe and not step.inputs.scalars:
                    logger.warning(f"ステップ {step.step_id}: 入力が指定されていません")
                    warning_count += 1
                
                if not step.outputs.dataframe and not step.outputs.scalar:
                    logger.warning(f"ステップ {step.step_id}: 出力が指定されていません")
                    warning_count += 1
        
        # データフレーム参照関係の検証
        missing_dataframes = referenced_dataframes - created_dataframes - {'df_main'}  # df_mainは初期データとして存在
        if missing_dataframes:
            logger.warning(f"参照されているが作成されていないデータフレーム: {missing_dataframes}")
            warning_count += 1
        
        # スカラー値参照関係の検証
        missing_scalars = referenced_scalars - created_scalars
        if missing_scalars:
            logger.warning(f"参照されているが作成されていないスカラー: {missing_scalars}")
            warning_count += 1
        
        # 検証統計の出力
        logger.info(f"検証統計: 総ステップ数={len(response.plan)}, エラー={error_count}, 警告={warning_count}")
        logger.info(f"データフレーム: 作成={len(created_dataframes)}, 参照={len(referenced_dataframes)}, 未作成参照={len(missing_dataframes)}")
        logger.info(f"スカラー値: 作成={len(created_scalars)}, 参照={len(referenced_scalars)}, 未作成参照={len(missing_scalars)}")
        if error_count == 0 and warning_count == 0:
            logger.info("すべてのステップが検証に合格しました")
        
        # エラーや重大な警告がある場合、LLMに修正を依頼
        max_fix_attempts = 3  # 修正試行の最大回数
        current_attempt = 0
        
        # エラーメッセージと警告メッセージを収集
        error_messages = []
        if error_count > 0:
            error_messages.append(f"エラーが{error_count}件検出されました。")
        
        if missing_dataframes:
            error_messages.append(f"参照されているが作成されていないデータフレーム: {missing_dataframes}")
        
        if missing_scalars:
            error_messages.append(f"参照されているが作成されていないスカラー: {missing_scalars}")
        
        # ステップごとのエラーと警告を詳細に収集
        step_issues = []
        for i, step in enumerate(response.plan):
            step_errors = []
            
            # 操作タイプのチェック
            op_type = step.operation
            if op_type not in OPERATION_PARAM_MAP:
                step_errors.append(f"未定義の操作タイプ '{op_type}' が使用されています")
                continue
                
            expected_param_class = OPERATION_PARAM_MAP[op_type]
            actual_params = step.params
            
            # パラメータ型のチェック
            if not isinstance(actual_params, expected_param_class):
                step_errors.append(f"操作タイプ '{op_type}' に対して、期待されるパラメータクラスは '{expected_param_class.__name__}' ですが、実際は '{type(actual_params).__name__}' です")
            
            # 必須パラメータのチェック
            param_schema = expected_param_class.model_json_schema()
            required_fields = param_schema.get('required', [])
            missing_fields = []
            
            for field in required_fields:
                if not hasattr(actual_params, field) or getattr(actual_params, field) is None:
                    missing_fields.append(field)
            
            if missing_fields:
                step_errors.append(f"必須パラメータが不足しています: {', '.join(missing_fields)}")
            
            # 入出力の存在確認
            if not step.inputs.dataframe and not step.inputs.left_dataframe and not step.inputs.right_dataframe and not step.inputs.scalars:
                step_errors.append("入力が指定されていません")
            
            if not step.outputs.dataframe and not step.outputs.scalar:
                step_errors.append("出力が指定されていません")
            
            # ステップに問題があればリストに追加
            if step_errors:
                step_issues.append({
                    "step_id": step.step_id,
                    "operation": step.operation,
                    "errors": step_errors
                })
        
        # エラーや重大な警告がある場合のみ修正を試みる
        original_plan = response
        while (error_count > 0 or len(missing_dataframes) > 0 or len(missing_scalars) > 0) and current_attempt < max_fix_attempts:
            current_attempt += 1
            logger.info(f"プラン修正を試行します (試行 {current_attempt}/{max_fix_attempts})...")
            
            # 修正用のプロンプトを作成
            fix_prompt = f"""
# エラー修正タスク
以下のAnalysisPlanに含まれるエラーを全て修正してください。

## 修正すべき問題（カラム）
{error_messages}

## 修正すべき問題（パラメータ）
{json.dumps(step_issues, indent=2, ensure_ascii=False)}

## 現在のプラン（修正対象）
```json
{json.dumps(original_plan.model_dump(), indent=2, ensure_ascii=False)}
```

## 修正方法
1. 各ステップの操作タイプに対して適切なパラメータクラスを使用してください。
   - 例: 'filter'操作は'FilterParams'を使用する必要があります。
2. 各パラメータクラスの必須フィールドが設定されていることを確認してください。
3. すべてのデータフレームとスカラー値が使用前に作成されていることを確認してください。
4. 各ステップの入力と出力が適切に設定されていることを確認してください。

修正したプランのJSONのみを返してください。他のテキストは含めないでください。
            """
            
            try:
                logger.info("LLMにプラン修正を依頼しています...")
                logger.info(f"修正用のプロンプト: {fix_prompt}")
                
                # LLM呼び出しで修正を試みる
                fix_llm = ChatOpenAI(model="gpt-4.1-mini")
                # fix_response = fix_llm.with_structured_output(
                fix_response = await fix_llm.with_structured_output(
                    AnalysisPlan,
                    method="json_mode"
                # ).invoke([
                ).ainvoke([
                    {"role": "system", "content": "あなたはデータ分析プランのエラー修正を行う専門家です。検出されたエラーに基づいて、プランを正確に修正してください。"},
                    {"role": "user", "content": fix_prompt}
                ])
                
                # 修正されたプランを検証
                logger.info("修正されたプランを検証します...")
                
                # 検証用の変数をリセット
                error_count = 0
                warning_count = 0
                created_dataframes = set()
                referenced_dataframes = set()
                created_scalars = set()
                referenced_scalars = set()
                
                # 修正されたプランを再検証
                for i, step in enumerate(fix_response.plan):
                    op_type = step.operation
                    if op_type not in OPERATION_PARAM_MAP:
                        logger.warning(f"修正後もステップ {step.step_id}: 未定義の操作タイプ '{op_type}' が使用されています")
                        warning_count += 1
                        continue
                        
                    expected_param_class = OPERATION_PARAM_MAP[op_type]
                    actual_params = step.params
                    
                    # 入力データフレームの参照チェック
                    if step.inputs.dataframe:
                        referenced_dataframes.add(step.inputs.dataframe)
                    if step.inputs.left_dataframe:
                        referenced_dataframes.add(step.inputs.left_dataframe)
                    if step.inputs.right_dataframe:
                        referenced_dataframes.add(step.inputs.right_dataframe)
                    
                    # スカラー値の参照チェック
                    if step.inputs.scalars:
                        for scalar in step.inputs.scalars:
                            referenced_scalars.add(scalar)
                    
                    # 出力データフレームとスカラーの作成チェック
                    if step.outputs.dataframe:
                        created_dataframes.add(step.outputs.dataframe)
                    if step.outputs.scalar:
                        created_scalars.add(step.outputs.scalar)
                    
                    # その他の参照チェック（filter, scalar_arithmetic, calculate_column）
                    if op_type == 'filter' and hasattr(actual_params, 'conditions'):
                        for condition in actual_params.conditions:
                            if hasattr(condition, 'value_ref') and condition.value_ref:
                                referenced_scalars.add(condition.value_ref)
                    
                    if op_type == 'scalar_arithmetic':
                        if hasattr(actual_params, 'operand1_ref') and actual_params.operand1_ref:
                            referenced_scalars.add(actual_params.operand1_ref)
                        if hasattr(actual_params, 'operand2_ref') and actual_params.operand2_ref:
                            referenced_scalars.add(actual_params.operand2_ref)
                    
                    if op_type == 'calculate_column':
                        if hasattr(actual_params, 'left_scalar_ref') and actual_params.left_scalar_ref:
                            referenced_scalars.add(actual_params.left_scalar_ref)
                        if hasattr(actual_params, 'right_scalar_ref') and actual_params.right_scalar_ref:
                            referenced_scalars.add(actual_params.right_scalar_ref)
                    
                    # パラメータ型チェック
                    if not isinstance(actual_params, expected_param_class):
                        logger.warning(f"修正後もステップ {step.step_id}: 操作タイプ '{op_type}' に対して期待されるパラメータクラスではありません")
                        error_count += 1
                
                # データフレームとスカラーの参照関係を検証
                missing_dataframes = referenced_dataframes - created_dataframes - {'df_main'}
                missing_scalars = referenced_scalars - created_scalars
                
                if missing_dataframes:
                    logger.warning(f"修正後も参照されているが作成されていないデータフレームがあります: {missing_dataframes}")
                    warning_count += 1
                
                if missing_scalars:
                    logger.warning(f"修正後も参照されているが作成されていないスカラーがあります: {missing_scalars}")
                    warning_count += 1
                
                # 修正の結果を出力
                logger.info(f"修正試行 {current_attempt} 後の検証統計: エラー={error_count}, 警告={warning_count}")
                
                # エラーが解消されていれば修正を採用
                if error_count == 0 and len(missing_dataframes) == 0 and len(missing_scalars) == 0:
                    logger.info("修正が成功しました。修正後のプランを採用します。")
                    response = fix_response
                    break
                else:
                    # エラーメッセージと問題ステップを再収集
                    error_messages.clear()
                    step_issues.clear()
                    
                    if error_count > 0:
                        error_messages.append(f"修正後もエラーが{error_count}件残っています。")
                    
                    if missing_dataframes:
                        error_messages.append(f"修正後も参照されているが作成されていないデータフレーム: {missing_dataframes}")
                    
                    if missing_scalars:
                        error_messages.append(f"修正後も参照されているが作成されていないスカラー: {missing_scalars}")
                    
                    # ステップごとの問題を再収集
                    for i, step in enumerate(fix_response.plan):
                        step_errors = []
                        
                        # 操作タイプのチェック
                        op_type = step.operation
                        if op_type not in OPERATION_PARAM_MAP:
                            step_errors.append(f"未定義の操作タイプ '{op_type}' が使用されています")
                            continue
                            
                        expected_param_class = OPERATION_PARAM_MAP[op_type]
                        actual_params = step.params
                        
                        # パラメータ型のチェック
                        if not isinstance(actual_params, expected_param_class):
                            step_errors.append(f"操作タイプ '{op_type}' に対して、期待されるパラメータクラスは '{expected_param_class.__name__}' ですが、実際は '{type(actual_params).__name__}' です")
                        
                        # 必須パラメータのチェック
                        param_schema = expected_param_class.model_json_schema()
                        required_fields = param_schema.get('required', [])
                        missing_fields = []
                        
                        for field in required_fields:
                            if not hasattr(actual_params, field) or getattr(actual_params, field) is None:
                                missing_fields.append(field)
                        
                        if missing_fields:
                            step_errors.append(f"必須パラメータが不足しています: {', '.join(missing_fields)}")
                        
                        # 入出力の存在確認
                        if not step.inputs.dataframe and not step.inputs.left_dataframe and not step.inputs.right_dataframe and not step.inputs.scalars:
                            step_errors.append("入力が指定されていません")
                        
                        if not step.outputs.dataframe and not step.outputs.scalar:
                            step_errors.append("出力が指定されていません")
                        
                        # ステップに問題があればリストに追加
                        if step_errors:
                            step_issues.append({
                                "step_id": step.step_id,
                                "operation": step.operation,
                                "errors": step_errors
                            })
                    
                    original_plan = fix_response  # 次の修正のベースとなるプラン
            
            except Exception as e:
                logger.error(f"プラン修正中にエラーが発生しました: {e}", exc_info=True)
                break  # エラーが発生した場合は修正を中止
        
        if current_attempt >= max_fix_attempts and (error_count > 0 or len(missing_dataframes) > 0 or len(missing_scalars) > 0):
            logger.warning(f"最大試行回数 ({max_fix_attempts}) に達しましたが、エラーや警告が残っています。")
        
        # 生成されたプランをStateに格納して返す
        logger.info(f"生成されたプランのステップ数: {len(response.plan)}")
        logger.info(f"生成されたプラン: {response.plan}")
        return {**state, "validated_plan": response, "error_message": None}
    
    except Exception as e:
        logger.error(f"プラン生成中にエラーが発生しました: {e}", exc_info=True)
        return {**state, "error_message": f"プラン生成エラー: {str(e)}"}


# --- 型変換と適切な比較を行うヘルパー関数 ---
def safe_compare(series: pd.Series, operator: str, value: any) -> pd.Series:
    """
    データ型を考慮して安全に比較を行い、結果の真偽値マスクを返す。

    Args:
        series: 比較対象のPandas Series。
        operator: 比較演算子 ('>', '>=', '<', '<=', '==', '!=' など)。
        value: 比較する値。

    Returns:
        比較結果のブール値 Pandas Series。

    Raises:
        ValueError: 比較や型変換中にエラーが発生した場合。
        NotImplementedError: サポートされていない演算子の場合。
    """
    original_value = value # エラーメッセージ用に元の値を保持

    # isna/notna は値が不要
    if operator == 'isna':
        return series.isna()
    if operator == 'notna':
        return series.notna()

    # --- 文字列操作 ---
    if operator in ['contains', 'not contains', 'startswith', 'endswith']:
        # Seriesとvalueを文字列に変換して比較
        str_series = series.astype(str)
        str_value = str(value) if value is not None else '' # Noneの場合は空文字として扱うなど考慮
        try:
            if operator == 'contains':
                return str_series.str.contains(str_value, case=False, na=False)
            if operator == 'not contains':
                return ~str_series.str.contains(str_value, case=False, na=False)
            if operator == 'startswith':
                return str_series.str.startswith(str_value, na=False)
            if operator == 'endswith':
                return str_series.str.endswith(str_value, na=False)
        except Exception as e:
             raise ValueError(f"文字列操作 '{operator}' (値: {original_value}) 実行中にエラー: {e}")

    # --- 'in' / 'not in' ---
    if operator in ['in', 'not in']:
        if not isinstance(value, list):
            value = [value] # リストでなければリストにする
        # TODO: valueリスト内の要素とseriesの型を合わせる処理を追加するとより堅牢
        # 例: if pd.api.types.is_numeric_dtype(series): value = [pd.to_numeric(v, errors='coerce') for v in value]
        try:
            mask = series.isin(value)
            return mask if operator == 'in' else ~mask
        except Exception as e:
            # isinで型不一致などのエラーが起こる可能性がある
            raise ValueError(f"'{operator}' 操作 (値: {original_value}) 実行中にエラー: {e}")


    # --- 比較演算 (>、>=、<、<=、==、!=) ---
    # 比較値の型を推測・変換
    try:
        # まず数値に変換できるか試す
        num_value = pd.to_numeric(value)
        value_type = 'numeric'
        compare_val = num_value
    except (ValueError, TypeError):
        try:
            # 次に日付/時刻に変換できるか試す
            dt_value = pd.to_datetime(value)
            value_type = 'datetime'
            compare_val = dt_value
        except (ValueError, TypeError, OverflowError): # OverflowErrorも追加
            # どちらにも変換できなければ文字列として扱う
            value_type = 'string'
            compare_val = str(value) # 元々文字列かもしれないが念のためstr()

    # Seriesの型を比較値の型に合わせて変換・比較
    try:
        if value_type == 'numeric':
            numeric_series = pd.to_numeric(series, errors='coerce')
            if operator == '>': return (numeric_series > compare_val).fillna(False)
            if operator == '>=': return (numeric_series >= compare_val).fillna(False)
            if operator == '<': return (numeric_series < compare_val).fillna(False)
            if operator == '<=': return (numeric_series <= compare_val).fillna(False)
            if operator == '==': return (numeric_series == compare_val).fillna(False)
            if operator == '!=': return (numeric_series != compare_val).fillna(False)

        elif value_type == 'datetime':
            datetime_series = pd.to_datetime(series, errors='coerce')
            if operator == '>': return (datetime_series > compare_val).fillna(False)
            if operator == '>=': return (datetime_series >= compare_val).fillna(False)
            if operator == '<': return (datetime_series < compare_val).fillna(False)
            if operator == '<=': return (datetime_series <= compare_val).fillna(False)
            if operator == '==': return (datetime_series == compare_val).fillna(False)
            if operator == '!=': return (datetime_series != compare_val).fillna(False)

        elif value_type == 'string':
            # 文字列として比較 (==, != 以外は辞書順比較になるので注意)
            str_series = series.astype(str)
            if operator == '>': return str_series > compare_val
            if operator == '>=': return str_series >= compare_val
            if operator == '<': return str_series < compare_val
            if operator == '<=': return str_series <= compare_val
            if operator == '==': return str_series == compare_val
            if operator == '!=': return str_series != compare_val
        else:
             # このケースには到達しないはずだが念のため
             raise TypeError("予期せぬ比較値タイプです。")

    except Exception as e:
        # 型変換や比較中の予期せぬエラーをキャッチ
        raise ValueError(f"演算子 '{operator}' (値: {original_value}) 適用中にエラー: {e}")

    raise NotImplementedError(f"未対応または不正な演算子: {operator}")

# --- 実行ノード ---
def execute_plan_node(state: GraphState) -> Dict[str, Any]:
    logger.info("--- Node: execute_plan ---")
    plan = state.get("validated_plan")
    dataframes = state.get("dataframes", {})
    intermediate_results = state.get("intermediate_results", {})
    execution_log = state.get("execution_log", [])

    if not plan or not plan.plan:
        logger.error("実行する検証済みプランがありません。")
        return {"error_message": "実行する検証済みプランが見つかりません。", "execution_log": execution_log}
    if not dataframes:
         logger.warning("操作対象のデータフレームがありません。")
         # 空のDFなどを初期化するか、エラーにするかは要件次第

    current_dataframes = {name: df.copy() for name, df in dataframes.items()} # shallow copy
    current_intermediate_results = intermediate_results.copy()

    for i, step in enumerate(plan.plan):
        step_log = {"step_id": step.step_id, "operation": step.operation, "status": "pending"}
        print(f"--- ステップ {i+1}/{len(plan.plan)} ({step.step_id}: {step.operation}) 実行開始 ---")
        print(f"Inputs: {step.inputs}")
        print(f"Params: {step.params}")
        print(f"Operation: {step.operation}")
        print(f"Outputs: {step.outputs}")

        try:
            # --- 入力データ/スカラーの取得と検証 ---
            input_df_map: Dict[str, pd.DataFrame] = {}
            if step.inputs.dataframe:
                if step.inputs.dataframe not in current_dataframes:
                    raise ValueError(f"入力データフレーム '{step.inputs.dataframe}' が見つかりません。")
                input_df_map['main'] = current_dataframes[step.inputs.dataframe]
            if step.inputs.left_dataframe:
                if step.inputs.left_dataframe not in current_dataframes:
                     raise ValueError(f"左入力データフレーム '{step.inputs.left_dataframe}' が見つかりません。")
                input_df_map['left'] = current_dataframes[step.inputs.left_dataframe]
            if step.inputs.right_dataframe:
                 if step.inputs.right_dataframe not in current_dataframes:
                     raise ValueError(f"右入力データフレーム '{step.inputs.right_dataframe}' が見つかりません。")
                 input_df_map['right'] = current_dataframes[step.inputs.right_dataframe]

            input_scalars: Dict[str, Any] = {}
            if step.inputs.scalars:
                for scalar_key in step.inputs.scalars:
                    if scalar_key not in current_intermediate_results:
                        raise ValueError(f"入力スカラー '{scalar_key}' が中間結果に見つかりません。")
                    input_scalars[scalar_key] = current_intermediate_results[scalar_key]

            # --- 操作の実行 ---
            result_df: Optional[pd.DataFrame] = None
            result_scalar: Optional[Any] = None
            output_df_name = step.outputs.dataframe
            output_scalar_name = step.outputs.scalar

            # --- 各操作タイプのロジック ---
            if step.operation == 'load':
                # TODO: Loadの実装 (ファイルパスなどから読み込み)
                # params = cast(LoadParams, step.params)
                # result_df = pd.read_csv(params.source_path) # 例
                logger.warning("load 操作は未実装です。")
                pass

            elif step.operation == 'select':
                df = input_df_map['main']
                params = step.params # Pydanticが型を保証している想定
                # columnsが空リストの場合は全列を選択
                if not params.columns:
                    valid_cols = list(df.columns)
                else:
                    missing_cols = [c for c in params.columns if c not in df.columns]
                    if missing_cols:
                        logger.warning(f"列 {missing_cols} が存在しないため、選択から除外します。")
                    valid_cols = [c for c in params.columns if c in df.columns]
                if not valid_cols:
                     raise ValueError("選択可能な列がありません。")
                result_df = df[valid_cols].copy() # Copy推奨
                if params.drop_duplicates:
                     result_df = result_df.drop_duplicates(subset=params.subset_for_duplicates)
                     logger.info(f"重複行を除去しました (subset: {params.subset_for_duplicates})。")

            elif step.operation == 'filter':
                df = input_df_map['main']
                params = step.params
                mask = pd.Series(True, index=df.index) # Start with True

                for cond in params.conditions:
                    if cond.column not in df.columns:
                        raise ValueError(f"フィルタリング列 '{cond.column}' が存在しません。")
                    target_series = df[cond.column]
                    compare_value = None

                    if cond.value_ref:
                        if cond.value_ref not in current_intermediate_results:
                             raise ValueError(f"参照スカラー '{cond.value_ref}' が中間結果にありません。")
                        compare_value = current_intermediate_results[cond.value_ref]
                        logger.info(f"フィルタ条件で中間結果 '{cond.value_ref}' (値: {compare_value}) を使用。")
                    elif cond.operator not in ['isna', 'notna']:
                         compare_value = cond.value # リテラル値

                    current_mask = pd.Series(False, index=df.index) # Default to False for safety
                    # --- 演算子ごとの処理 ---
                    current_mask = safe_compare(target_series, cond.operator, compare_value)

                    # マスクを結合 (AND or OR)
                    if params.condition_logic == 'OR':
                        mask = mask | current_mask
                    else: # Default to AND
                        mask = mask & current_mask

                result_df = df[mask].copy()

            elif step.operation == 'group_by':
                df = input_df_map['main']
                params = step.params
                # パラメータの列が存在するかチェック
                missing_keys = [k for k in params.group_keys if k not in df.columns]
                if missing_keys: raise ValueError(f"グループ化キー列 {missing_keys} が存在しません。")
                agg_dict = {}
                for out_col, spec in params.aggregations.items():
                     if spec.column not in df.columns: raise ValueError(f"集計対象列 '{spec.column}' が存在しません。")
                     agg_dict[out_col] = (spec.column, spec.func) # Pandas agg形式に変換

                result_df = df.groupby(params.group_keys, as_index=False).agg(**agg_dict)

            elif step.operation == 'merge':
                 left_df = input_df_map['left']
                 right_df = input_df_map['right']
                 params: MergeParams = step.params # 型ヒントを追加すると分かりやすい

                 # suffixesを処理
                 merge_suffixes = params.suffixes
                 if merge_suffixes is None:
                  merge_suffixes = ('_x', '_y')
                 elif isinstance(merge_suffixes, list) and len(merge_suffixes) == 2:
                  merge_suffixes = tuple(merge_suffixes)

                 result_df = pd.merge(
                        left_df,
                        right_df,
                        on=params.on,
                        how=params.how,
                        suffixes=merge_suffixes # 処理済みのsuffixesを使用
                    )
                  
            elif step.operation == 'calculate_scalar':
                df = input_df_map['main']
                params = step.params
                if params.column not in df.columns: raise ValueError(f"計算対象列 '{params.column}' が存在しません。")
                target_series = df[params.column]
                # aggregationの種類に応じて適切なPandasメソッドを呼び出す
                if params.aggregation == 'mean':
                    result_scalar = target_series.mean()
                elif params.aggregation == 'sum':
                    result_scalar = target_series.sum()
                elif params.aggregation == 'count':
                    result_scalar = target_series.count() # Non-NA count
                elif params.aggregation == 'max':
                    result_scalar = target_series.max()
                elif params.aggregation == 'min':
                    result_scalar = target_series.min()
                elif params.aggregation == 'first':
                    result_scalar = target_series.iloc[0] if not target_series.empty else None
                elif params.aggregation == 'last':
                    result_scalar = target_series.iloc[-1] if not target_series.empty else None
                elif params.aggregation == 'nunique':
                    result_scalar = target_series.nunique()
                else:
                    raise NotImplementedError(f"未対応の集計関数: {params.aggregation}")
                # NumPy型を標準型に変換 (JSONシリアライズ可能にするため)
                if hasattr(result_scalar, 'item'): result_scalar = result_scalar.item()

            elif step.operation == 'scalar_arithmetic':
                params = step.params
                # expression 形式を優先、なければ operator 形式
                if params.expression:
                     # eval を安全に使うためのコンテキスト
                     eval_context = {"__builtins__": None} # 安全のため組み込み関数を無効化
                     eval_context.update(current_intermediate_results) # 中間結果をコンテキストに追加
                     try:
                         result_scalar = eval(params.expression, eval_context)
                     except Exception as e:
                          raise ValueError(f"スカラー演算式 '{params.expression}' の評価エラー: {e}")
                else:
                     # operator 形式の処理 (operand1, operand2 を解決)
                     op1 = None
                     if params.operand1_ref: op1 = current_intermediate_results[params.operand1_ref]
                     elif params.operand1_literal is not None: op1 = params.operand1_literal
                     else: raise ValueError("operand1 が指定されていません。")

                     op2 = None
                     if params.operand2_ref: op2 = current_intermediate_results[params.operand2_ref]
                     elif params.operand2_literal is not None: op2 = params.operand2_literal
                     else: raise ValueError("operand2 が指定されていません。")

                     if params.operator == 'multiply': result_scalar = op1 * op2
                     elif params.operator == 'add': result_scalar = op1 + op2
                     elif params.operator == 'subtract': result_scalar = op1 - op2
                     elif params.operator == 'divide': result_scalar = op1 / op2 if op2 != 0 else np.nan
                     else: raise NotImplementedError(f"未対応の演算子: {params.operator}")
                if hasattr(result_scalar, 'item'): result_scalar = result_scalar.item()


            elif step.operation == 'calculate_column':
                df = input_df_map['main']
                params = step.params
                result_df = df.copy()  # 元のDFをコピー
                
                # 左側オペランドの取得
                left_operand = None
                if params.left_column:
                    if params.left_column not in df.columns:
                        raise ValueError(f"指定された列 '{params.left_column}' が存在しません。")
                    left_operand = df[params.left_column]
                elif params.left_scalar_ref:
                    if params.left_scalar_ref not in current_intermediate_results:
                        raise ValueError(f"参照されたスカラー '{params.left_scalar_ref}' が存在しません。")
                    left_operand = current_intermediate_results[params.left_scalar_ref]
                elif params.left_value is not None:
                    left_operand = params.left_value
                
                # if_else操作以外の場合は右側オペランドが必要
                if params.operation != 'if_else':
                    right_operand = None
                    if params.right_column:
                        if params.right_column not in df.columns:
                            raise ValueError(f"指定された列 '{params.right_column}' が存在しません。")
                        right_operand = df[params.right_column]
                    elif params.right_scalar_ref:
                        if params.right_scalar_ref not in current_intermediate_results:
                            raise ValueError(f"参照されたスカラー '{params.right_scalar_ref}' が存在しません。")
                        right_operand = current_intermediate_results[params.right_scalar_ref]
                    elif params.right_value is not None:
                        right_operand = params.right_value
                
                # 演算の実行
                if params.operation == 'add':
                    result_df[params.new_column_name] = left_operand + right_operand
                elif params.operation == 'subtract':
                    result_df[params.new_column_name] = left_operand - right_operand
                elif params.operation == 'multiply':
                    result_df[params.new_column_name] = left_operand * right_operand
                elif params.operation == 'divide':
                    # ゼロ除算防止
                    if isinstance(right_operand, (int, float)) and right_operand == 0:
                        result_df[params.new_column_name] = np.nan
                    else:
                        result_df[params.new_column_name] = left_operand / right_operand
                elif params.operation == 'compare':
                    # 比較演算
                    if params.compare_operator == '==':
                        result_df[params.new_column_name] = left_operand == right_operand
                    elif params.compare_operator == '!=':
                        result_df[params.new_column_name] = left_operand != right_operand
                    elif params.compare_operator == '>':
                        result_df[params.new_column_name] = left_operand > right_operand
                    elif params.compare_operator == '>=':
                        result_df[params.new_column_name] = left_operand >= right_operand
                    elif params.compare_operator == '<':
                        result_df[params.new_column_name] = left_operand < right_operand
                    elif params.compare_operator == '<=':
                        result_df[params.new_column_name] = left_operand <= right_operand
                    else:
                        raise ValueError(f"未対応の比較演算子: {params.compare_operator}")
                elif params.operation == 'if_else':
                    # 条件分岐
                    if params.condition_column not in df.columns:
                        raise ValueError(f"条件列 '{params.condition_column}' が存在しません。")
                    
                    condition = df[params.condition_column]
                    
                    # boolean型でない場合はbooleanに変換
                    if not pd.api.types.is_bool_dtype(condition):
                        condition = condition.astype(bool)
                    
                    # 真偽値に応じた値を設定
                    if params.true_value is not None and params.false_value is not None:
                        # 両方とも指定されている場合
                        result_df[params.new_column_name] = np.where(condition, params.true_value, params.false_value)
                    elif params.true_value is not None:
                        # 真の場合のみ指定されている場合
                        result_df[params.new_column_name] = np.where(condition, params.true_value, np.nan)
                    elif params.false_value is not None:
                        # 偽の場合のみ指定されている場合
                        result_df[params.new_column_name] = np.where(condition, np.nan, params.false_value)
                else:
                    raise NotImplementedError(f"未対応の演算タイプ: {params.operation}")
                
                logger.info(f"列 '{params.new_column_name}' を構造化演算 '{params.operation}' で計算しました。")

            elif step.operation == 'drop_duplicates':
                 df = input_df_map['main']
                 params = step.params
                 result_df = df.drop_duplicates(subset=params.subset, keep=params.keep)

            elif step.operation == 'sort':
                 df = input_df_map['main']
                 params = step.params
                 result_df = df.sort_values(by=params.by, ascending=params.ascending)

            elif step.operation == 'rename':
                 df = input_df_map['main']
                 params = step.params
                 result_df = df.rename(columns=params.columns)

            else:
                raise NotImplementedError(f"未定義の操作タイプ: {step.operation}")

            # --- 出力結果の保存 ---
            if result_df is not None and output_df_name:
                current_dataframes[output_df_name] = result_df
                logger.info(f"ステップ完了。データフレーム '{output_df_name}' を更新/作成。形状: {result_df.shape}")
                print(f"{output_df_name}:{result_df}")
            elif result_scalar is not None and output_scalar_name:
                current_intermediate_results[output_scalar_name] = result_scalar
                logger.info(f"ステップ完了。中間スカラー '{output_scalar_name}' を保存。値: {result_scalar}")
                print(f"{output_scalar_name}:{result_scalar}")
            elif result_df is None and result_scalar is None:
                 logger.warning(f"ステップ {step.step_id} は結果を出力しませんでした。")


            step_log["status"] = "success"
            if result_df is not None: step_log["output_shape"] = result_df.shape
            if result_scalar is not None: step_log["output_scalar_value"] = result_scalar

            if (i+1) == len(plan.plan):
              current_dataframes["df_final_output"] = result_df

        except Exception as e:
            logger.error(f"ステップ {step.step_id} ({step.operation}) でエラー発生: {e}", exc_info=True)
            step_log["status"] = "error"
            step_log["error_message"] = str(e)
            execution_log.append(step_log) # エラーログを追加
            return {
                "dataframes": current_dataframes, # エラー発生時点の状態
                "intermediate_results": current_intermediate_results,
                "execution_log": execution_log,
                "error_message": f"ステップ {step.step_id} ({step.operation}) でエラー: {e}"
            }

        execution_log.append(step_log) # 成功ログを追加

    logger.info("--- 全プランの実行完了 ---")
    return {
        "dataframes": current_dataframes,
        "intermediate_results": current_intermediate_results,
        "execution_log": execution_log,
        "error_message": None # 成功時はエラーなし
    }

# --- LangGraph---
from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)
workflow.add_node("plan_generator", plan_generator_node)
workflow.add_node("execute_plan", execute_plan_node)

workflow.set_entry_point("plan_generator")
workflow.add_edge("plan_generator", "execute_plan")
workflow.add_edge("execute_plan", END)

app = workflow.compile()

# サンプル実行部分は削除し、実際の実行はtest_run_agent.pyなどで行う
# initial_state = {
#     "query": "取引先ごとの平均価格が全体の平均価格の2倍以上の取引先のIDと取引名を抽出する",
#     "dataframes": {"df_main": df}, # ここに初期DFを入れる
#     "intermediate_results": {},
#     "execution_log": [],
#     "error_message": None,
#     # plan_generatorで生成するため、validated_planはセットしない
# }
# result = app.invoke(initial_state)
# print("--- Final State ---")
# print("Error:", result.get("error_message"))
# print("Final DF names:", list(result.get("dataframes", {}).keys()))
# print(result.get("dataframes", {}).get("df_final_output")) # 最終結果DFの確認など

# 最後にQueryAgentクラスを追加（sample.pyの末尾）
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import os
import logging
import json
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

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
        
        # LLM設定
        # sample.pyですでに設定されているため、ここでは特に何もしない
        # LLMの初期化
        try:
            self.llm = ChatOpenAI(model=model, temperature=0)
            self.logger.info(f"LLMクライアントを初期化しました: model={model}")
        except Exception as e:
            self.logger.error(f"LLMクライアントの初期化に失敗しました: {e}")
            self.llm = None  # 初期化失敗

    def load_data(self, data_source: Any):
        """
        指定されたデータソースからデータを読み込み、データフレームと列情報を設定する

        Args:
            data_source (Any): データソース。ファイルパス(str)、pandas DataFrame、または辞書(DB接続情報など)。
        """
        self.data_source_info = data_source  # データソース情報を保存
        self.logger.info(f"データソースの読み込みを開始します: {type(data_source)}")
        
        try:
            if isinstance(data_source, str):
                # ファイルパスの場合
                file_path = data_source
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

                self.logger.info(f"ファイルを読み込みます: {file_path}")
                # ファイル拡張子に基づいて読み込み方法を選択
                if file_path.lower().endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.lower().endswith(('.xlsx', '.xls')):
                    self.data = pd.read_excel(file_path)
                else:
                    raise ValueError(f"サポートされていないファイル形式です: {file_path}")

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
            
            if final_df is not None:
                formatted_result = await self._format_results_as_text(query, execution_log, final_df, intermediate_results)
                if return_type == "df":
                    data = all_dfs  # すべてのDataFrameを辞書で返す
                else:
                    data = None
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
        if result_df.empty:
            return "クエリに一致するデータがありませんでした。"
        
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
            
            return f"クエリの結果は以下の通りです：\n\n{formatted_text}"
        except Exception as e:
            self.logger.error(f"結果のフォーマット中にエラーが発生しました: {e}", exc_info=True)
            # エラー時はシンプルな出力を返す
            if len(result_df) <= 20:
                df_str = result_df.to_string(index=False)
                return f"クエリの結果は以下の通りです：\n\n{df_str}"
            
            # 大きなデータフレームの場合は先頭10行と概要を表示
            head_str = result_df.head(10).to_string(index=False)
            return f"クエリの結果（全{len(result_df)}行中の先頭10行）：\n\n{head_str}\n\n合計 {len(result_df)} 行のデータが見つかりました。"
