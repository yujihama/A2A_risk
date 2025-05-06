from ..core.models import AnalysisPlan, GraphState
from ..core.ops_map import OPERATION_PARAM_MAP
from ..core.logging_config import setup_logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
import json
import operator

logger = setup_logger(__name__)

async def plan_generator_node(state: GraphState) -> dict:
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
- **データ操作以外のクエリの場合** データ操作で回答が得られない場合はRAGを用いてクエリに回答することも可能です（自然言語の抽出や要約など）。またRAGの場合はデータ絞り込みも内部で実施するため、事前、事後のデータ操作は不要です。

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
        step_issues = []  # ← ここで必ず初期化
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
        {json.dumps(df_schema_info, ensure_ascii=False, indent=2)}

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
            step_errors = []  # 各ステップのエラーを格納
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
            
            # --- カラム存在チェック ---
            # filter
            if op_type == 'filter' and hasattr(actual_params, 'conditions'):
                df_name = step.inputs.dataframe
                if df_name and df_name in dataframes:
                    df_columns = list(dataframes[df_name].columns)
                    for condition in actual_params.conditions:
                        if condition.column not in df_columns:
                            msg = f"filter条件で指定されたカラム '{condition.column}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
                for condition in actual_params.conditions:
                    if hasattr(condition, 'value_ref') and condition.value_ref:
                        referenced_scalars.add(condition.value_ref)
            # select
            if op_type == 'select' and hasattr(actual_params, 'columns'):
                df_name = step.inputs.dataframe
                if df_name and df_name in dataframes:
                    df_columns = list(dataframes[df_name].columns)
                    for col in actual_params.columns:
                        if col not in df_columns:
                            msg = f"selectで指定されたカラム '{col}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
            # group_by
            if op_type == 'group_by' and hasattr(actual_params, 'group_keys') and hasattr(actual_params, 'aggregations'):
                df_name = step.inputs.dataframe
                if df_name and df_name in dataframes:
                    df_columns = list(dataframes[df_name].columns)
                    for key in actual_params.group_keys:
                        if key not in df_columns:
                            msg = f"group_byのgroup_keysで指定されたカラム '{key}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
                    for agg_name, agg_spec in actual_params.aggregations.items():
                        if agg_spec.column not in df_columns:
                            msg = f"group_byのaggregationsで指定されたカラム '{agg_spec.column}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
            # calculate_scalar
            if op_type == 'calculate_scalar' and hasattr(actual_params, 'column'):
                df_name = step.inputs.dataframe
                if df_name and df_name in dataframes:
                    df_columns = list(dataframes[df_name].columns)
                    if actual_params.column not in df_columns:
                        msg = f"calculate_scalarで指定されたカラム '{actual_params.column}' はデータフレーム '{df_name}' に存在しません。"
                        logger.warning(f"ステップ {step.step_id}: {msg}")
                        error_count += 1
                        step_errors.append(msg)
            # calculate_column
            if op_type == 'calculate_column':
                df_name = step.inputs.dataframe
                if df_name and df_name in dataframes:
                    df_columns = list(dataframes[df_name].columns)
                    # left_column
                    if hasattr(actual_params, 'left_column') and actual_params.left_column:
                        if actual_params.left_column not in df_columns:
                            msg = f"calculate_columnのleft_columnで指定されたカラム '{actual_params.left_column}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
                    # right_column
                    if hasattr(actual_params, 'right_column') and actual_params.right_column:
                        if actual_params.right_column not in df_columns:
                            msg = f"calculate_columnのright_columnで指定されたカラム '{actual_params.right_column}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
                    # condition_column (if_else)
                    if hasattr(actual_params, 'condition_column') and actual_params.condition_column:
                        if actual_params.condition_column not in df_columns:
                            msg = f"calculate_columnのcondition_columnで指定されたカラム '{actual_params.condition_column}' はデータフレーム '{df_name}' に存在しません。"
                            logger.warning(f"ステップ {step.step_id}: {msg}")
                            error_count += 1
                            step_errors.append(msg)
            # use_llm
            if op_type == 'use_llm' and hasattr(actual_params, 'column_name'):
                df_name = step.inputs.dataframe
                if df_name and df_name in dataframes:
                    df_columns = list(dataframes[df_name].columns)
                    if actual_params.column_name not in df_columns:
                        msg = f"use_llmで指定されたカラム '{actual_params.column_name}' はデータフレーム '{df_name}' に存在しません。"
                        logger.warning(f"ステップ {step.step_id}: {msg}")
                        error_count += 1
                        step_errors.append(msg)
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
                step_errors.append(f"操作タイプ '{op_type}' に対して、期待されるパラメータクラス '{expected_param_class.__name__}' ではありません")
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
                    step_errors.append(f"必須パラメータが不足しています: {', '.join(missing_fields)}")
                else:
                    logger.info(f"ステップ {step.step_id}: 必須パラメータはすべて設定されています")
                
                # 入出力の存在確認
                if not step.inputs.dataframe and not step.inputs.left_dataframe and not step.inputs.right_dataframe and not step.inputs.scalars:
                    logger.warning(f"ステップ {step.step_id}: 入力が指定されていません")
                    warning_count += 1
                    step_errors.append("入力が指定されていません")
                
                if not step.outputs.dataframe and not step.outputs.scalar:
                    logger.warning(f"ステップ {step.step_id}: 出力が指定されていません")
                    warning_count += 1
                    step_errors.append("出力が指定されていません")
            # ステップに問題があればリストに追加
            if step_errors:
                step_issues.append({
                    "step_id": step.step_id,
                    "operation": step.operation,
                    "errors": step_errors
                })
        
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