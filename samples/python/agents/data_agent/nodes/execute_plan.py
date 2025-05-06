from ..core.models import GraphState
from ..core.utils import safe_compare
from ..core.logging_config import setup_logger
from ..core.ops_map import OPERATION_PARAM_MAP
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, cast
from langchain_openai import ChatOpenAI
logger = setup_logger(__name__)

async def execute_plan_node(state: GraphState) -> Dict[str, Any]: # <-- Make async
    logger.info("--- Node: execute_plan ---")
    plan = state.get("validated_plan")
    dataframes = state.get("dataframes", {})
    intermediate_results = state.get("intermediate_results", {})
    execution_log = state.get("execution_log", [])
    embedding_mgr = state.get("embedding_manager") # Get embedding manager

    if not plan or not plan.plan:
        logger.error("実行する検証済みプランがありません。")
        return {"error_message": "実行する検証済みプランが見つかりません。", "execution_log": execution_log}
    if not dataframes:
         logger.warning("操作対象のデータフレームがありません。")
         # 空のDFなどを初期化するか、エラーにするかは要件次第

    current_dataframes = {name: df.copy() for name, df in dataframes.items()}
    current_intermediate_results = intermediate_results.copy()

    for i, step in enumerate(plan.plan):
        step_log = {"step_id": step.step_id, "operation": step.operation, "status": "pending"}
        logger.info(f"--- ステップ {i+1}/{len(plan.plan)} ({step.step_id}: {step.operation}) 実行開始 ---")
        logger.info(f"Inputs: {step.inputs}")
        logger.info(f"Params: {step.params}")
        logger.info(f"Operation: {step.operation}")
        logger.info(f"Outputs: {step.outputs}")

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
                 params = step.params # 型ヒントを追加すると分かりやすい

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
                        # Convert operands to numeric if necessary
                        try:
                            num_left = pd.to_numeric(left_operand)
                            num_right = pd.to_numeric(right_operand)
                            result_df[params.new_column_name] = num_left / num_right
                        except (TypeError, ValueError):
                            # Handle non-numeric division gracefully (e.g., NaN or raise error)
                            result_df[params.new_column_name] = np.nan
                            logger.warning(f"列 '{params.new_column_name}' の除算で非数値データが見つかりました。結果は NaN になります。")

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

            elif step.operation == 'use_llm':
                if 'main' in input_df_map and input_df_map['main'] is not None:
                    input = input_df_map['main']
                    print(f"use_llm: input_df_map['main']:{input.head()}")
                elif input_scalars is not None:
                    input = input_scalars
                else:
                    raise ValueError("入力データが指定されていません。")
                
                params = step.params
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(method="json_mode")
                
                if output_df_name:
                    for row in input.iterrows():
                        prompt = f"以下のデータフレームの{params.use_column_name}列のデータに対して、以下の結果をJSON形式で回答してください。\n{params.prompt}\n{str(row)}"
                        output_llm = llm.invoke(prompt)
                        row[params.output_column_name] = output_llm
                    result_df = input

                if output_scalar_name:
                    prompt = f"以下のデータフレームの{params.use_column_name}列のデータに対して、以下の結果をJSON形式で回答してください。\n{params.prompt}\n{input.to_string()}"
                    output_llm = llm.invoke(prompt)
                    result_scalar = output_llm

            elif step.operation == 'rag':
                if not embedding_mgr or not embedding_mgr.vectorstore:
                    raise ValueError("RAG操作を実行するには、構築済みのベクトルストアを持つEmbeddingManagerが必要です。")

                params = step.params # Pydanticが型を保証
                user_query = state.get("query") # 元のユーザー質問を取得

                # 1. 関連ドキュメントを検索
                logger.info(f"RAG: ベクトル検索を実行中 (query='{params.retrieval_query}', k={params.top_k})")
                retrieved_docs = embedding_mgr.search(params.retrieval_query, k=params.top_k)
                context = "\\n---\\n".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"RAG: {len(retrieved_docs)}件のドキュメントを検索しました。Context length={len(context)}")
                if not retrieved_docs:
                    logger.warning("RAG: 検索クエリに一致するドキュメントが見つかりませんでした。")
                    context = "関連情報は見つかりませんでした。" # または空にする

                # 2. プロンプトを構築
                try:
                    prompt = params.prompt_template.format(context=context, question=user_query)
                    # promtに「回答はJSON形式で出力してください。」を追加
                    prompt = f"回答はJSON形式で出力してください。\n{prompt}"    
                    logger.info(f"RAG: プロンプト構築完了。 prompt:{prompt}")
                except KeyError as e:
                    raise ValueError(f"RAGプロンプトテンプレートが無効です。'{e}' プレースホルダが必要です。テンプレート: {params.prompt_template}")

                # 3. LLMを呼び出し
                # TODO: モデル名をconfig等から取得できるようにする
                rag_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(method="json_mode")
                
                response = await rag_llm.ainvoke(prompt) # 非同期呼び出し
                result_scalar = response #dict型
                logger.info(f"RAG: LLMからの応答を取得しました。 result:{result_scalar}")
                # 4.　結果を保存
                if output_df_name:
                    result_df = pd.DataFrame([result_scalar]) # 辞書をリストでラップして1行のDataFrameを作成

                print(f"rag_result: {result_scalar}") # Log scalar output

            else:
                raise NotImplementedError(f"未定義の操作タイプ: {step.operation}")

            # --- 出力結果の保存 ---
            if result_df is not None:
                current_dataframes[output_df_name] = result_df
                logger.info(f"ステップ完了。データフレーム '{output_df_name}' を更新/作成。形状: {result_df.shape}")
                print(f"{output_df_name}:{result_df}")
            elif result_scalar is not None:
                 current_intermediate_results[output_scalar_name] = result_scalar
                 logger.info(f"ステップ完了。中間スカラー '{output_scalar_name}' を保存。値: {result_scalar}")
                 print(f"{output_scalar_name}: {result_scalar}") # Log scalar output
            elif result_df is None and result_scalar is None:
                 logger.warning(f"ステップ {step.step_id} は結果を出力しませんでした。")


            step_log["status"] = "success"
            if result_df is not None: step_log["output_shape"] = result_df.shape
            if result_scalar is not None: step_log["output_scalar_value"] = result_scalar

            if (i+1) == len(plan.plan) and result_df is not None: # Ensure final output is DF if applicable
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