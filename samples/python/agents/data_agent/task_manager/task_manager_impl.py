import asyncio
import logging
import os  # 追加
from typing import Callable, Coroutine, Any, Union, AsyncIterable, Dict
import pandas as pd
import numpy as np

# --- ロギング設定 ---
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "data_agent.log")

# logsディレクトリを作成 (存在しない場合)
current_working_directory = os.getcwd() # 追加
logs_absolute_path = os.path.abspath(LOGS_DIR) # 追加
print(f"[DEBUG] Current Working Directory: {current_working_directory}") # 追加
print(f"[DEBUG] Attempting to create logs directory at: {logs_absolute_path}") # 追加
os.makedirs(LOGS_DIR, exist_ok=True)

# ルートロガーを取得
root_logger = logging.getLogger()
# ハンドラがまだ設定されていない場合のみ設定 (重複設定防止)
# if not root_logger.hasHandlers(): # 一時的にコメントアウト
root_logger.setLevel(logging.DEBUG) # 必要に応じてレベル調整

# 既存のハンドラを一旦クリアする (他の設定との競合を避けるため)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# フォーマッタを作成
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ファイルハンドラを作成
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG) # ファイルに出力するログレベル
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# コンソールハンドラを作成 (コンソール出力も維持)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # コンソールに出力するログレベル
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# --- テストログ出力 ---
root_logger.debug("--- Logging configured in task_manager_impl.py. Attempting to write to file. ---")
# --- ロギング設定ここまで ---

# 共通ライブラリからのインポート (プロジェクトルートからの絶対パスに修正)
from A2A_risk.samples.python.common.server.task_manager import InMemoryTaskManager
from A2A_risk.samples.python.common.types import (
    Task, TaskIdParams, TaskQueryParams, GetTaskRequest, TaskNotFoundError, SendTaskRequest, CancelTaskRequest, TaskNotCancelableError, SetTaskPushNotificationRequest, GetTaskPushNotificationRequest, GetTaskResponse, CancelTaskResponse, SendTaskResponse, SetTaskPushNotificationResponse, GetTaskPushNotificationResponse, PushNotificationNotSupportedError, TaskSendParams, TaskStatus, TaskState, TaskResubscriptionRequest, SendTaskStreamingRequest, SendTaskStreamingResponse, Artifact, PushNotificationConfig, TaskStatusUpdateEvent, JSONRPCError, TaskPushNotificationConfig, InternalError, Message, TextPart, DataPart, JSONRPCResponse
)
from A2A_risk.samples.python.common.server.utils import new_not_implemented_error

# agent.py から run_agent をインポート (相対インポートはそのまま)
from A2A_risk.samples.python.agents.data_agent.agent import run_agent 

logger = logging.getLogger(__name__)

class DataAgentTaskManager(InMemoryTaskManager):
    """
    DataAgent固有のタスク処理ロジックを実装したTaskManager。
    QueryAgentの実装を使ってタスクを非同期に実行する。
    """
    def __init__(self, agent_runner: Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]):
        super().__init__()
        self.agent_runner = agent_runner

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        タスクを受け付け、初期状態を保存し、バックグラウンド処理を開始する。
        即座にTaskオブジェクト(SUBMITTED状態)を返す。
        """
        logger.info(f"--- Start on_send_task for request ID: {request.id} ---")
        logger.debug(f"Received task send request details: {request}")
        task_send_params = request.params

        # タスクを初期状態で保存 (upsert_taskはInMemoryTaskManagerで実装済み)
        task = await self.upsert_task(task_send_params)
        logger.info(f"Task {task.id} created with state {task.status.state}")

        # バックグラウンドでタスク処理を開始
        asyncio.create_task(self._process_task(task.id, task_send_params.message))
        logger.info(f"Background task started for {task.id}")

        # 即座に現在のTaskオブジェクトを返す
        task_result = self.append_task_history(task, task_send_params.historyLength)
        response = SendTaskResponse(id=request.id, result=task_result)
        logger.info(f"--- End on_send_task for request ID: {request.id}. Returning response. ---")
        logger.debug(f"Returning response details: {response}")
        return response

    async def _process_task(self, task_id: str, input_message: Message):
        """
        バックグラウンドで実行されるタスク処理。
        QueryAgentを呼び出し、結果をTaskストアに反映する。
        """
        logger.info(f"Processing task {task_id}...")
        try:
            # 状態を WORKING に更新
            await self.update_task_status(task_id, TaskState.WORKING)

            # 入力メッセージからデータを取得
            input_data_dict = None
            if input_message.parts:
                # 最初の Part が DataPart であることを期待
                if isinstance(input_message.parts[0], DataPart):
                    input_data_dict = input_message.parts[0].data
                    logger.info(f"Task {task_id}: Extracted data from DataPart: {input_data_dict}")
                elif isinstance(input_message.parts[0], TextPart):
                    # TextPart の場合は、互換性のために辞書に変換
                    input_data_dict = {"input": input_message.parts[0].text}
                    logger.info(f"Task {task_id}: Converted TextPart to data dict: {input_data_dict}")
                else:
                    logger.warning(f"Task {task_id}: First message part is neither TextPart nor DataPart.")
            
            if input_data_dict is None:
                logger.error(f"Task {task_id}: Could not extract valid input data from message.")
                # エラーメッセージを作成
                error_message = Message(role="agent", parts=[TextPart(text="Invalid input message format.")])
                await self.update_task_status(task_id, TaskState.FAILED, error_message)
                return

            # agent_runner に抽出した辞書データを渡す
            logger.info(f"Running agent for task {task_id} with data: {input_data_dict}")
            agent_response = await self.agent_runner(input_data_dict)
            output_text = agent_response.get("output", "Agent did not produce output.")
            logger.info(f"Agent for task {task_id} finished. Output: '{output_text}'")

            # 結果をMessageオブジェクトとして作成
            parts = []
            if isinstance(output_text, dict):
                text_for_message = output_text.get("text", str(output_text))
                parts.append(TextPart(text=text_for_message))
                # if output_text.get("data") is not None:
                #     print(f"output_text['data'] type: {type(output_text['data'])}")
                #     try:
                #         if isinstance(output_text["data"], dict):
                #             data_dict = output_text["data"]
                #         elif isinstance(output_text["data"], list):
                #             data_dict = {"items": output_text["data"]} 
                #         elif isinstance(output_text["data"], str):
                #             data_dict = {"value": output_text["data"]}
                #         else:
                #             # DataFrame型の場合、カラムの型を表示
                #             if isinstance(output_text["data"], pd.DataFrame):
                #                 print("[DEBUG] DataFrame columns and dtypes:")
                #                 print(output_text["data"].dtypes)
                #             def to_serializable(obj):
                #                 # print(f"DEBUG: type(obj) = {type(obj)}, value = {obj}")
                #                 if isinstance(obj, pd.DataFrame):
                #                     df_copy = obj.copy()
                #                     for col in df_copy.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
                #                         df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d %H:%M:%S")
                #                     df_copy = df_copy.where(pd.notnull(df_copy), None)  
                #                     return df_copy.to_dict(orient="records")
                #                 elif isinstance(obj, pd.Series):
                #                     # Seriesのdtypeがdatetime64[ns]なら、各要素を文字列に変換
                #                     if pd.api.types.is_datetime64_any_dtype(obj):
                #                         return obj.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                #                     else:
                #                         return obj.to_dict()
                #                 elif isinstance(obj, (pd.Timestamp, np.datetime64)):
                #                     print(f"obj type: pd.Timestamp, np.datetime64")
                #                     return obj.strftime("%Y-%m-%d %H:%M:%S")
                #                 elif isinstance(obj, dict):
                #                     return {k: to_serializable(v) for k, v in obj.items()}
                #                 elif isinstance(obj, list):
                #                     return [to_serializable(v) for v in obj]
                #                 else:
                #                     return obj
                #             data_dict = {k: (to_serializable(v) if hasattr(v, "to_dict") else v)
                #                         for k, v in output_text["data"].items()}
                #         parts.append(DataPart(data=data_dict))
                #     except Exception as e:
                #         logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            else:
                parts.append(TextPart(text=output_text))
            result_message = Message(role="agent", parts=parts)

            # 状態を COMPLETED に更新し、結果メッセージを保存
            await self.update_task_status(task_id, TaskState.COMPLETED, result_message)
            logger.info(f"Task {task_id} completed successfully.")

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            # エラーメッセージを作成
            error_message = Message(role="agent", parts=[TextPart(text=f"Task failed: {e}")])
            # 状態を FAILED に更新
            await self.update_task_status(task_id, TaskState.FAILED, error_message)

    async def update_task_status(self, task_id: str, state: TaskState, message: Message | None = None):
        """タスクの状態とメッセージを更新する"""
        logger.info(f"Updating task {task_id} status to {state}")
        status = TaskStatus(state=state, message=message)
        # update_storeはInMemoryTaskManagerで実装済み
        task = await self.update_store(task_id=task_id, status=status, artifacts=[]) # Artifactsは今回なし

    # --- ストリーミング/Resubscribe用のメソッド (今回は実装しない) ---
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        logger.warning("Streaming task sending (on_send_task_subscribe) is not implemented in this prototype.")
        return new_not_implemented_error(request.id)

    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        logger.warning("Task resubscription (on_resubscribe_to_task) is not implemented in this prototype.")
        return new_not_implemented_error(request.id)

# --- 必要であれば依存関係を requirements.txt に追加 ---
# langchain-openai
# python-dotenv
# langchain
# langchain-core
# sse-starlette (ストリーミング実装時)
# uvicorn
# starlette
# pydantic

# 