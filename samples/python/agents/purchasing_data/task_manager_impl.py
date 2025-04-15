import asyncio
import logging
from typing import Callable, Coroutine, Any, Union, AsyncIterable, Dict

# 共通ライブラリからのインポート (プロジェクトルートからの絶対パスに修正)
from samples.python.common.server.task_manager import InMemoryTaskManager
from samples.python.common.types import (
    SendTaskRequest, SendTaskResponse, TaskState, TaskStatus, TextPart, Message,
    SendTaskStreamingRequest, SendTaskStreamingResponse, TaskStatusUpdateEvent, JSONRPCResponse,
    TaskResubscriptionRequest, DataPart
)
from samples.python.common.server.utils import new_not_implemented_error

# agent.py から run_agent をインポート (相対インポートはそのまま)
from .agent import run_agent 

logger = logging.getLogger(__name__)

class PurchasingDataTaskManager(InMemoryTaskManager):
    """
    PurchasingDataAgent固有のタスク処理ロジックを実装したTaskManager。
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
        logger.info(f"Received task send request: {request.params.id}")
        task_send_params = request.params

        # タスクを初期状態で保存 (upsert_taskはInMemoryTaskManagerで実装済み)
        task = await self.upsert_task(task_send_params)
        logger.info(f"Task {task.id} created with state {task.status.state}")

        # バックグラウンドでタスク処理を開始
        asyncio.create_task(self._process_task(task.id, task_send_params.message))
        logger.info(f"Background task started for {task.id}")

        # 即座に現在のTaskオブジェクトを返す
        task_result = self.append_task_history(task, task_send_params.historyLength)
        return SendTaskResponse(id=request.id, result=task_result)

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
            result_message = Message(role="agent", parts=[TextPart(text=output_text)])

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