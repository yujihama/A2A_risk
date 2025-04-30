import logging
from typing import Any, Dict

from ..core.node_base import Node, NodeResult

logger = logging.getLogger(__name__)


class ErrorNode(Node):
    id = "error"

    async def run(self, state: Dict[str, Any], toolbox):  # noqa: ANN001
        logger.error("[ErrorNode] invoked")

        message = state.get("error_message", "Unknown error")
        patch = {"final_result": {"error": message}}
        events = [{"type": "node", "name": "error", "msg": message}]
        return NodeResult(observation=message, patch=patch, events=events) 