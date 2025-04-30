from dataclasses import dataclass
from typing import Union

from .condition import Condition


@dataclass
class Edge:
    src: str
    dst: str
    condition: Condition
    priority: int = 0

    async def is_true(self, state, result, toolbox):  # noqa: ANN001
        return await self.condition.evaluate(state, result, toolbox) 