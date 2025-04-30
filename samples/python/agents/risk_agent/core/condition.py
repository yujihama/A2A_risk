from __future__ import annotations

import abc
from typing import Any, Dict

from pydantic import BaseModel, Field


class Condition(abc.ABC):
    """遷移判定の抽象クラス"""

    priority: int = 0  # 小さいほど高優先度

    @abc.abstractmethod
    async def evaluate(self, state: "DynamicAgentState", result: "NodeResult", toolbox: "ToolBox") -> bool:  # noqa: F821
        """true なら遷移成立"""
        raise NotImplementedError


class RuleCondition(Condition):
    def __init__(self, expr: str, *, priority: int = 0):
        self._code = compile(expr, "<rule-cond>", "eval")
        self.priority = priority

    async def evaluate(self, state, result, toolbox):  # noqa: ANN001
        try:
            return bool(eval(self._code, {}, {"state": state, "result": result}))
        except Exception:
            return False


class LLMCondition(Condition, BaseModel):  # type: ignore[misc]
    prompt_template: str = Field(..., alias="prompt", description="プロンプトテンプレート")
    priority: int = 10

    async def evaluate(self, state, result, toolbox):  # noqa: ANN001
        if not toolbox.llm:
            return False
        state_payload = state.dict() if hasattr(state, "dict") else state
        res_payload = result.dict() if hasattr(result, "dict") else result
        template = self.prompt_template
        if "{state" in template or "{result" in template:
            prompt = template.format(state=state_payload, result=res_payload)
        else:
            prompt = template
        try:
            resp = await toolbox.llm.ainvoke(prompt)
            # try direct field
            if isinstance(resp, dict) and "should_transition" in resp:
                return bool(resp["should_transition"])

            # fallbacks: parse json from text/content
            import json
            possible_fields = []
            if isinstance(resp, dict):
                possible_fields.extend([resp.get("text"), resp.get("content"), resp.get("message"), str(resp)])
            else:
                possible_fields.append(str(resp))

            for field in possible_fields:
                if not field:
                    continue
                try:
                    parsed = json.loads(field)
                    if isinstance(parsed, dict) and "should_transition" in parsed:
                        return bool(parsed["should_transition"])
                except Exception:  # noqa: BLE001
                    continue

            return False
        except Exception:
            return False 