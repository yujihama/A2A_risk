import os
import asyncio
import json
from typing import Any, Dict

from aiolimiter import AsyncLimiter
from langchain_openai import ChatOpenAI

RATE_PER_MIN = int(os.getenv("OPENAI_RATE_LIMIT", "60"))  # requests per minute
_concurrency = int(os.getenv("OPENAI_CONCURRENCY", "5"))
_limiter = AsyncLimiter(max_rate=RATE_PER_MIN, time_period=60)
_sem = asyncio.Semaphore(_concurrency)


class OpenAILLMWrapper:
    """Async wrapper that respects rate limits using aiolimiter."""

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.0):
        self._chat = ChatOpenAI(model=model, temperature=temperature).with_structured_output(method="json_mode")

    async def ainvoke(self, prompt: str) -> Dict[str, Any]:  # noqa: D401
        async with _sem:
            async with _limiter:
                resp = await self._chat.ainvoke(prompt)
                # If response is plain string, try parse JSON
                if isinstance(resp, str):
                    try:
                        parsed = json.loads(resp)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:  # noqa: BLE001
                        pass
                    return {"text": resp}
                if isinstance(resp, dict):
                    return resp
                try:
                    return resp.__dict__
                except Exception:  # noqa: BLE001
                    return {"text": str(resp)} 