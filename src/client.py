"""LLM 适配器：封装 OpenAI SDK 调用 DeepSeek API。

提供重试、超时、Token 计数等横切关注点。
所有 LLM 调用必须经过此模块。
"""

import json
import time
import logging
from typing import Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from config.settings import nlu_config

logger = logging.getLogger(__name__)


class LLMClient:
    """封装对 DeepSeek（OpenAI 兼容）API 的调用。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or nlu_config.api_key
        self.base_url = base_url or nlu_config.base_url
        self.model = model or nlu_config.model
        self.max_tokens = max_tokens or nlu_config.max_tokens
        self.temperature = temperature or nlu_config.temperature
        self.timeout = timeout or nlu_config.timeout
        self.max_retries = max_retries or nlu_config.max_retries

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _call_api(self, messages: list[dict], tools: Optional[list[dict]] = None,
                  tool_choice: Optional[dict] = None) -> dict:
        """底层 API 调用，返回原始 response 对象。

        Args:
            messages: 消息列表。
            tools: function calling tool 定义（可选）。
            tool_choice: 强制调用指定 function（可选）。

        Returns:
            API 返回的 choice 字典: {content, tool_calls, usage}。
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "LLM 调用 (尝试 %d/%d), model=%s, fc=%s",
                    attempt, self.max_retries, self.model,
                    "yes" if tools else "no",
                )
                t0 = time.perf_counter()

                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                }
                if tools:
                    kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

                response = self._client.chat.completions.create(**kwargs)

                elapsed = time.perf_counter() - t0
                msg = response.choices[0].message

                usage = response.usage
                if usage:
                    logger.info(
                        "LLM 调用成功 (%.2fs), prompt=%d, completion=%d, total=%d",
                        elapsed,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                    )

                return {
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls or [],
                    "usage": usage,
                }

            except RateLimitError as e:
                logger.warning("速率限制 (尝试 %d/%d): %s", attempt, self.max_retries, e)
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

            except APITimeoutError as e:
                logger.warning("请求超时 (尝试 %d/%d): %s", attempt, self.max_retries, e)
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1)

            except APIError as e:
                logger.error("API 错误 (尝试 %d/%d): %s", attempt, self.max_retries, e)
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2)

        raise RuntimeError(
            f"LLM 调用失败，已重试 {self.max_retries} 次，"
            f"最后一次错误: {last_error}"
        )

    def call(self, prompt: str, system_prompt: str) -> str:
        """纯文本调用，返回模型生成的文本。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call_api(messages)["content"]

    def call_with_function(
        self,
        prompt: str,
        system_prompt: str,
        tools: list[dict],
        tool_choice: dict,
    ) -> dict:
        """Function calling 调用，返回 tool_calls 结果。

        Args:
            prompt: 用户消息。
            system_prompt: 系统提示词。
            tools: function tool 定义列表。
            tool_choice: 强制调用指定 function。

        Returns:
            {"name": 函数名, "arguments": 参数字典}。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result = self._call_api(messages, tools=tools, tool_choice=tool_choice)

        if result["tool_calls"]:
            tc = result["tool_calls"][0]
            return {
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
        else:
            # 兜底：LLM 没返回 tool_call，尝试从 content 解析
            return {"name": "parse_intent", "arguments": {}}


# ─── 模块级便捷实例 ─────────────────────────────────────────

_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """获取默认 LLMClient 单例（懒初始化）。"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def call_llm(prompt: str, system_prompt: str) -> str:
    """便捷函数：使用默认客户端调用 LLM。"""
    return get_client().call(prompt, system_prompt)
