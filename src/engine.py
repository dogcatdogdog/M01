"""业务编排引擎：驱动 NLU 意图解析的完整流水线。

流水线（优先级从高到低）：
  1. Function Calling → 强制 LLM 返回结构化 JSON
  2. 文本解析 → FC 不可用时兜底解析 LLM 文本输出
"""

import json
import logging
from typing import Optional

from src.client import LLMClient, get_client
from src.processor import load_prompt_template, format_input, parse_output, detect_injection
from src.schemas import IntentCategory, IntentResult, IntentSlot, INTENT_ROUTING, get_nlu_function_schema

logger = logging.getLogger(__name__)


# ─── 模拟 API 调用 ──────────────────────────────────────────

def _simulate_api_call(routing_target: str, raw_text: str) -> None:
    """模拟调用下游功能模块（打印路由信息）。"""
    logger.info("模拟调用: %s()", routing_target)
    if routing_target == "open_gesture_recognition":
        print(f"  [模拟] 已调用 {routing_target}() — 手势识别功能已开启")
    elif routing_target == "close_gesture_recognition":
        print(f"  [模拟] 已调用 {routing_target}() — 手势识别功能已关闭")
    else:
        print(f"  [模拟] 已调用 {routing_target}() — 无匹配功能（fallback）")


# ─── Function Calling 结果解析 ─────────────────────────────

def _parse_function_result(fc_args: dict, user_text: str) -> IntentResult:
    """将 function calling 返回的参数转为 IntentResult。

    Args:
        fc_args: LLM 返回的 function arguments 字典。
        user_text: 原始用户输入。

    Returns:
        校验后的 IntentResult。
    """
    intent_str = fc_args.get("intent", "unknown")
    try:
        intent = IntentCategory(intent_str)
    except ValueError:
        logger.warning("FC 返回未知意图: '%s'，降级为 unknown", intent_str)
        intent = IntentCategory.UNKNOWN

    confidence = float(fc_args.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    if intent == IntentCategory.UNKNOWN:
        confidence = 0.0

    slots: list[IntentSlot] = []
    raw_slots = fc_args.get("slots", [])
    if isinstance(raw_slots, list):
        for s in raw_slots:
            if isinstance(s, dict):
                slots.append(IntentSlot(
                    key=str(s.get("key", "")),
                    value=str(s.get("value", "")),
                ))

    return IntentResult(
        intent=intent,
        confidence=confidence,
        slots=slots,
        raw_text=user_text,
    )


# ─── 引擎类 ─────────────────────────────────────────────────

class NLUEngine:
    """NLU 意图解析引擎。

    策略：优先使用 function calling 获取结构化输出；
    失败时回退到纯文本 + JSON 解析。

    封装完整的解析流水线：
    加载 prompt → function calling（或文本兜底）→ 解析 → 模拟 API 调用。
    """

    def __init__(
        self,
        prompt_path: str = "prompts/nlu_intent_parsing.md",
        client: Optional[LLMClient] = None,
        simulate: bool = True,
    ) -> None:
        self.prompt_path = prompt_path
        self._client = client
        self.simulate = simulate
        self._system_prompt: Optional[str] = None

        # function calling schema
        self._tool_schema = get_nlu_function_schema()
        self._tool_choice = {
            "type": "function",
            "function": {"name": "parse_intent"},
        }

    @property
    def system_prompt(self) -> str:
        """懒加载 system prompt 模板。"""
        if self._system_prompt is None:
            self._system_prompt = load_prompt_template(self.prompt_path)
            logger.info(
                "已加载 NLU prompt (%s), 长度=%d",
                self.prompt_path, len(self._system_prompt),
            )
        return self._system_prompt

    @property
    def client(self) -> LLMClient:
        """获取 LLM 客户端。"""
        if self._client is None:
            self._client = get_client()
        return self._client

    def parse(self, user_text: str) -> IntentResult:
        """执行 NLU 意图解析。"""
        logger.info("NLU 解析开始: '%s'", user_text[:100])

        # ── 前置注入检测 ──
        injection_reason = detect_injection(user_text)
        if injection_reason:
            logger.warning("注入攻击已拦截: %s", injection_reason)
            result = IntentResult(
                intent=IntentCategory.UNKNOWN,
                confidence=0.0,
                raw_text=user_text,
            )
            if self.simulate:
                _simulate_api_call(result.routing_target, user_text)
            return result

        # ── 主路径：Function Calling ──
        try:
            fc_result = self.client.call_with_function(
                prompt=user_text,
                system_prompt=self.system_prompt,
                tools=[self._tool_schema],
                tool_choice=self._tool_choice,
            )
            if fc_result["arguments"]:
                result = _parse_function_result(fc_result["arguments"], user_text)
                logger.info(
                    "NLU(FC) 结果: intent=%s, confidence=%.2f, routing=%s",
                    result.intent.value, result.confidence, result.routing_target,
                )
                if self.simulate:
                    _simulate_api_call(result.routing_target, user_text)
                return result
        except Exception as e:
            logger.warning("Function calling 失败，降级到文本解析: %s", e)

        # ── 兜底路径：纯文本 + JSON 解析 ──
        fmt = format_input(user_text, self.system_prompt)
        raw = self.client.call(
            prompt=fmt["prompt"],
            system_prompt=fmt["system_prompt"],
        )
        result = parse_output(raw, user_text)
        logger.info(
            "NLU(文本) 结果: intent=%s, confidence=%.2f, routing=%s",
            result.intent.value, result.confidence, result.routing_target,
        )
        if self.simulate:
            _simulate_api_call(result.routing_target, user_text)
        return result


# ─── 便捷函数 ────────────────────────────────────────────────

_default_engine: Optional[NLUEngine] = None


def get_engine() -> NLUEngine:
    """获取默认 NLUEngine 单例。"""
    global _default_engine
    if _default_engine is None:
        _default_engine = NLUEngine()
    return _default_engine


def parse_intent(user_text: str) -> IntentResult:
    """便捷函数：对用户输入执行 NLU 意图解析。"""
    return get_engine().parse(user_text)
