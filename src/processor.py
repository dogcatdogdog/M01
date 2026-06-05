"""数据处理层：格式化 NLU 输入 / 解析并校验 LLM 输出。

负责：
1. 加载 system prompt 模板
2. 将语音文本 + prompt 组装为 LLM 入参
3. 解析 LLM 返回的 JSON 为 IntentResult
4. 对解析结果做 schema 校验，异常时安全降级
"""

import json
import logging
import re
import string
from typing import Optional

from src.schemas import IntentCategory, IntentResult, IntentSlot, INTENT_ROUTING

logger = logging.getLogger(__name__)


# ─── 注入检测 ───────────────────────────────────────────────

# 需要拦截的注入关键词（大小写不敏感）
_INJECTION_KEYWORDS = [
    "ignore previous",
    "override all rules",
    "system:",
    "system prompt",
    "new instruction",
    "disregard",
    "you are now",
    "your new task",
]

# 正常语音命令的最大合理长度
_MAX_VOICE_COMMAND_LENGTH = 100


def detect_injection(user_text: str) -> Optional[str]:
    """检测输入是否为注入攻击，返回拒绝原因或 None。

    在调用 LLM 之前执行，节省 API 成本并提高安全性。
    """
    text = user_text.strip()
    text_lower = text.lower()

    # 1. 包含 JSON 结构
    if re.search(r'\{\s*"intent"\s*:', text):
        return "输入包含 JSON 结构（疑似注入攻击）"

    # 2. 全英文（>80% ASCII 字母，且无中文字符）
    ascii_letters = sum(1 for c in text if c in string.ascii_letters)
    has_cjk = bool(re.search(r'[一-鿿]', text))
    if not has_cjk and len(text) > 5 and ascii_letters / max(len(text), 1) > 0.6:
        return "输入主要为英文（疑似注入攻击）"

    # 3. 包含系统注入关键词
    for kw in _INJECTION_KEYWORDS:
        if kw in text_lower:
            return f"输入包含注入关键词: '{kw}'"

    # 4. 超长输入（超出语音指令合理范围）
    if len(text) > _MAX_VOICE_COMMAND_LENGTH:
        return f"输入过长 ({len(text)} 字符，超出语音指令合理范围)"

    return None


# ─── 公开函数 ───────────────────────────────────────────────

def load_prompt_template(path: str = "prompts/nlu_intent_parsing.md") -> str:
    """从文件加载 NLU 意图解析的 system prompt 模板。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_input(user_text: str, system_prompt: str) -> dict[str, str]:
    """将原始语音文本和 system prompt 组装为 LLM 调用参数。

    Args:
        user_text: ASR 转录后的用户语音文本。
        system_prompt: 已加载的 NLU 解析 prompt 模板。

    Returns:
        {"prompt": ..., "system_prompt": ...} 字典。
    """
    cleaned_text = user_text.strip()
    if not cleaned_text:
        logger.warning("输入文本为空，将交由 NLU 判定为 unknown。")

    return {"prompt": cleaned_text, "system_prompt": system_prompt}


def parse_output(raw_response: str, user_text: str = "") -> IntentResult:
    """解析 LLM 返回的文本为 IntentResult。

    内置容错机制：
    - JSON 被 markdown 代码块包裹 → 自动剥离
    - JSON 前后有解释文字 → 正则提取
    - 非法 JSON → 降级为 unknown
    - intent 不在已知枚举中 → 降级为 unknown

    Args:
        raw_response: LLM 返回的原始文本。
        user_text: 原始用户输入，用于结果追踪。

    Returns:
        经过校验的 IntentResult 实例。
    """
    json_str = _extract_json(raw_response)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error("LLM 返回非 JSON 内容: %.200s", raw_response)
        return _unknown_result(user_text)

    # 提取并校验 intent
    intent_str = data.get("intent", "")
    confidence = _clamp_confidence(data.get("confidence"))
    slots = _parse_slots(data.get("slots", []))

    try:
        intent = IntentCategory(intent_str)
    except ValueError:
        logger.warning("LLM 返回未知意图: '%s'，降级为 unknown", intent_str)
        intent = IntentCategory.UNKNOWN
        confidence = min(confidence, 0.3)

    # unknown 意图强制零置信度
    if intent == IntentCategory.UNKNOWN:
        confidence = 0.0

    return IntentResult(
        intent=intent,
        confidence=confidence,
        slots=slots,
        raw_text=user_text,
        routing_target=INTENT_ROUTING.get(intent, "fallback_handler"),
    )


# ─── 内部辅助函数 ───────────────────────────────────────────

def _extract_json(text: str) -> str:
    """从 LLM 响应中提取 JSON 子串。

    处理三种情况：
    1. ```json ... ``` 代码块
    2. 裸 { ... } 结构
    3. 其他（原样返回，后续解析失败时走容错逻辑）
    """
    # 匹配 markdown 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # 匹配第一个 { ... } 结构
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)

    return text


def _clamp_confidence(value: object) -> float:
    """解析置信度值，钳制到 [0.0, 1.0] 范围。"""
    try:
        conf = float(value)  # type: ignore[arg-type]
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return 0.5


def _parse_slots(raw_slots: object) -> list[IntentSlot]:
    """解析槽位列表（当前版本不使用槽位，但保留解析能力）。"""
    if not isinstance(raw_slots, list):
        return []
    result: list[IntentSlot] = []
    for item in raw_slots:
        if isinstance(item, dict):
            result.append(IntentSlot(
                key=str(item.get("key", "")),
                value=str(item.get("value", "")),
            ))
    return result


def _unknown_result(user_text: str) -> IntentResult:
    """生成 unknown 意图结果（解析失败时的安全降级）。"""
    return IntentResult(
        intent=IntentCategory.UNKNOWN,
        confidence=0.0,
        raw_text=user_text,
    )
