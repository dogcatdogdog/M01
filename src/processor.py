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
from typing import Optional

from src.schemas import IntentCategory, IntentResult, IntentSlot, INTENT_ROUTING

logger = logging.getLogger(__name__)


# ─── 注入检测 ───────────────────────────────────────────────
#
# 参考:
#   - Google Security Blog: "Mitigating prompt injection attacks" (分层防御)
#   - OWASP LLM01:2025 Prompt Injection
#   - protectai/llm-guard: 基于 ML 的注入扫描器
#
# 设计原则（语言无关，不做关键词匹配）:
#   1. 正向约束：定义什么是"合法语音指令"，不符合即可疑
#      合法语音指令 = 简短 + 自然语言 + 无语义攻击结构
#   2. 结构异常检测：代码块/JSON/系统指令语法在任何语言中都不该出现
#      在语音输入中
#   3. 字符级攻击检测：零宽字符、控制字符、Unicode 混淆
#   4. 不做语言判断（不禁止英文、法文、阿拉伯文等）

# 合法的语音指令最大长度（超出即非正常语音）
_MAX_VOICE_COMMAND_LENGTH = 100

# 这些结构在任何语言中都不应该出现在正常的语音输入里
# （不是"坏词"，而是"语音中不可能出现的结构模式"）
_STRUCTURAL_SIGNATURES = [
    # JSON / 数据序列化结构
    (r'\{\s*"(?:intent|confidence|slots|command|action)"', "JSON 结构"),
    (r'<json>|</json>|<\?xml', "XML/数据序列化结构"),
    # Markdown / 代码块
    (r'```', "代码块标记"),
    (r'<system[^>]*>|</system>|<<SYS>>', "系统指令标记"),
    # RestructuredText / 其他标记语言
    (r'^={3,}\s*$|^-{3,}\s*$|^\*{3,}\s*$', "分隔线标记"),
    # MIME/HTTP 风格头
    (r'^[A-Za-z-]+:\s', "协议头风格语法"),
]

# Unicode 控制字符和混淆字符（零宽、方向覆盖等）
_DANGEROUS_UNICODE_PATTERNS = [
    (r'[​‌‍‎‏]', "零宽字符"),          # zero-width chars
    (r'[‪-‮]', "Unicode 方向控制字符"),               # bidi override
    (r'[￰-￿]', "Unicode 特殊区域字符"),               # specials block
]


def detect_injection(user_text: str) -> Optional[str]:
    """语言无关的注入检测。检测输入是否偏离'合法语音指令'的特征。

    合法语音指令的特征:
      - 简短 (<=100 字符)
      - 不包含代码/标记/数据序列化结构
      - 不包含 Unicode 攻击字符
    """
    text = user_text.strip()

    # 空输入是合法的（LLM 判 unknown）
    if not text:
        return None

    # ── 第一层：长度约束 ──────────────────────────────────
    if len(text) > _MAX_VOICE_COMMAND_LENGTH:
        return f"输入过长 ({len(text)} 字符，语音指令上限 {_MAX_VOICE_COMMAND_LENGTH})"

    # ── 第二层：结构异常检测（语言无关）────────────────────
    for pattern, label in _STRUCTURAL_SIGNATURES:
        if re.search(pattern, text):
            return f"输入包含非语音结构: {label}"

    # ── 第三层：Unicode 攻击字符检测 ──────────────────────
    for pattern, label in _DANGEROUS_UNICODE_PATTERNS:
        if re.search(pattern, text):
            return f"输入包含攻击性 Unicode 字符: {label}"

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
