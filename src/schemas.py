"""意图分类定义与结构化输出类型。

NLU 可识别的全部意图及路由映射均在此集中定义。
"""

from dataclasses import dataclass, field
from enum import Enum


# ─── 意图枚举 ───────────────────────────────────────────────

class IntentCategory(str, Enum):
    """NLU 可识别的所有意图类别。

    继承 str 和 Enum，保证可以当字符串使用（兼容 Python 3.10）。
    """
    BEGIN_GESTURE_RECOGNITION = "begin_gesture_recognition"  # 开启手势识别
    CLOSE_GESTURE_RECOGNITION = "close_gesture_recognition"  # 关闭手势识别
    UNKNOWN = "unknown"                                      # 无法识别

    def __str__(self) -> str:
        return self.value


# ─── 路由映射表 ─────────────────────────────────────────────

INTENT_ROUTING: dict[IntentCategory, str] = {
    IntentCategory.BEGIN_GESTURE_RECOGNITION: "open_gesture_recognition",
    IntentCategory.CLOSE_GESTURE_RECOGNITION: "close_gesture_recognition",
    IntentCategory.UNKNOWN: "fallback_handler",
}


# ─── 意图结果数据结构 ───────────────────────────────────────

@dataclass
class IntentSlot:
    """意图槽位，承载解析出的参数。"""
    key: str
    value: str


@dataclass
class IntentResult:
    """NLU 解析产出的最终结构化结果。

    routing_target 默认为空，__post_init__ 会根据 intent 自动补全。
    """
    intent: IntentCategory
    confidence: float
    raw_text: str = ""
    slots: list[IntentSlot] = field(default_factory=list)
    routing_target: str = ""

    def __post_init__(self) -> None:
        """自动补全 routing_target（若未显式指定）。"""
        if not self.routing_target:
            self.routing_target = INTENT_ROUTING.get(
                self.intent, "fallback_handler"
            )


# ─── Function Calling Schema ─────────────────────────────────

def get_nlu_function_schema() -> dict:
    """返回 NLU 意图解析的 OpenAI function calling tool schema。

    通过 function calling 强制 LLM 返回结构化 JSON，
    不再依赖 prompt 中"只输出 JSON"的指令。
    """
    return {
        "type": "function",
        "function": {
            "name": "parse_intent",
            "description": "解析用户语音输入的意图",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [e.value for e in IntentCategory],
                        "description": (
                            "意图类别。"
                            "begin_gesture_recognition=开启手势识别, "
                            "close_gesture_recognition=关闭手势识别, "
                            "unknown=无法识别"
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "置信度。0.9+=明确, "
                            "0.7-0.89=语义清晰但用词不标准, "
                            "0.5-0.69=模糊, 0.0=unknown"
                        ),
                    },
                    "slots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                            },
                        },
                        "description": "参数槽位，当前版本为空数组 []",
                    },
                },
                "required": ["intent", "confidence", "slots"],
            },
        },
    }
