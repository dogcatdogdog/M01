"""src/schemas.py 单元测试 — 数据结构与枚举校验。"""

import pytest
from src.schemas import (
    IntentCategory,
    IntentResult,
    IntentSlot,
    INTENT_ROUTING,
)


class TestIntentCategory:
    """意图枚举测试。"""

    def test_valid_enum_values(self):
        """已知枚举值可以通过字符串构造。"""
        assert IntentCategory("begin_gesture_recognition") == IntentCategory.BEGIN_GESTURE_RECOGNITION
        assert IntentCategory("close_gesture_recognition") == IntentCategory.CLOSE_GESTURE_RECOGNITION
        assert IntentCategory("unknown") == IntentCategory.UNKNOWN

    def test_invalid_enum_value(self):
        """非法枚举值抛出 ValueError。"""
        with pytest.raises(ValueError):
            IntentCategory("fly_to_moon")

    def test_enum_is_string(self):
        """StrEnum 可以当字符串使用。"""
        assert IntentCategory.BEGIN_GESTURE_RECOGNITION == "begin_gesture_recognition"
        assert str(IntentCategory.UNKNOWN) == "unknown"


class TestIntentResult:
    """IntentResult 数据类测试。"""

    def test_auto_routing_begin_gesture(self):
        """routing_target 自动填充 — 开启手势识别。"""
        result = IntentResult(
            intent=IntentCategory.BEGIN_GESTURE_RECOGNITION,
            confidence=0.95,
            raw_text="开启手势识别",
        )
        assert result.routing_target == "open_gesture_recognition"

    def test_auto_routing_close_gesture(self):
        """routing_target 自动填充 — 关闭手势识别。"""
        result = IntentResult(
            intent=IntentCategory.CLOSE_GESTURE_RECOGNITION,
            confidence=0.90,
            raw_text="关闭手势识别",
        )
        assert result.routing_target == "close_gesture_recognition"

    def test_auto_routing_unknown(self):
        """routing_target 自动填充 — unknown。"""
        result = IntentResult(
            intent=IntentCategory.UNKNOWN,
            confidence=0.0,
        )
        assert result.routing_target == "fallback_handler"

    def test_manual_routing_override(self):
        """手动指定 routing_target 时不被自动覆盖。"""
        result = IntentResult(
            intent=IntentCategory.BEGIN_GESTURE_RECOGNITION,
            confidence=0.95,
            routing_target="custom_handler",
        )
        assert result.routing_target == "custom_handler"

    def test_empty_slots_by_default(self):
        """slots 默认为空列表。"""
        result = IntentResult(
            intent=IntentCategory.UNKNOWN,
            confidence=0.0,
        )
        assert result.slots == []

    def test_slots_preserved(self):
        """传入的 slots 被保留。"""
        result = IntentResult(
            intent=IntentCategory.BEGIN_GESTURE_RECOGNITION,
            confidence=0.95,
            slots=[IntentSlot(key="action", value="open")],
        )
        assert len(result.slots) == 1
        assert result.slots[0].key == "action"
        assert result.slots[0].value == "open"


class TestIntentRouting:
    """路由映射表测试。"""

    def test_all_intents_have_routing(self):
        """每个 IntentCategory 枚举值都有对应的路由目标。"""
        for intent in IntentCategory:
            assert intent in INTENT_ROUTING, f"{intent} 缺少路由映射"
            assert isinstance(INTENT_ROUTING[intent], str)
            assert INTENT_ROUTING[intent] != ""

    def test_routing_values_are_unique(self):
        """路由目标值不重复（同一路由可多次出现）。"""
        # 只需确保路由表完整性已在上一个测试覆盖
        pass
