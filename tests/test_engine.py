"""src/engine.py 集成测试 — 用 Mock LLM 验证流水线。"""

import pytest
from unittest.mock import MagicMock
from src.engine import NLUEngine
from src.schemas import IntentCategory


def _mock_fc_result(intent: str, confidence: float, slots=None):
    """创建模拟的 function calling 返回值。"""
    mock_client = MagicMock()
    mock_client.call_with_function.return_value = {
        "name": "parse_intent",
        "arguments": {
            "intent": intent,
            "confidence": confidence,
            "slots": slots or [],
        },
    }
    return mock_client


class TestNLUEngineWithMock:
    """使用 Mock LLM 的引擎测试（模拟 function calling）。"""

    # ── Function Calling 路径 ──

    def test_begin_gesture_recognition(self):
        mock = _mock_fc_result("begin_gesture_recognition", 0.95)
        engine = NLUEngine(client=mock, simulate=False)
        result = engine.parse("开启手势识别")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION
        assert result.confidence == 0.95
        assert result.routing_target == "open_gesture_recognition"

    def test_close_gesture_recognition(self):
        mock = _mock_fc_result("close_gesture_recognition", 0.90)
        engine = NLUEngine(client=mock, simulate=False)
        result = engine.parse("关闭手势识别")
        assert result.intent == IntentCategory.CLOSE_GESTURE_RECOGNITION
        assert result.confidence == 0.90

    def test_unknown(self):
        mock = _mock_fc_result("unknown", 0.0)
        engine = NLUEngine(client=mock, simulate=False)
        result = engine.parse("今天天气怎么样")
        assert result.intent == IntentCategory.UNKNOWN
        assert result.confidence == 0.0

    # ── FC 失败处理 ──

    def test_fc_throws_exception_returns_unknown(self):
        """FC 抛异常时直接返回 unknown，不崩溃。"""
        mock_client = MagicMock()
        mock_client.call_with_function.side_effect = RuntimeError("FC 网络错误")
        engine = NLUEngine(client=mock_client, simulate=False)
        result = engine.parse("开启手势识别")
        assert result.intent == IntentCategory.UNKNOWN
        assert result.confidence == 0.0

    def test_fc_returns_empty_arguments(self):
        """FC 未返回 tool_call 时返回 unknown。"""
        mock_client = MagicMock()
        mock_client.call_with_function.return_value = {"name": "parse_intent", "arguments": {}}
        engine = NLUEngine(client=mock_client, simulate=False)
        result = engine.parse("测试")
        assert result.intent == IntentCategory.UNKNOWN
        assert result.confidence == 0.0

    # ── 参数测试 ──

    def test_simulate_disabled(self):
        mock = _mock_fc_result("unknown", 0.0)
        engine = NLUEngine(client=mock, simulate=False)
        assert engine.simulate is False

    def test_simulate_enabled_by_default(self):
        mock = _mock_fc_result("unknown", 0.0)
        engine = NLUEngine(client=mock)
        assert engine.simulate is True

    def test_prompt_loaded_on_first_parse(self):
        """首次 parse 时懒加载 system prompt。"""
        mock = _mock_fc_result("unknown", 0.0)
        engine = NLUEngine(client=mock, simulate=False)
        assert engine._system_prompt is None
        engine.parse("测试")
        assert engine._system_prompt is not None
