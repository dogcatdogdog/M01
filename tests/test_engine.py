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
    # 兜底路径的 call() 也用不到，但保留以防降级测试
    mock_client.call.return_value = (
        f'{{"intent": "{intent}", "confidence": {confidence}, "slots": []}}'
    )
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

    # ── 兜底路径：FC 失败 → 降级到文本解析 ──

    def test_fc_fails_fallback_to_text(self):
        """FC 抛异常时自动降级到文本路径。"""
        mock_client = MagicMock()
        # FC 会失败
        mock_client.call_with_function.side_effect = RuntimeError("FC failed")
        # 文本路径返回正常
        mock_client.call.return_value = (
            '{"intent": "begin_gesture_recognition", "confidence": 0.95, "slots": []}'
        )
        engine = NLUEngine(client=mock_client, simulate=False)
        result = engine.parse("开启手势识别")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION

    def test_fc_then_text_also_garbage(self):
        """FC 失败 + 文本也是非法 JSON → unknown。"""
        mock_client = MagicMock()
        mock_client.call_with_function.side_effect = RuntimeError("FC failed")
        mock_client.call.return_value = "garbage"
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
