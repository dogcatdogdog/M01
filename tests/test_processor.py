"""src/processor.py 单元测试 — 输入格式化与输出解析。"""

import pytest
from src.processor import (
    load_prompt_template,
    format_input,
    parse_output,
    _extract_json,
    _clamp_confidence,
)
from src.schemas import IntentCategory


# ─── _extract_json 测试 ─────────────────────────────────────

class TestExtractJson:
    """JSON 提取函数测试。"""

    def test_clean_json(self):
        """纯 JSON 原样返回。"""
        text = '{"intent": "begin_gesture_recognition", "confidence": 0.95}'
        assert _extract_json(text) == text

    def test_json_in_markdown_fence(self):
        """```json ... ``` 代码块被正确剥离。"""
        text = '```json\n{"intent": "unknown", "confidence": 0.0}\n```'
        result = _extract_json(text)
        assert result == '{"intent": "unknown", "confidence": 0.0}'

    def test_json_in_markdown_fence_no_lang(self):
        """``` ... ``` 无语言标注的代码块。"""
        text = '```\n{"intent": "close_gesture_recognition", "confidence": 0.9}\n```'
        result = _extract_json(text)
        assert result == '{"intent": "close_gesture_recognition", "confidence": 0.9}'

    def test_json_with_prefix_text(self):
        """JSON 前有解释文字时提取 JSON 部分。"""
        text = '解析结果如下：{"intent": "begin_gesture_recognition", "confidence": 0.95}'
        result = _extract_json(text)
        assert result == '{"intent": "begin_gesture_recognition", "confidence": 0.95}'

    def test_json_with_suffix_text(self):
        """JSON 后有解释文字时提取 JSON 部分。"""
        text = '{"intent": "unknown", "confidence": 0.0}（这是解析结果）'
        result = _extract_json(text)
        assert result == '{"intent": "unknown", "confidence": 0.0}'

    def test_no_json_found(self):
        """非 JSON 输入原样返回。"""
        assert _extract_json("plain text") == "plain text"


# ─── _clamp_confidence 测试 ─────────────────────────────────

class TestClampConfidence:
    """置信度校验函数测试。"""

    def test_normal_value(self):
        assert _clamp_confidence(0.85) == 0.85

    def test_too_high(self):
        """超出 1.0 钳制到 1.0。"""
        assert _clamp_confidence(1.5) == 1.0

    def test_negative(self):
        """负数钳制到 0.0。"""
        assert _clamp_confidence(-0.5) == 0.0

    def test_zero(self):
        assert _clamp_confidence(0.0) == 0.0

    def test_one(self):
        assert _clamp_confidence(1.0) == 1.0

    def test_string_number(self):
        """字符串数字可正确解析。"""
        assert _clamp_confidence("0.8") == 0.8

    def test_non_numeric(self):
        """非数字默认返回 0.5。"""
        assert _clamp_confidence("high") == 0.5

    def test_none(self):
        """None 默认返回 0.5。"""
        assert _clamp_confidence(None) == 0.5


# ─── format_input 测试 ──────────────────────────────────────

class TestFormatInput:
    """输入格式化函数测试。"""

    @pytest.fixture
    def system_prompt(self):
        return "test system prompt"

    def test_normal_input(self, system_prompt):
        result = format_input("开启手势识别", system_prompt)
        assert result["prompt"] == "开启手势识别"
        assert result["system_prompt"] == system_prompt

    def test_whitespace_trimmed(self, system_prompt):
        """首尾空白被去除。"""
        result = format_input("  开启手势识别  ", system_prompt)
        assert result["prompt"] == "开启手势识别"

    def test_empty_string(self, system_prompt):
        """空字符串不报错，prompt 为空字符串。"""
        result = format_input("", system_prompt)
        assert result["prompt"] == ""

    def test_whitespace_only(self, system_prompt):
        """全空白返回空字符串。"""
        result = format_input("   ", system_prompt)
        assert result["prompt"] == ""


# ─── parse_output 测试 ──────────────────────────────────────

class TestParseOutput:
    """输出解析函数测试。"""

    # ── 正常解析 ──

    def test_begin_gesture(self):
        raw = '{"intent": "begin_gesture_recognition", "confidence": 0.95, "slots": []}'
        result = parse_output(raw, "开启手势识别")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION
        assert result.confidence == 0.95
        assert result.routing_target == "open_gesture_recognition"
        assert result.raw_text == "开启手势识别"

    def test_close_gesture(self):
        raw = '{"intent": "close_gesture_recognition", "confidence": 0.90, "slots": []}'
        result = parse_output(raw, "关闭手势识别")
        assert result.intent == IntentCategory.CLOSE_GESTURE_RECOGNITION
        assert result.confidence == 0.90

    def test_unknown(self):
        raw = '{"intent": "unknown", "confidence": 0.0, "slots": []}'
        result = parse_output(raw, "今天天气怎么样")
        assert result.intent == IntentCategory.UNKNOWN
        assert result.confidence == 0.0

    def test_with_slots(self):
        """带槽位的解析。"""
        raw = '{"intent": "begin_gesture_recognition", "confidence": 0.95, "slots": [{"key": "action", "value": "open"}]}'
        result = parse_output(raw, "开启手势识别")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION
        assert len(result.slots) == 1
        assert result.slots[0].key == "action"

    # ── 容错、容错、容错 ──

    def test_invalid_json(self):
        """非法 JSON 降级为 unknown。"""
        result = parse_output("not json at all", "原始输入")
        assert result.intent == IntentCategory.UNKNOWN
        assert result.confidence == 0.0

    def test_empty_response(self):
        """空响应降级为 unknown。"""
        result = parse_output("", "原始输入")
        assert result.intent == IntentCategory.UNKNOWN

    def test_unknown_intent_name(self):
        """LLM 返回未知意图名时降级。"""
        raw = '{"intent": "do_something_weird", "confidence": 0.9, "slots": []}'
        result = parse_output(raw, "做奇怪的事")
        assert result.intent == IntentCategory.UNKNOWN
        assert result.confidence <= 0.3

    def test_confidence_clamped_high(self):
        """confidence > 1.0 被钳制。"""
        raw = '{"intent": "begin_gesture_recognition", "confidence": 1.5, "slots": []}'
        result = parse_output(raw, "开启手势识别")
        assert result.confidence == 1.0

    def test_confidence_clamped_low(self):
        """confidence < 0.0 被钳制。"""
        raw = '{"intent": "begin_gesture_recognition", "confidence": -0.5, "slots": []}'
        result = parse_output(raw, "开启手势识别")
        assert result.confidence == 0.0

    def test_slots_not_a_list(self):
        """slots 不是数组时返回空。"""
        raw = '{"intent": "unknown", "confidence": 0.0, "slots": "invalid"}'
        result = parse_output(raw, "")
        assert result.slots == []

    def test_unknown_forced_zero_confidence(self):
        """unknown 意图强制置信度为 0.0。"""
        raw = '{"intent": "unknown", "confidence": 0.8, "slots": []}'
        result = parse_output(raw, "")
        assert result.confidence == 0.0


# ─── load_prompt_template 测试 ──────────────────────────────

class TestLoadPromptTemplate:
    """提示词加载测试。"""

    def test_loads_non_empty(self):
        prompt = load_prompt_template()
        assert len(prompt) > 0

    def test_loads_string(self):
        prompt = load_prompt_template()
        assert isinstance(prompt, str)

    def test_contains_key_words(self):
        """提示词包含关键意图定义。"""
        prompt = load_prompt_template()
        assert "begin_gesture_recognition" in prompt
        assert "close_gesture_recognition" in prompt
        assert "unknown" in prompt
