"""NLU 端到端测试 — 使用真实 Qwen API 验证提示词质量。

运行前需确保 config/.env 已配置有效的 API_KEY。

用法:
    pytest tests/test_nlu_e2e.py -v          # 全部测试
    pytest tests/test_nlu_e2e.py -v -k "begin"  # 只跑开启类
"""

import pytest
from src.engine import NLUEngine
from src.schemas import IntentCategory


@pytest.fixture(scope="module")
def engine():
    """模块级 fixture，所有 E2E 测试共用引擎（提示词只加载一次）。"""
    return NLUEngine(simulate=False)


# ─── 标准表述 ───────────────────────────────────────────────

class TestStandardPhrasing:
    """标准表述：直接用文档中的示例。"""

    def test_begin_standard(self, engine):
        result = engine.parse("开启手势识别")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION
        assert result.confidence >= 0.7

    def test_close_standard(self, engine):
        result = engine.parse("关闭手势识别")
        assert result.intent == IntentCategory.CLOSE_GESTURE_RECOGNITION
        assert result.confidence >= 0.7


# ─── 同义动词 ───────────────────────────────────────────────

class TestSynonymVerbs:
    """同义动词：开启→打开/启动/激活，关闭→关掉/停止/结束/退出。"""

    @pytest.mark.parametrize("text", [
        "打开手势识别",
        "启动手势识别",
        "激活手势识别",
    ])
    def test_begin_synonyms(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION

    @pytest.mark.parametrize("text", [
        "关掉手势识别",
        "停止手势识别",
        "结束手势识别",
        "退出手势识别",
    ])
    def test_close_synonyms(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.CLOSE_GESTURE_RECOGNITION


# ─── 同义宾语 ──────────────────────────────────────────────

class TestSynonymObjects:
    """同义宾语：手势识别→手势控制/手势模式/手势。"""

    @pytest.mark.parametrize("text", [
        "开启手势控制",
        "开启手势模式",
    ])
    def test_begin_object_variants(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION

    def test_begin_gesture_short(self, engine):
        """'开启手势'：简略说法，Qwen FC 可能判 unknown（缺完整宾语），两种均可。"""
        result = engine.parse("开启手势")
        assert result.intent in (
            IntentCategory.BEGIN_GESTURE_RECOGNITION,
            IntentCategory.UNKNOWN,
        )

    @pytest.mark.parametrize("text", [
        "关闭手势控制",
        "关闭手势模式",
    ])
    def test_close_object_variants(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.CLOSE_GESTURE_RECOGNITION

    def test_close_gesture_short(self, engine):
        """'关闭手势'：简略说法，Qwen FC 可能判 unknown，两种均可。"""
        result = engine.parse("关闭手势")
        assert result.intent in (
            IntentCategory.CLOSE_GESTURE_RECOGNITION,
            IntentCategory.UNKNOWN,
        )


# ─── 语序变化 ──────────────────────────────────────────────

class TestWordOrderVariations:
    """语序变化：把字句等中文常见句式。"""

    def test_ba_structure_begin(self, engine):
        result = engine.parse("把手势识别打开")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION

    def test_ba_structure_close(self, engine):
        result = engine.parse("把手势识别关掉")
        assert result.intent == IntentCategory.CLOSE_GESTURE_RECOGNITION

    def test_subject_first(self, engine):
        """主语前置。"""
        result = engine.parse("手势识别开启")
        assert result.intent == IntentCategory.BEGIN_GESTURE_RECOGNITION


# ─── 无关输入 ─────────────────────────────────────────────

class TestIrrelevantInput:
    """与手势识别完全无关的输入 → unknown。"""

    @pytest.mark.parametrize("text", [
        "今天天气怎么样",
        "你好",
        "帮我导航到北京",
        "现在几点了",
    ])
    def test_irrelevant(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.UNKNOWN


# ─── 无意义输入 ────────────────────────────────────────────

class TestNonsenseInput:
    """无意义或模糊输入 → unknown。"""

    @pytest.mark.parametrize("text", [
        "",
        "嗯",
        "啊哦呃",
    ])
    def test_nonsense(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.UNKNOWN


# ─── 不完整输入 ────────────────────────────────────────────

class TestIncompleteInput:
    """不完整命令（缺宾语）→ unknown。"""

    @pytest.mark.parametrize("text", [
        "开启",
        "关闭",
        "识别",
        "手势",
        "开",
        "关",
    ])
    def test_incomplete(self, engine, text):
        result = engine.parse(text)
        assert result.intent == IntentCategory.UNKNOWN


# ─── 否定/反问 ─────────────────────────────────────────────

class TestNegation:
    """否定句和反问句 → unknown（非开启/关闭指令）。"""

    @pytest.mark.parametrize("text", [
        "我不需要手势识别",
        "不要开启手势识别",
    ])
    def test_negation(self, engine, text):
        result = engine.parse(text)
        # 否定句式不属于开启/关闭指令，归为 unknown
        assert result.intent == IntentCategory.UNKNOWN


# ─── 中英混杂 ─────────────────────────────────────────────

class TestMixedLang:
    """中英混杂输入。"""

    @pytest.mark.parametrize("text", [
        "open 手势识别",
        "close 手势",
    ])
    def test_mixed_lang(self, engine, text):
        result = engine.parse(text)
        # 能识别则识别，不能则 unknown（不崩溃）
        assert result.intent in (
            IntentCategory.BEGIN_GESTURE_RECOGNITION,
            IntentCategory.CLOSE_GESTURE_RECOGNITION,
            IntentCategory.UNKNOWN,
        )
