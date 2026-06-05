"""NLU 意图解析模块入口。

用法：
    python src/main.py                          # 交互式
    python src/main.py "开启手势识别"             # 单次解析
    python src/main.py --file test_cases.txt     # 批量测试
"""

import sys
import json
import logging
import argparse
from typing import NoReturn

from src.engine import get_engine
from src.schemas import IntentResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _result_to_dict(result: IntentResult) -> dict:
    """将 IntentResult 转为可序列化的字典。"""
    return {
        "intent": result.intent.value,
        "confidence": result.confidence,
        "slots": [{"key": s.key, "value": s.value} for s in result.slots],
        "routing_target": result.routing_target,
        "raw_text": result.raw_text,
    }


def print_result(result: IntentResult) -> None:
    """格式化打印解析结果。"""
    output = _result_to_dict(result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def interactive_loop() -> NoReturn:
    """交互式 NLU 解析循环。"""
    print("=" * 50)
    print("  NLU 意图解析 — 交互模式")
    print("  输入语音文本，输入 quit / exit 退出")
    print("=" * 50)

    engine = get_engine()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            sys.exit(0)

        if not user_input:
            continue

        result = engine.parse(user_input)
        print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLU 意图解析模块 — 多模态交互系统"
    )
    parser.add_argument(
        "text", nargs="?", default=None,
        help="要解析的语音文本（不传则进入交互模式）",
    )
    parser.add_argument(
        "--file", "-f", default=None,
        help="批量测试文件（每行一条输入）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示 DEBUG 级别日志",
    )
    parser.add_argument(
        "--no-simulate", action="store_true",
        help="不模拟 API 调用，只输出 JSON",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    engine = get_engine()
    if args.no_simulate:
        engine.simulate = False

    # 批量文件模式
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"批量解析 {len(lines)} 条输入:\n")
        passed = 0
        for i, line in enumerate(lines, 1):
            # 支持格式: "输入文本" 或 输入文本||期望意图
            if "||" in line:
                text, expected = line.split("||", 1)
                text, expected = text.strip(), expected.strip()
            else:
                text = line.strip()
                expected = None

            result = engine.parse(text)
            actual_intent = result.intent.value
            match = ""
            if expected:
                ok = actual_intent == expected
                match = " ✅" if ok else f" ❌ (期望: {expected})"
                if ok:
                    passed += 1
            print(f"[{i}] '{text}' → {actual_intent} (conf={result.confidence:.2f}){match}")

        if expected is not None:
            print(f"\n通过: {passed}/{len(lines)}")
        return

    # 单次解析模式
    if args.text:
        result = engine.parse(args.text)
        print_result(result)
        return

    # 默认交互模式
    interactive_loop()


if __name__ == "__main__":
    main()
