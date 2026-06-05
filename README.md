# M01 — 多模态交互 NLU 意图解析模块

通用大模型验证框架的 NLU 意图解析原型，对应多模态交互模块（M-01）第 1.2 节：**语音文本 → LLM 解析意图 → 结构化输出 → 模拟 API 调用**。

## 功能

```
输入: "开启手势识别"
  → function calling (Qwen API)
  → 输出: {intent: "begin_gesture_recognition", confidence: 0.95, routing: "open_gesture_recognition"}
  → 模拟: [模拟] 已调用 open_gesture_recognition() — 手势识别功能已开启
```

### 支持的意图

| 意图 | 触发示例 | 路由目标 |
|------|---------|----------|
| `begin_gesture_recognition` | "开启手势识别"、"打开手势控制" | `open_gesture_recognition()` |
| `close_gesture_recognition` | "关闭手势识别"、"关掉手势" | `close_gesture_recognition()` |
| `unknown` | "今天天气怎么样"、注入攻击 | `fallback_handler()` |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp config/.env.example config/.env
# 编辑 config/.env，填入阿里云百炼 API Key
```

格式：
```
API_KEY="sk-xxx"
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL="qwen-turbo"
```

### 3. 运行

```bash
# 单次解析
python src/main.py "开启手势识别"

# 交互模式
python src/main.py

# 批量测试（支持 || 期望意图标注）
python src/main.py --file test_cases.txt
```

## 项目结构

```
M01/
├── config/
│   ├── settings.py            # 配置管理（从 config/.env 加载）
│   └── .env.example           # API 配置模板
├── src/
│   ├── schemas.py             # 意图枚举、数据结构、Function Schema
│   ├── client.py              # LLM 适配器（OpenAI 兼容 API）
│   ├── processor.py           # 输入格式化 + JSON 解析 + 注入检测
│   ├── engine.py              # 编排引擎（FC 为主，文本兜底）
│   └── main.py                # CLI 入口
├── prompts/
│   └── nlu_intent_parsing.md  # NLU 意图解析系统提示词
├── tests/
│   ├── test_schemas.py        # 数据结构测试（11 用例）
│   ├── test_processor.py      # 数据处理测试（32 用例）
│   ├── test_engine.py         # 引擎测试 + Mock（8 用例）
│   ├── test_nlu_e2e.py        # 端到端真实 API 测试（35 用例）
│   └── test_adversarial.py    # 对抗测试：注入/安全/边界（58 用例）
├── .claude/
│   └── agents/
│       ├── coder.md           # 代码编写子代理
│       └── tester.md          # 测试编写子代理
└── CLAUDE.md                  # Claude Code 项目配置
```

## 测试

```bash
# 全部测试（144 用例）
pytest tests/ -v

# 只跑对抗测试
pytest tests/test_adversarial.py -v

# 只跑 E2E（需要 API Key）
pytest tests/test_nlu_e2e.py -v
```

## 安全架构

```
输入文本
  → detect_injection()           ← 代码层前置过滤
     ├─ JSON 结构 → 拦截
     ├─ 全英文输入 → 拦截
     ├─ 注入关键词 → 拦截
     └─ 超长输入 → 拦截
  → function calling (LLM)       ← API 层面 Schema 约束
  → 降级: 文本解析                ← 兜底路径
```

## 技术栈

- **Python** 3.10+
- **Qwen-turbo** (阿里云百炼 DashScope, OpenAI 兼容 API)
- **Function Calling** 强制结构化输出
- **pytest** 144 用例全覆盖
