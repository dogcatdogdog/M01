# CLAUDE.md

<!-- 本文件基于 centminmod/my-claude-code-setup (2.4k stars) 模板改编 -->
<!-- 角色：主控 (Orchestrator)，负责规划、调度子代理、验证结果 -->

## 项目信息

- **项目:** 通用大模型验证框架 (LLM Verification Framework)
- **技术栈:** Python 3.10+, OpenAI SDK (DeepSeek 兼容), Pydantic
- **入口:** `src/main.py`
- **定位:** 标准化 LLM 验证脚手架，不含 Web 框架，纯 Python 脚本

## 沟通规范

- **语言:** 对话用中文，代码注释用中文，变量/函数名用英文
- **主控模式:** 我是主控（Orchestrator）。复杂任务由我做规划 → 分配子代理执行 → 验证结果
- **不确定就问:** 需求不清晰时停下来确认，不要猜

## 工作流程

### 1. 默认走计划模式

任何非简单任务（≥3 步或有架构决策）先做计划，确认后再动手。如果中途出错，停下来重新计划。

### 2. 子代理策略（保持主上下文干净）

- 研究、探索、并行分析 → 交给子代理
- 复杂代码编写 → 交给 coder 子代理
- 测试编写和运行 → 交给 tester 子代理
- 每个子代理只做一件事，专注执行

### 3. 做完前必须验证

- 不验证不算完成
- 问自己："一个高级工程师会批准这个吗？"
- 有测试就跑测试，没测试就手动验证

### 4. 简洁优先

- 用最少的代码解决问题
- 不要为单次使用创建抽象
- 如果改动感觉 hacky，停下来想更好的方案
- 不顺手重构无关代码

### 5. 自主修 Bug

- 收到 Bug 报告：直接修，不要反复问
- 指着日志、报错 → 然后自己解决
- 不让用户做"上下文切换"

## 常用命令

```bash
# 运行主程序
python src/main.py

# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 代码检查
python -c "from src.main import *; print('Import OK')"
```

## 架构

```
├── config/          # 模型参数、环境变量映射
├── data/            # inputs/ → 中间态 → outputs/
├── prompts/         # Prompt 模板 (.txt/.yaml)
├── src/
│   ├── client.py    # LLM 适配器（OpenAI SDK 封装）
│   ├── processor.py # 数据清洗与格式化
│   ├── engine.py    # 业务编排引擎
│   └── main.py      # 入口
├── tests/
├── CLAUDE.md
└── requirements.txt
```

## 编码规范（仅列出和默认不同的）

- **类型注解:** 所有函数参数和返回值必须有类型注解
- **导入:** 全部放在文件顶部
- **LLM 调用:** 统一走 `client.py`，业务代码不直接调 `openai.OpenAI`
- **配置:** API Key / URL 从环境变量读取，不硬编码
- **命名:** 变量/函数 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`

## 子代理

本项目配置了两个自定义子代理（后续配置）：

| 子代理 | 用途 | 触发场景 |
|--------|------|----------|
| `coder` | 编写和修改代码 | 实现新功能、重构、修 Bug |
| `tester` | 编写测试和运行测试 | 新功能需要测试、验证改动、跑全量测试 |

子代理配置方式：`/agents` → 创建 → 定义 system prompt 和工具权限。

## 记忆系统

任务进行中的重要信息可以用 auto memory 记录，跨会话保留：

| 记录时机 | 记录内容 |
|----------|----------|
| 新约定达成时 | 为什么这样约定、如何应用 |
| 架构决策时 | 选择了什么方案、原因 |
| 踩坑修 Bug 时 | 问题现象、根因、修复方式 |
| 构建/测试命令变更时 | 新命令 |

---

## 参考资料

- 本模板基于: https://github.com/centminmod/my-claude-code-setup (2390+ stars)
- 主控模式参考: BorisCherny ACE (Agentic Context Engineering)
