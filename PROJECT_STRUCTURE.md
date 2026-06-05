# 通用大模型验证框架 (General LLM Verification Framework) - 架构指南

本项目是一个通用的 LLM 功能验证框架，旨在为各种基于大模型的应用（如报告生成、自动化分析、内容转换等）提供标准化的开发底座。

## 1. 通用目录结构 (General Structure)

```text
llm_project_root/
├── config/             # 配置管理：模型参数、环境变量映射、业务开关
├── data/               # 数据中心：输入源 (inputs)、中间态缓存、输出结果 (outputs)
├── prompts/            # 模板仓库：针对不同任务阶段的 Prompt 模板 (.txt/.yaml)
├── src/                # 核心逻辑 (Source Code)
│   ├── client.py       # LLM 适配器：封装 API 调用、重试、Token 计数及流式控制
│   ├── processor.py    # 数据清洗器：负责输入数据的提取、清洗及格式化转换
│   ├── engine.py       # 业务执行引擎：负责编排业务流程（如目前的报告生成逻辑）
│   ├── main.py         # 框架入口：统一的启动与调度脚本
│   └── __init__.py     # 模块初始化
├── tests/              # 验证单元：覆盖数据处理、提示词效果及 API 稳定性测试
├── .env.example        # 敏感信息模板 (API Keys, Endpoints)
├── PROJECT_STRUCTURE.md # 本架构指南（通用版）
└── requirements.txt    # 核心依赖库
```

## 2. 核心组件职责 (Component Responsibilities)

- **`processor.py`**: 负责将非结构化的原始数据转化为模型可解析的“语境数据”。
- **`client.py`**: 负责与底层模型（如 OpenAI, DeepSeek, Local LLM）通信，确保接口的统一性。
- **`engine.py`**: 框架的“大脑”，根据配置加载对应的 `prompts` 和 `data`，驱动模型完成特定任务。

## 3. 扩展性说明 (Extensibility)

- **更换任务**: 只需在 `prompts/` 下新增任务模板，并在 `engine.py` 中定义新的编排流程。
- **更换模型**: 在 `config/` 或 `.env` 中修改配置，`client.py` 会自动适配。
- **数据适配**: 在 `processor.py` 中增加新的解析方法。

---
*本文档定义了项目的标准化布局，所有业务功能开发应在此脚手架基础上扩展。*
