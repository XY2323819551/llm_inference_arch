# LLM推理架构项目

## 项目概述
这是一个专注于大语言模型(LLM)推理架构的研究和实现项目。项目实现了多种推理方法，包括Chain of Thought (CoT)、React、Plan-and-Execute以及Self-Refine等推理架构，旨在提升LLM的推理能力和任务执行效果。

## 项目目录结构

```
.
├── inference_arch/          # 推理架构实现目录
│   ├── cot.py              # Chain of Thought推理实现
│   ├── react.py            # ReAct推理架构实现
│   ├── plan-and-execute.py # 计划与执行推理架构
│   ├── self-refine.py      # 自我优化推理架构
│   └── __init__.py         # 包初始化文件
├── llm_pool/               # LLM模型池
│   └── llm.py              # LLM基础实现
├── tool_pool/              # 工具池
│   └── serpapi.py          # SerpAPI搜索工具实现
├── docs/                   # 文档目录
│   ├── inference-cot.md    # Chain of Thought架构文档
│   ├── inference-react.md  # ReAct架构文档
│   ├── inference-refine.md # Self-Refine架构文档
│   └── inference-PlanAndExecute.md # Plan-and-Execute架构文档
├── Makefile               # 项目管理脚本
├── .env                   # 环境变量配置
└── .gitignore            # Git忽略配置文件
```

### 主要模块说明

#### 推理架构模块 (inference_arch)
- **cot.py**: 实现了Chain of Thought推理方法，通过引导模型展示思考过程来提升推理能力
- **react.py**: 实现了ReAct推理架构，结合了推理和行动的能力
- **plan-and-execute.py**: 实现了计划执行推理架构，模拟项目经理的工作方式
- **self-refine.py**: 实现了自优化推理架构，通过多轮优化提升输出质量

#### 模型池 (llm_pool)
- **llm.py**: 提供与大语言模型交互的核心功能，支持多种模型调用

#### 工具池 (tool_pool)
- **serpapi.py**: 实现了基于SerpAPI的网络搜索功能

#### 文档 (docs)
详细的技术文档，包括每种推理架构的实现原理、应用场景和示例：
- Chain of Thought推理架构文档
- ReAct推理架构文档
- Self-Refine推理架构文档
- Plan-and-Execute推理架构文档

## 运行方法

### 环境要求
- Python 3.8+
- 必要的Python包（建议使用虚拟环境）

### 安装步骤

1. 克隆项目
```bash
git clone [项目地址]
python -m inference_arch.cot
python -m inference_arch.plan-and-execute
python -m inference_arch.react
python -m inference_arch.self-refine
```

2. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填入必要的API密钥
```

### 项目维护

使用Makefile提供的命令进行项目维护：

```bash
# 清理项目
make clean

# 统计代码行数
make count
```

## 注意事项

- 确保在运行前正确配置所有必要的API密钥
- 建议在虚拟环境中运行项目
- 详细的架构说明请参考docs目录下的相关文档







