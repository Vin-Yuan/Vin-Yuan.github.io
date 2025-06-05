---
title: Agent
mathjax: true
date: 2025-06-03 15:12:59
caeegories: LLM, Agent
taes: LLM, Agent
---


## Bonus 架构图 (现代 Function Calling Agent 流程)
```python
User Input
   ↓
Embedding Model ───────→ 语义索引 (FAISS / Qdrant / Weaviate)
                             ↓
                 Top-K Function Candidates
                             ↓
            Inject into Prompt as JSON Schema
                             ↓
               LLM 生成 Function Call
                             ↓
          调用工具 / API + 结构校验 + 反馈
```
## MCP server 流程图
```plaintext
                    +-------------+
                    | MCP Client |
                    +------+-----+
                           |
                           | submit task / query result
                           v
                    +------+------+
                    | MCP Server  |
                    +------+------+
                           |
            +--------------+---------------+
            |                              |
            v                              v
     +-------------+               +---------------+
     |   Agent 1   |               |    Agent 2    |
     +-------------+               +---------------+
         |   ^                          |   ^
         |   | Task Execution           |   | Task Execution
         +---+--------------------------+---+
```