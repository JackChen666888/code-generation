# Code Assistant Web Visualization

这是一个为 Code Assistant (LangGraph + RAG) 智能体添加的 Web 可视化界面，可以实时展示每个阶段的输入和输出。

## 功能特点

- **实时追踪**：可视化显示 LangGraph 工作流的每个节点执行状态
- **输入/输出展示**：清晰展示每个节点的输入和输出数据
- **执行时间线**：记录完整的执行过程，包括每个阶段的时间戳
- **无需修改原始代码**：通过包装器模式实现，不影响原有代码逻辑

## 工作流节点

1. **RAG Retrieve** - 从知识库检索相关文档
2. **Generate** - 基于检索内容生成代码解决方案
3. **Check Code** - 检查生成的代码是否有错误
4. **Reflect** - 如果有错误，反思并总结错误信息用于重试

## 安装依赖

```bash
pip install -r web/requirements.txt
```

## 启动服务器

```bash
cd web
python server.py
```

服务器将在 `http://localhost:8000` 启动

## 使用方法

1. 在浏览器中打开 `http://localhost:8000`
2. 在输入框中输入问题（例如："如何使用 RunnableLambda 传递字符串？"）
3. 点击"提交查询"按钮
4. 观察界面上各节点的执行状态和输入/输出

## 项目结构

```
web/
├── server.py          # FastAPI 服务器
├── requirements.txt    # Python 依赖
├── README.md         # 本文档
└── templates/
    └── index.html     # 前端界面
```

## 技术实现

### 后端 (server.py)

- `TracedCodeAssistant` 类：继承自原始 `CodeAssistant`，添加追踪功能
- `_execute_with_tracing` 方法：手动执行工作流并在每个节点发送更新
- SSE 端点 `/api/monitor/{session_id}`：向前端推送实时更新

### 前端 (index.html)

- 使用 Server-Sent Events (SSE) 接收后端推送的更新
- 可视化节点状态（空闲/活跃/成功/错误）
- 实时更新输入/输出面板
- 显示执行时间线

## 配置说明

确保原始 Code Assistant 的配置文件正确设置：
- `config/configs.py` 中的 API 密钥
- `config/settings.py` 中的各种配置

## 注意事项

- 服务器需要网络连接以访问 API
- 确保原始 Code Assistant 可以正常工作
- 首次运行可能需要等待模型加载
