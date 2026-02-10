### WiseCardPackage 卡片包自动生成项目的探索。



## 目标

通过 WiseCardPackage 的具体例子来自动化生成卡片包。



当前实现思路： LangGraph + RAG 

（由于上下文受限，后续会探索结合 agent skill。其中 agent skill 复杂结合“思维链模式（CoT）” 来规定任务的具体流程，同时在 skill 的 resources 中添加具体的例子来让 LLM 学习。使用 skill 的目的是为了**防止上下文被压缩**）



RAG 通过搭建动态布局（DSL）的领域知识来让模型学习（在对话的生命周期中学习而不是长期学习）



## 参考项目

LangGraph 经典实战项目：https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/code_assistant/langgraph_code_assistant.ipynb



知乎介绍：https://zhuanlan.zhihu.com/p/1976720013428819800



![架构图](./pics/framework.png)



1. code_assistant_rag.ipynb 文件以 jupyter 的形式展示了整个过程

2. code-generation/main_draft.py 文件是整个流程的大体过程（比较粗糙）

3. code-generation/code_solution_agent 中具体对文件进行了划分，可以通过运行 code_solution_agent/main.py 来直接尝试。

这里通过 FastAPI 快速搭建了一个页面，可以运行 code_solution_agent/web/server.py 来运行，可以在页面看到整个 agent 迭代的过程。

具体的例子可以见 web-example-shown 中的视频和下载的 html 的例子。

**注意，这里是对整个流程进行调试，具体的执行结果需要看 LLM 模型的回复质量以及 RAG 知识库的质量。**

执行项目需要再 config/keys.json 中填入模型对应的 API



**未来的可行方案**

![架构图-future](./pics/framework-future.png)



1. intent recognition （意图识别）： question/reflection 进来后先分析，判断是否需要走 RAG 或者是直接进行 code generation
2. rag retriever：从 DSL 的知识库中进行向量匹配，匹配出 top-k 个知识，作为 model 的 context 传入给 code generation 中。
3. code generation：根据描述进行 code generation 或者是 debug
4. code check（评估）：通过评估来评价 code generation 的质量，同时给予反馈或报错
5. reflect：总结反馈或报错，重新走到 intent-recognition node



以此搭建具备自迭代能力的、具有领域知识的 agent。













