from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import os 
from config.configs import load_key
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict

# LangSmith 配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = load_key('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = "code-assistant"

# rag -----------------------
loader = TextLoader("./files/lcel_doc.txt", encoding='utf-8')
docs = loader.load()
# 切分文档
text_splitter = CharacterTextSplitter(chunk_size = 5000, 
                    chunk_overlap = 0, separator="\n\n\n --- \n\n\n", keep_separator = True)

segments = text_splitter.split_documents(docs)

# 导入 embedding mode
if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("DASHSCOPE_API_KEY")

embedding_model = DashScopeEmbeddings(model = "text-embedding-v1")

redis_url = "redis://localhost:6379"
config = RedisConfig(
    index_name = "code_assistant",
    redis_url = redis_url
)
vector_store = RedisVectorStore(embedding_model, config = config)

# 删除已有索引以避免数据重复
try:
    vector_store.delete_index()
    print(f"Deleted existing index: {config.index_name}")
except Exception as e:
    print(f"No existing index to delete or error deleting: {e}")


# 添加文档
vector_store.add_documents(segments)
print(f"Added {len(segments)} documents to vector store")

# 获取 retriever
retriever = vector_store.as_retriever()

# rag -----------------------


# Data model (规定 LLM 的输出格式)
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


# prompt 模板
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
    Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user
    question based on the above provided documentation. Ensure any code you provide can be executed \n
    with all required imports and variables defined. Structure your answer with a description of the code solution. \n
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# llm
llm = ChatTongyi(
    model="qwen3-coder-plus",
    api_key=load_key('DASHSCOPE_API_KEY')
)

# 链式搭建
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)


# 规定图节点的基本数据类型
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    error: str          # 判断是否出现错误
    messages: List      # 记录历史消息
    generation: str     # 记录 code generation 给出的 code
    iterations: int     # 记录迭代次数，超出最大次数直接终止
    rag_context: List   # 记录 rag 得到的 context
    error_summary: str  # 记录 reflect 节点总结的错误信息


# Max tries
max_iterations = 3

### rag Nodes
def rag_retrieve(state: GraphState) -> GraphState:
    """
    从 RAG 中获取领域知识
    """
    print("---RAG process---")

    # 查询 rag
    if state.get("error_summary"):
        query = state["error_summary"]
        print(f"Retrieving based on error summary: {query}")
    elif state['error'] in ('', 'no'):
        query = state['messages'][0][1]
        print(f"Retrieving based on original question")
    else:
        query = state['messages'][-1][1]
        print(f"Retrieving based on last message")

    # 定义 rag 的数量
    relative_segmens = retriever.invoke(query, k = 2)

    texts = [seg.page_content for seg in relative_segmens]
    print(texts)
    
    return {
          "messages": state["messages"],
          "iterations": state["iterations"],
          "error": state["error"],
          "rag_context": texts,
          "generation": state.get("generation", ""),
          "error_summary": state.get("error_summary", "")
      }
    


def generate(state: GraphState):
    """
    生成 code
    """
    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
    rag_context = state["rag_context"]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]

    # Solution
    code_solution = code_gen_chain.invoke(
        {"context": rag_context, "messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {
        "generation": code_solution, 
        "messages": messages, 
        "iterations": iterations,
        'error': state["error"],
        "rag_context": rag_context,
        "error_summary": state.get("error_summary", "")
    }


def code_check(state: GraphState):
    """
    检查生成的 code 是否存在错误
    """
    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
            "rag_context": state['rag_context'],
            "error_summary": state.get("error_summary", "")
        }

    # Check execution
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
            "rag_context": state['rag_context'],
            "error_summary": state.get("error_summary", "")
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
        "rag_context": state['rag_context'],
        "error_summary": state.get("error_summary", "")
    }


def reflect(state: GraphState):
    """
    基于错误反思
    """

    print("---SUMMARIZING ERROR---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]
    rag_context = state['rag_context']

    # 获取最后一条错误消息
    error_message = messages[-1][1] if messages else ""

    # 使用 LLM 总结/浓缩错误信息
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个编程助手。请总结以下错误信息，提取关键错误点和相关代码部分，为后续的 RAG 检索提供简洁的查询。"),
            ("user", "错误信息:\n{error}")
        ]
    )
    reflection_chain = reflection_prompt | llm
    error_summary = reflection_chain.invoke({"error": error_message})

    messages += [("assistant", f"Error summary: {error_summary}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations, 
            "error": "yes", 'rag_context': rag_context, "error_summary": error_summary.content}


### Edges


def decide_to_finish(state: GraphState):
    """
    决定是否需要结束
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "reflect"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("rag_retrieve", rag_retrieve) # rag
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("reflect", reflect)  # reflect

# Build graph
workflow.add_edge(START, "rag_retrieve")
workflow.add_edge("rag_retrieve", "generate")
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect"
    },
)
workflow.add_edge("reflect", "rag_retrieve")
graph = workflow.compile()

question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
solution = graph.invoke({"messages": [("user", question)], "iterations": 0, "error": "", "rag_context": [],
                         "error_summary": ''})
