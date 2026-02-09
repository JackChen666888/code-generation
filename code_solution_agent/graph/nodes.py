from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from typing import List, TypedDict



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

def rag_retrieve(state: GraphState, retriever) -> GraphState:
    print("---RAG process---")

    if state.get("error_summary"):
        query = state["error_summary"]
        print(f"Retrieving based on error summary: {query}")
    elif state["error"] in ("", "no"):
        query = state["messages"][0][1]
        print("Retrieving based on original question")
    else:
        query = state["messages"][-1][1]
        print("Retrieving based on last message")

    relative_segments = retriever.invoke(query, k=2)
    texts = [seg.page_content for seg in relative_segments]
    print(texts)

    return {
        "messages": state["messages"],
        "iterations": state["iterations"],
        "error": state["error"],
        "rag_context": texts,
        "generation": state.get("generation"),
        "error_summary": state.get("error_summary", "")
    }


def generate(state: GraphState, code_gen_chain: Runnable) -> GraphState:
    print("---GENERATING CODE SOLUTION---")

    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
    rag_context = state["rag_context"]

    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:")]

    try:
        code_solution = code_gen_chain.invoke({"context": rag_context, "messages": messages})
    except Exception as e:
        print(f"Error generating code solution: {e}")
        code_solution = None

    if code_solution is None:
        print("---NO CODE SOLUTION GENERATED---")
        messages += [("assistant", f"Failed to generate a code solution. RAG context length: {len(rag_context)}")]
        return {
            "generation": None,
            "messages": messages,
            "iterations": iterations + 1,
            "error": "yes",
            "rag_context": rag_context,
            "error_summary": "Failed to generate code solution"
        }

    if isinstance(code_solution, str):
        print("---PLAIN TEXT RESPONSE---")
        prefix = code_solution
        imports = ""
        code = ""
    else:
        prefix = getattr(code_solution, "prefix", "") or ""
        imports = getattr(code_solution, "imports", "") or ""
        code = getattr(code_solution, "code", "") or ""

    response = f"Prefix: {prefix}, Imports: {imports}, Code: {code}"
    messages += [("assistant", response)]

    iterations = iterations + 1
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": state["error"],
        "rag_context": rag_context,
        "error_summary": state.get("error_summary", "")
    }


def code_check(state: GraphState) -> GraphState:
    print("---CHECKING CODE---")

    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    if code_solution is None:
        print("---NO CODE SOLUTION TO CHECK---")
        return {
            "generation": None,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
            "rag_context": state["rag_context"],
            "error_summary": state.get("error_summary", "No code solution generated")
        }

    if isinstance(code_solution, str):
        imports = ""
        code = code_solution
    else:
        imports = getattr(code_solution, "imports", "") or ""
        code = getattr(code_solution, "code", "") or ""

    if not code:
        print("---NO CODE TO CHECK---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
            "rag_context": state["rag_context"],
            "error_summary": state.get("error_summary", "No code generated")
        }

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
            "error": (state["error"] or "yes"),
            "rag_context": state["rag_context"],
            "error_summary": state.get("error_summary", "")
        }

    try:
        exec(imports + "" + code)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": (state["error"] or "yes"),
            "rag_context": state["rag_context"],
            "error_summary": state.get("error_summary", "")
        }

    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
        "rag_context": state["rag_context"],
        "error_summary": state.get("error_summary", "")
    }


def reflect(state: GraphState, llm: ChatTongyi) -> GraphState:
    print("---SUMMARIZING ERROR---")

    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]
    rag_context = state["rag_context"]

    error_message = messages[-1][1] if messages else ""

    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个编程助手。请总结以下错误信息，提取关键错误点和相关代码部分，为后续的 RAG 检索提供简洁的查询。"),
        ("user", "错误信息: {error}")
    ])
    reflection_chain = reflection_prompt | llm

    try:
        error_summary = reflection_chain.invoke({"error": error_message})
        error_summary_text = error_summary.content if hasattr(error_summary, "content") else str(error_summary)
    except Exception as e:
        print(f"Error during reflection: {e}")
        error_summary_text = str(error_message)

    messages += [("assistant", f"Error summary: {error_summary_text}")]
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "yes",
        "rag_context": rag_context,
        "error_summary": error_summary_text
    }
