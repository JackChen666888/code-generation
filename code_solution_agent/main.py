"""
Code Assistant - Main Entry Point

This module serves as the entry point for the Code Assistant application.
It initializes all components (config, RAG, chains, graph) and runs the workflow.
"""

import sys
import os

# Add parent directory to path to import from config folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import END, StateGraph, START
from graph.nodes import GraphState

# Local imports
from config.configs import load_key
from config.settings import setup_langsmith
from rag.retriever import create_retriever, get_embedding_model
from chains.code_chain import create_code_generation_chain, get_chat_model
from graph.nodes import rag_retrieve, generate, code_check, reflect
from graph.edges import decide_to_finish


def initialize_components():
    """
    Initialize all components needed for the Code Assistant.

    Returns:
        tuple: (retriever, code_gen_chain, llm)
    """
    # Setup LangSmith
    setup_langsmith(
        api_key=load_key('LANGSMITH_API_KEY'),
        project_name="code-assistant"
    )

    # Get API key
    dashscope_api_key = load_key('DASHSCOPE_API_KEY')

    # Create embedding model
    embedding_model = get_embedding_model(dashscope_api_key)

    # Create retriever
    retriever = create_retriever(embedding_model, reset_index=True)

    # Create chat model
    llm = get_chat_model(dashscope_api_key, model_name="qwen3-coder-plus")

    # Create code generation chain
    code_gen_chain = create_code_generation_chain(llm)

    return retriever, code_gen_chain, llm


def build_workflow(retriever, code_gen_chain, llm):
    """
    Build and compile the LangGraph workflow.

    Args:
        retriever: The RAG retriever instance
        code_gen_chain: The code generation chain
        llm: The chat model instance

    Returns:
        Compiled LangGraph workflow
    """
    # Create state graph
    workflow = StateGraph(GraphState)

    # Add nodes with partial functions (binding dependencies)
    workflow.add_node("rag_retrieve", lambda state: rag_retrieve(state, retriever))
    workflow.add_node("generate", lambda state: generate(state, code_gen_chain))
    workflow.add_node("check_code", code_check)
    workflow.add_node("reflect", lambda state: reflect(state, llm))

    # Build graph edges
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

    # Compile graph
    return workflow.compile()


def run_query(graph: object, question: str) -> dict:
    """
    Run a query through the Code Assistant graph.

    Args:
        graph: The compiled LangGraph
        question: The user question

    Returns:
        dict: The result containing the solution and messages
    """
    initial_state = {
        "messages": [("user", question)],
        "iterations": 0,
        "error": "",
        "rag_context": [],
        "error_summary": ''
    }

    return graph.invoke(initial_state)


def main():
    """
    Main function to run the Code Assistant.
    """
    print("Initializing Code Assistant...")

    # Initialize components
    retriever, code_gen_chain, llm = initialize_components()
    print("Components initialized successfully!")

    # Build workflow
    graph = build_workflow(retriever, code_gen_chain, llm)
    print("Workflow built successfully!")

    # Example question
    question = "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
    print(f"\nQuery: {question}\n")

    # Run query
    result = run_query(graph, question)

    # Display result
    print("\n=== Result ===")
    print(f"Final messages: {result['messages']}")
    print(f"Final error state: {result['error']}")
    print(f"Total iterations: {result['iterations']}")


if __name__ == "__main__":
    main()
