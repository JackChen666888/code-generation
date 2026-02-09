"""
Code Assistant API

This module provides a simple API for using the Code Assistant functionality.
Import this module and use the `CodeAssistant` class to query the assistant.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from config.configs import load_key
from config.settings import setup_langsmith
from graph.nodes import GraphState
from rag.retriever import create_retriever, get_embedding_model
from chains.code_chain import create_code_generation_chain, get_chat_model
from graph.nodes import rag_retrieve, generate, code_check, reflect
from graph.edges import decide_to_finish


class CodeAssistant:
    """
    Code Assistant - A RAG-powered code generation assistant with error reflection.

    This class provides a simple interface for querying the code assistant.
    It uses LangGraph to manage a workflow that:
    1. Retrieves relevant documentation from RAG
    2. Generates code solutions
    3. Checks the code for errors
    4. Reflects on errors and retries if needed
    """

    def __init__(self, api_key: str = None, reset_index: bool = True):
        """
        Initialize the Code Assistant.

        Args:
            api_key: DashScope API key (if None, will load from config)
            reset_index: Whether to reset the RAG index
        """
        self.api_key = api_key or load_key('DASHSCOPE_API_KEY')
        self.reset_index = reset_index
        self._initialized = False

    def _initialize(self):
        """
        Lazy initialization of components.
        """
        if self._initialized:
            return

        # Setup LangSmith
        setup_langsmith(
            api_key=load_key('LANGSMITH_API_KEY'),
            project_name="code-assistant"
        )

        # Create embedding model
        embedding_model = get_embedding_model(self.api_key)

        # Create retriever
        self.retriever = create_retriever(embedding_model, reset_index=self.reset_index)

        # Create chat model
        self.llm = get_chat_model(self.api_key, model_name="qwen3-coder-plus")

        # Create code generation chain
        self.code_gen_chain = create_code_generation_chain(self.llm)

        # Build workflow
        self.graph = self._build_workflow()

        self._initialized = True

    def _build_workflow(self):
        """
        Build and compile the LangGraph workflow.
        """
        workflow = StateGraph(GraphState)

        # Add nodes with partial functions (binding dependencies)
        workflow.add_node("rag_retrieve", lambda state: rag_retrieve(state, self.retriever))
        workflow.add_node("generate", lambda state: generate(state, self.code_gen_chain))
        workflow.add_node("check_code", code_check)
        workflow.add_node("reflect", lambda state: reflect(state, self.llm))

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

        return workflow.compile()

    def query(self, question: str) -> dict:
        """
        Query the Code Assistant with a question.

        Args:
            question: The user question

        Returns:
            dict: The result containing:
                - messages: List of conversation messages
                - generation: The final code solution
                - error: Error state ('yes' or 'no')
                - iterations: Number of iterations performed
                - rag_context: Retrieved context
                - error_summary: Summary of errors (if any)
        """
        self._initialize()

        initial_state = {
            "messages": [("user", question)],
            "iterations": 0,
            "error": "",
            "rag_context": [],
            "error_summary": ''
        }

        return self.graph.invoke(initial_state)

    def get_code(self, question: str) -> str:
        """
        Query the assistant and return only the generated code.

        Args:
            question: The user question

        Returns:
            str: The generated code (imports + code block)
        """
        result = self.query(question)
        generation = result.get('generation')
        if generation:
            return f"{generation.imports}\n{generation.code}"
        return ""


# Convenience function for quick usage
def query_assistant(question: str, api_key: str = None) -> dict:
    """
    Quick query function for the Code Assistant.

    Args:
        question: The user question
        api_key: DashScope API key (optional)

    Returns:
        dict: The query result
    """
    assistant = CodeAssistant(api_key=api_key)
    return assistant.query(question)
