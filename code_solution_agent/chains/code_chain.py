from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import Runnable
from models.schemas import CodeSolution


def create_code_generation_chain(llm: ChatTongyi) -> Runnable:
    """
    Create the code generation chain with prompt template and LLM.

    Args:
        llm: The chat model to use

    Returns:
        Configured chain instance
    """
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant with expertise in LCEL, LangChain expression language.
    Here is a full set of LCEL documentation:  -------  {context}  -------
    Answer the user question based on the above provided documentation. Ensure any code you provide can be executed
    with all required imports and variables defined. Structure your answer with a description of the code solution.
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # Try to use structured output
    try:
        return code_gen_prompt | llm.with_structured_output(CodeSolution)
    except Exception as e:
        print(f"Warning: Could not use structured output: {e}")
        print("Falling back to plain text output")
        # Fallback to plain output
        return code_gen_prompt | llm


def get_chat_model(api_key: str, model_name: str = "qwen3-coder-plus") -> ChatTongyi:
    """
    Get the chat model instance.

    Args:
        api_key: The DashScope API key
        model_name: The model name

    Returns:
        Configured chat model instance
    """
    return ChatTongyi(model=model_name, api_key=api_key)
