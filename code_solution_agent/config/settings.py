import os


# LangSmith 配置
def setup_langsmith(api_key: str, project_name: str = "code-assistant"):
    """Setup LangSmith tracing configuration."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name


# Redis 配置
REDIS_URL = "redis://localhost:6379"
REDIS_INDEX_NAME = "code_assistant"
REDOCUMENT_PATH = "./files/lcel_doc.txt"

# 最大迭代次数
MAX_ITERATIONS = 3

