import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

from config.settings import REDIS_URL, REDIS_INDEX_NAME, REDOCUMENT_PATH


def create_retriever(embedding_model: DashScopeEmbeddings, reset_index: bool = True):
    """
    Create and configure the RAG retriever.

    Args:
        embedding_model: The embedding model to use for vectorization
        reset_index: Whether to reset (delete and recreate) the index

    Returns:
        Configured retriever instance
    """
    # 加载文档
    loader = TextLoader(REDOCUMENT_PATH, encoding='utf-8')
    docs = loader.load()

    # 切分文档
    text_splitter = CharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=0,
        separator="\n\n\n --- \n\n\n",
        keep_separator=True
    )
    segments = text_splitter.split_documents(docs)

    # 配置 Redis 向量存储
    config = RedisConfig(
        index_name=REDIS_INDEX_NAME,
        redis_url=REDIS_URL
    )
    vector_store = RedisVectorStore(embedding_model, config=config)

    # 删除已有索引以避免数据重复
    if reset_index:
        try:
            vector_store.delete_index()
            print(f"Deleted existing index: {config.index_name}")
        except Exception as e:
            print(f"No existing index to delete or error deleting: {e}")

    # 添加文档
    vector_store.add_documents(segments)
    print(f"Added {len(segments)} documents to vector store")

    return vector_store.as_retriever()


def get_embedding_model(api_key: str, model_name: str = "text-embedding-v1") -> DashScopeEmbeddings:
    """
    Get or create the embedding model.

    Args:
        api_key: The DashScope API key
        model_name: The embedding model name

    Returns:
        Configured embedding model instance
    """
    if not os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = api_key

    return DashScopeEmbeddings(model=model_name)
