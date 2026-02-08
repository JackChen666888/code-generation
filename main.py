from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import json
from datetime import datetime
import asyncio
import threading
import os
from pathlib import Path

# Add parent directory to path for importing config
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.configs import load_key
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = load_key('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = "code-assistant"

# DashScope API Key
if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = load_key("DASHSCOPE_API_KEY")

app = FastAPI(title="Agent Workflow Visualizer", version="1.0.0")

# Storage for workflow execution data
workflow_runs: Dict[str, Dict[str, Any]] = {}

# Thread lock for protecting workflow_runs access
workflow_runs_lock = threading.Lock()

# Load documents for RAG
parent_dir = Path(__file__).parent
doc_path = parent_dir / "files" / "lcel_doc.txt"

loader = TextLoader(str(doc_path), encoding='utf-8')
docs = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=0,
    separator="\n\n\n --- \n\n\n",
    keep_separator=True
)

segments = text_splitter.split_documents(docs)

# Setup embedding model
embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

# Setup Redis vector store
redis_url = "redis://localhost:6379"
config = RedisConfig(index_name="code_assistant", redis_url=redis_url)
vector_store = RedisVectorStore(embedding_model, config=config)

# Try to delete existing index to avoid duplicates
try:
    vector_store.delete_index()
    print(f"Deleted existing index: {config.index_name}")
except Exception as e:
    print(f"No existing index to delete or error deleting: {e}")

# Add documents
vector_store.add_documents(segments)
print(f"Added {len(segments)} documents to vector store")

# Get retriever
retriever = vector_store.as_retriever()

# Data model for LLM output
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


# Prompt template
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in LCEL, LangChain expression language.
    Here is a full set of LCEL documentation:  -------  {context}  -------  Answer the user
    question based on the above provided documentation. Ensure any code you provide can be executed
    with all required imports and variables defined. Structure your answer with a description of the code solution.
    Then list the imports. And finally list the functioning code block. Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# LLM
llm = ChatTongyi(model="qwen3-coder-plus", api_key=load_key('DASHSCOPE_API_KEY'))

# Chain setup
code_gen_chain = code_gen_prompt | llm.with_structured_output(code)

# Graph Definition
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

# Node execution wrapper with logging
def wrap_node(node_name: str, node_func):
    """Wraps a node function to log input and output."""
    def wrapped(state: GraphState) -> GraphState:
        run_id = state.get("_run_id", "default")

        # Initialize if not exists
        with workflow_runs_lock:
            if run_id not in workflow_runs:
                workflow_runs[run_id] = {
                    "id": run_id,
                    "started_at": datetime.now().isoformat(),
                    "status": "running",
                    "nodes": [],
                    "edges": []
                }

        # Create deep copy for logging - capture full state
        input_state = {
            "error": state.get("error", ""),
            "messages": state.get("messages", []),
            "iterations": state.get("iterations", 0),
            "rag_context": state.get("rag_context", []),
            "generation": state.get("generation"),
            "error_summary": state.get("error_summary", "")
        }

        # Execute the node
        print(f"---Executing node: {node_name}---")
        result = node_func(state)
        print(f"---Node {node_name} completed---")

        # Log output - capture full state
        output_state = {
            "error": result.get("error", ""),
            "messages": result.get("messages", []),
            "iterations": result.get("iterations", 0),
            "rag_context": result.get("rag_context", []),
            "generation": result.get("generation"),
            "error_summary": result.get("error_summary", "")
        }

        # Record node execution
        node_execution = {
            "node_name": node_name,
            "input": input_state,
            "output": output_state,
            "timestamp": datetime.now().isoformat()
        }

        # Append nodes list with lock
        with workflow_runs_lock:
            workflow_runs[run_id]["nodes"].append(node_execution)

        return result
    return wrapped


# Node functions from test.py
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
    relative_segmens = retriever.invoke(query, k=2)

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
        "error": state["error"],
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
            "error": "yes", "rag_context": rag_context, "error_summary": error_summary.content}


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


# Build the workflow graph
workflow = StateGraph(GraphState)

# Add wrapped nodes
workflow.add_node("rag_retrieve", wrap_node("rag_retrieve", rag_retrieve))
workflow.add_node("generate", wrap_node("generate", generate))
workflow.add_node("check_code", wrap_node("check_code", code_check))
workflow.add_node("reflect", wrap_node("reflect", reflect))

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

graph = workflow.compile()

# Graph metadata
GRAPH_METADATA = {
    "name": "Code Assistant Agent",
    "description": "A LangGraph-based agent for code generation with RAG and self-reflection",
    "nodes": [
        {
            "id": "rag_retrieve",
            "label": "RAG Retrieve",
            "description": "Retrieves relevant documentation from vector store",
            "type": "retrieval"
        },
        {
            "id": "generate",
            "label": "Generate Code",
            "description": "Uses LLM to generate code solution based on RAG context",
            "type": "generation"
        },
        {
            "id": "check_code",
            "label": "Check Code",
            "description": "Validates generated code by executing imports and code block",
            "type": "validation"
        },
        {
            "id": "reflect",
            "label": "Reflect",
            "description": "Analyzes errors and summarizes for next iteration",
            "type": "reflection"
        }
    ],
    "edges": [
        {"from": "START", "to": "rag_retrieve", "type": "normal"},
        {"from": "rag_retrieve", "to": "generate", "type": "normal"},
        {"from": "generate", "to": "check_code", "type": "normal"},
        {"from": "check_code", "to": "reflect", "type": "conditional", "condition": "error == 'yes' and iterations < max_iterations"},
        {"from": "check_code", "to": "END", "type": "conditional", "condition": "error == 'no' or iterations >= max_iterations"},
        {"from": "reflect", "to": "rag_retrieve", "type": "normal"}
    ]
}


# API Models
class ExecuteRequest(BaseModel):
    question: str = Field(..., description="User's question or prompt")
    max_iterations: int = Field(default=3, description="Maximum number of iterations")


class ExecuteResponse(BaseModel):
    run_id: str
    status: str
    result: Dict[str, Any]


class NodeExecution(BaseModel):
    node_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    timestamp: str


class WorkflowRun(BaseModel):
    id: str
    started_at: str
    status: str
    nodes: List[NodeExecution]
    edges: List[Dict[str, Any]]


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main visualization page."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Workflow Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .node-card {
            transition: all 0.3s ease;
        }
        .node-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .node-input-output {
            background: #f8fafc;
            border-left: 3px solid #3b82f6;
        }
        .log-entry {
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .status-running { background: #fef3c7; border-color: #f59e0b; color: #92400e; }
        .status-completed { background: #d1fae5; border-color: #10b981; color: #065f46; }
        .status-error { background: #fee2e2; border-color: #ef4444; color: #991b1b; }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">🤖 Agent Workflow Visualizer</h1>
            <p class="text-gray-600">Visualize LangGraph agent execution with real-time node I/O tracking</p>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left Panel: Controls and Graph Overview -->
            <div class="lg:col-span-1 space-y-6">
                <!-- Execute Panel -->
                <div class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Execute Workflow</h2>
                    <textarea id="question" rows="3" placeholder="Enter your question..."
                        class="w-full border border-gray-300 rounded-lg p-3 mb-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500">How can I directly pass a string to a runnable?</textarea>
                    <div class="flex gap-2">
                        <button onclick="executeWorkflow()" id="executeBtn"
                            class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition">
                            ▶ Execute
                        </button>
                        <button onclick="clearResults()"
                            class="bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold py-2 px-4 rounded-lg transition">
                            Clear
                        </button>
                    </div>
                </div>

                <!-- Graph Overview -->
                <div class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Workflow Graph</h2>
                    <div id="graphOverview" class="space-y-2"></div>
                </div>

                <!-- Runs List -->
                <div class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Execution Runs</h2>
                    <div id="runsList" class="space-y-2 max-h-64 overflow-y-auto">
                        <p class="text-gray-500 text-sm">No runs yet</p>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Execution Details -->
            <div class="lg:col-span-2 space-y-6">
                <!-- Current Run Status -->
                <div id="runStatus" class="bg-white rounded-lg shadow p-6 hidden">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold text-gray-800">Execution Details</h2>
                        <span id="statusBadge" class="px-3 py-1 rounded-full text-sm font-medium"></span>
                    </div>
                    <div id="executionLog" class="space-y-4"></div>
                </div>

                <!-- Placeholder -->
                <div id="placeholder" class="bg-white rounded-lg shadow p-12 text-center">
                    <div class="text-gray-400 text-6xl mb-4">📊</div>
                    <p class="text-gray-500">Execute the workflow to see detailed node execution logs</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentRunId = null;
        let pollInterval = null;

        async function init() {
            await loadGraph();
            await loadRuns();
        }

        async function loadGraph() {
            const response = await fetch('/api/graph');
            const graph = await response.json();
            renderGraph(graph);
        }

        function renderGraph(graph) {
            const container = document.getElementById('graphOverview');
            container.innerHTML = `
                <div class="text-sm text-gray-600 mb-3">${graph.description}</div>
                <div class="space-y-2">
                    ${graph.nodes.map(node => `
                        <div class="node-card p-3 rounded-lg border-2 ${getNodeColor(node.type)}">
                            <div class="font-medium">${node.label}</div>
                            <div class="text-xs text-gray-500 mt-1">${node.description}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        function getNodeColor(type) {
            const colors = {
                'retrieval': 'border-purple-300 bg-purple-50',
                'generation': 'border-blue-300 bg-blue-50',
                'validation': 'border-green-300 bg-green-50',
                'reflection': 'border-orange-300 bg-orange-50'
            };
            return colors[type] || 'border-gray-300 bg-gray-50';
        }

        async function executeWorkflow() {
            const question = document.getElementById('question').value;
            const btn = document.getElementById('executeBtn');
            btn.disabled = true;
            btn.textContent = '⏳ Executing...';

            try {
                const response = await fetch('/api/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const result = await response.json();
                currentRunId = result.run_id;

                document.getElementById('placeholder').classList.add('hidden');
                document.getElementById('runStatus').classList.remove('hidden');

                startPolling();
                await loadRuns();
            } catch (error) {
                console.error('Error:', error);
            } finally {
                btn.disabled = false;
                btn.textContent = '▶ Execute';
            }
        }

        function startPolling() {
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(pollRun, 500);
        }

        function stopPolling() {
            if (pollInterval) clearInterval(pollInterval);
        }

        async function pollRun() {
            if (!currentRunId) return;

            try {
                const response = await fetch(`/api/runs/${currentRunId}`);
                const run = await response.json();

                document.getElementById('statusBadge').textContent = run.status;
                document.getElementById('statusBadge').className = `px-3 py-1 rounded-full text-sm font-medium status-${run.status}`;

                renderExecutionLog(run);

                if (run.status === 'completed' || run.status === 'error') {
                    stopPolling();
                }
            } catch (error) {
                console.error('Poll error:', error);
            }
        }

        function renderExecutionLog(run) {
            const container = document.getElementById('executionLog');
            container.innerHTML = `
                <div class="text-sm text-gray-500">Started: ${new Date(run.started_at).toLocaleString()}</div>
                <div class="mt-4 space-y-4">
                    ${run.nodes.map((node, index) => `
                        <div class="log-entry bg-gray-50 rounded-lg p-4 border border-gray-200">
                            <div class="flex items-center gap-3 mb-4">
                                <span class="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded">${node.node_name}</span>
                                <span class="text-gray-400 text-xs">${new Date(node.timestamp).toLocaleTimeString()}</span>
                            </div>
                            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                <div class="node-input-output rounded p-3">
                                    <div class="text-sm font-semibold text-gray-700 mb-3 border-b pb-2">INPUT</div>
                                    ${renderStateDetails(node.input)}
                                </div>
                                <div class="node-input-output rounded p-3">
                                    <div class="text-sm font-semibold text-gray-700 mb-3 border-b pb-2">OUTPUT</div>
                                    ${renderStateDetails(node.output)}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        function renderStateDetails(state) {
            const details = [];
            for (const [key, value] of Object.entries(state)) {
                const formattedValue = formatValue(value);
                details.push(`
                    <div class="mb-2">
                        <span class="text-xs font-bold text-blue-600 uppercase">${key}:</span>
                        <div class="mt-1 text-xs text-gray-700 bg-white rounded p-2 border border-gray-100">
                            ${formattedValue}
                        </div>
                    </div>
                `);
            }
            return details.join('');
        }

        function formatValue(value) {
            if (value === null || value === undefined) {
                return '<span class="italic text-gray-400">null</span>';
            }
            if (typeof value === 'string') {
                return `<span class="text-gray-700">"${escapeHtml(value)}"</span>`;
            }
            if (typeof value === 'number') {
                return `<span class="text-blue-600">"${value}"</span>`;
            }
            if (typeof value === 'boolean') {
                return `<span class="${value ? 'text-green-600' : 'text-red-600'}">${value}</span>`;
            }
            if (Array.isArray(value)) {
                if (value.length === 0) {
                    return '<span class="italic text-gray-400">[]</span>';
                }
                const items = value.map((item, idx) => {
                    return `<div class="ml-2 mt-1"><span class="text-gray-400">${idx}:</span> ${formatValue(item)}</div>`;
                }).join('');
                return `<div class="text-gray-400">[</div>${items}<div class="text-gray-400">]</div>`;
            }
            if (typeof value === 'object') {
                const entries = Object.entries(value).map(([k, v]) => {
                    return `<div class="ml-2 mt-1"><span class="text-blue-400">"${escapeHtml(k)}":</span> ${formatValue(v)}</div>`;
                }).join('');
                return `<div class="text-gray-400">{</div>${entries}<div class="text-gray-400">}</div>`;
            }
            return escapeHtml(String(value));
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatState(state) {
            return JSON.stringify(state, null, 2);
        }

        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const runs = await response.json();
                renderRuns(runs);
            } catch (error) {
                console.error('Error loading runs:', error);
            }
        }

        function renderRuns(runs) {
            const container = document.getElementById('runsList');
            if (runs.length === 0) {
                container.innerHTML = '<p class="text-gray-500 text-sm">No runs yet</p>';
                return;
            }

            container.innerHTML = runs.map(run => `
                <button onclick="viewRun('${run.id}')"
                    class="w-full text-left p-3 rounded-lg border-2 ${run.id === currentRunId ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'} transition">
                    <div class="flex justify-between items-center">
                        <span class="font-medium text-sm">${run.id}</span>
                        <span class="text-xs px-2 py-1 rounded status-${run.status}">${run.status}</span>
                    </div>
                    <div class="text-xs text-gray-500 mt-1">${run.nodes.length} nodes executed</div>
                </button>
            `).join('');
        }

        async function viewRun(runId) {
            currentRunId = runId;
            document.getElementById('placeholder').classList.add('hidden');
            document.getElementById('runStatus').classList.remove('hidden');
            await pollRun();
            await loadRuns();
        }

        function clearResults() {
            currentRunId = null;
            stopPolling();
            document.getElementById('placeholder').classList.remove('hidden');
            document.getElementById('runStatus').classList.add('hidden');
            document.getElementById('executionLog').innerHTML = '';
            loadRuns();
        }

        init();
    </script>
</body>
</html>
    """)


@app.get("/api/graph")
async def get_graph():
    """Get the workflow graph metadata."""
    return GRAPH_METADATA


@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_workflow(request: ExecuteRequest):
    """Execute the workflow and return the run ID."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # Initialize run entry immediately to avoid race condition
    with workflow_runs_lock:
        workflow_runs[run_id] = {
            "id": run_id,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "nodes": [],
            "edges": []
        }

    async def run_async():
        try:
            result = await asyncio.to_thread(
                graph.invoke,
                {
                    "messages": [("user", request.question)],
                    "iterations": 0,
                    "error": "",
                    "rag_context": [],
                    "error_summary": "",
                    "_run_id": run_id
                }
            )
            with workflow_runs_lock:
                workflow_runs[run_id]["status"] = "completed"
                workflow_runs[run_id]["result"] = result
                workflow_runs[run_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            with workflow_runs_lock:
                workflow_runs[run_id]["status"] = "error"
                workflow_runs[run_id]["error"] = str(e)
                workflow_runs[run_id]["completed_at"] = datetime.now().isoformat()

    # Run in background
    asyncio.create_task(run_async())

    return ExecuteResponse(
        run_id=run_id,
        status="running",
        result={}
    )


@app.get("/api/runs")
async def get_runs():
    """Get all workflow runs."""
    with workflow_runs_lock:
        return list(workflow_runs.values())


@app.get("/api/runs/{run_id}", response_model=WorkflowRun)
async def get_run(run_id: str):
    """Get a specific workflow run."""
    with workflow_runs_lock:
        if run_id not in workflow_runs:
            raise HTTPException(status_code=404, detail="Run not found")
        return workflow_runs[run_id]


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a workflow run."""
    with workflow_runs_lock:
        if run_id not in workflow_runs:
            raise HTTPException(status_code=404, detail="Run not found")
        del workflow_runs[run_id]
    return {"message": "Run deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
