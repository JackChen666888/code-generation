"""
Web Server for Code Assistant Visualization

This module provides a web interface to visualize the LangGraph workflow
and display input/output for each stage without modifying the original code.
"""

import sys
import os
import json
import uuid
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio

# Import the original CodeAssistant
from api import CodeAssistant
from config.configs import load_key


class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str
    api_key: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    success: bool
    message: str
    result: Optional[dict] = None


app = FastAPI(title="Code Assistant Visualization")


# Store active WebSocket connections
active_connections = {}

# Store message queues for SSE (async queues)
message_queues = {}
queues_lock = asyncio.Lock()


async def get_queue(session_id: str) -> asyncio.Queue:
    """Get or create an async message queue for a session."""
    async with queues_lock:
        if session_id not in message_queues:
            message_queues[session_id] = asyncio.Queue()
        return message_queues[session_id]


async def cleanup_queue(session_id: str):
    """Clean up a session's message queue."""
    async with queues_lock:
        if session_id in message_queues:
            del message_queues[session_id]


@app.get("/")
async def get_root():
    """Serve the main HTML page."""
    html_file = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections[session_id] = websocket

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back if needed
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]


async def send_update(session_id: str, update: dict):
    """Send an update to the connected client."""
    update_json = json.dumps(update, ensure_ascii=False)

    # Send to WebSocket if connected
    if session_id in active_connections:
        try:
            await active_connections[session_id].send_text(update_json)
        except Exception as e:
            print(f"Error sending WebSocket update: {e}")

    # Send to SSE queue
    async with queues_lock:
        if session_id in message_queues:
            try:
                await message_queues[session_id].put(update_json)
            except Exception as e:
                print(f"Error sending SSE update: {e}")


class TracedCodeAssistant(CodeAssistant):
    """
    Wrapped CodeAssistant that traces and reports each step.
    """

    def __init__(self, session_id: str, api_key: str = None, reset_index: bool = True):
        super().__init__(api_key=api_key, reset_index=reset_index)
        self.session_id = session_id
        self.trace_log = []

    async def _send_trace(self, node_name: str, step_type: str, data: dict):
        """Send a trace update to the client."""
        update = {
            "type": "trace",
            "node": node_name,
            "step_type": step_type,  # "input" or "output"
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        await send_update(self.session_id, update)

    async def _send_stage_start(self, stage_name: str, input_data: dict):
        """Send stage start notification."""
        update = {
            "type": "stage_start",
            "stage": stage_name,
            "input": self._sanitize_state(input_data),
            "timestamp": datetime.now().isoformat()
        }
        await send_update(self.session_id, update)

    async def _send_stage_end(self, stage_name: str, output_data: dict):
        """Send stage end notification."""
        update = {
            "type": "stage_end",
            "stage": stage_name,
            "output": self._sanitize_state(output_data),
            "timestamp": datetime.now().isoformat()
        }
        await send_update(self.session_id, update)

    def _sanitize_state(self, state: dict) -> dict:
        """Sanitize state for JSON serialization."""
        result = {}
        for key, value in state.items():
            if key == "generation":
                if hasattr(value, '__dict__'):
                    result[key] = {
                        "prefix": getattr(value, "prefix", ""),
                        "imports": getattr(value, "imports", ""),
                        "code": getattr(value, "code", "")
                    }
                else:
                    result[key] = str(value)
            elif key == "messages":
                result[key] = [
                    {"role": msg[0], "content": msg[1] if len(msg) > 1 else ""}
                    for msg in value
                ]
            elif key == "rag_context":
                result[key] = value  # Already a list of strings
            else:
                result[key] = value
        return result

    async def traced_query(self, question: str) -> dict:
        """
        Query the Code Assistant with tracing of each step.
        """
        self._initialize()

        # Send query start
        await send_update(self.session_id, {
            "type": "query_start",
            "question": question,
            "timestamp": datetime.now().isoformat()
        })

        # Create initial state
        initial_state = {
            "messages": [("user", question)],
            "iterations": 0,
            "error": "",
            "rag_context": [],
            "error_summary": ''
        }

        # Execute with tracing
        try:
            # Get the graph's compiled structure
            graph_dict = self.graph.get_graph().print_ascii()

            # Manually execute the workflow with tracing
            result = await self._execute_with_tracing(initial_state)

            # Send query complete
            await send_update(self.session_id, {
                "type": "query_complete",
                "result": self._sanitize_state(result),
                "timestamp": datetime.now().isoformat()
            })

            return result
        except Exception as e:
            await send_update(self.session_id, {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise

    async def _execute_with_tracing(self, initial_state: dict) -> dict:
        """
        Execute the workflow manually with tracing at each step.
        """
        from graph.nodes import rag_retrieve, generate, code_check, reflect
        from graph.edges import decide_to_finish

        state = initial_state.copy()
        max_iterations = 3  # Default max iterations

        while state["iterations"] < max_iterations and state["error"] != "no":
            # Stage 1: RAG Retrieve
            await self._send_stage_start("rag_retrieve", state)
            state = rag_retrieve(state, self.retriever)
            await self._send_stage_end("rag_retrieve", state)

            # Stage 2: Generate
            await self._send_stage_start("generate", state)
            state = generate(state, self.code_gen_chain)
            await self._send_stage_end("generate", state)

            # Stage 3: Check Code
            await self._send_stage_start("check_code", state)
            state = code_check(state)
            await self._send_stage_end("check_code", state)

            # Check if should finish
            decision = decide_to_finish(state)
            if decision == "end":
                break

            # Stage 4: Reflect (only if there's an error)
            if decision == "reflect":
                await self._send_stage_start("reflect", state)
                state = reflect(state, self.llm)
                await self._send_stage_end("reflect", state)

        return state





@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """Query the Code Assistant."""
    # Generate session ID if not provided
    session_id = str(uuid.uuid4())

    # Initialize queue for this session immediately
    await get_queue(session_id)

    try:
        # Create traced assistant
        assistant = TracedCodeAssistant(
            session_id=session_id,
            api_key=request.api_key,
            reset_index=False  # Don't reset index for queries
        )

        # Run the query in the background
        asyncio.create_task(assistant.traced_query(request.question))

        return QueryResponse(
            success=True,
            message="Query started. Connect to WebSocket for real-time updates.",
            result={"session_id": session_id}
        )
    except Exception as e:
        return QueryResponse(
            success=False,
            message=f"Error starting query: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_connections": len(active_connections)}


@app.get("/api/monitor/{session_id}")
async def monitor_session(session_id: str):
    """SSE endpoint for monitoring session updates."""
    async def event_generator():
        # Initialize queue for this session
        msg_queue = await get_queue(session_id)

        try:
            while True:
                # Wait for new messages with timeout
                try:
                    msg = await asyncio.wait_for(msg_queue.get(), timeout=30.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            await cleanup_queue(session_id)
        except Exception as e:
            print(f"SSE Error: {e}")
            await cleanup_queue(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
