import asyncio
import json
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from poseblend.schema.run_data import RunData

GUI_DIST_DIR = Path(__file__).resolve().parent.parent.parent / "gui" / "dist"


class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self._connections.remove(websocket)

    async def broadcast(self, message: str):
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001
                self._connections.remove(ws)


def create_app(run_data: RunData, run_dir: Path) -> tuple[FastAPI, Callable[[], None]]:
    app = FastAPI()
    manager = ConnectionManager()
    loop: asyncio.AbstractEventLoop | None = None

    def _serialize() -> str:
        return json.dumps(run_data.model_dump(mode="json"), default=str)

    def broadcast_update():
        nonlocal loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        message = _serialize()
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), loop)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal loop
        loop = asyncio.get_running_loop()
        await manager.connect(websocket)
        try:
            await websocket.send_text(_serialize())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.get("/files/{file_path:path}")
    async def serve_file(file_path: str):
        full_path = run_dir / file_path
        if not full_path.exists() or not full_path.is_file():
            return Response(status_code=404, content="File not found")
        return FileResponse(full_path)

    if GUI_DIST_DIR.exists():
        @app.get("/")
        async def serve_index():
            return FileResponse(GUI_DIST_DIR / "index.html")

        app.mount("/assets", StaticFiles(directory=GUI_DIST_DIR / "assets"), name="assets")

        @app.get("/{catch_all:path}")
        async def spa_fallback(catch_all: str):
            file_path = GUI_DIST_DIR / catch_all
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(GUI_DIST_DIR / "index.html")

    return app, broadcast_update
