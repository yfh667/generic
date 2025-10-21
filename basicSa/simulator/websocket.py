# websocket_manager.py
import asyncio
import json
import logging
from threading import Thread, Event
import websockets
from typing import Any, Set, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebSocketManager")


class WebSocketManager:
    def __init__(self):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running = Event()
        self.server_thread: Optional[Thread] = None
        self._stop_event = asyncio.Event()
        logger.info("WebSocket manager initialized")

    def start_server(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Start WebSocket server in a background thread."""
        if self._is_running.is_set():
            logger.warning("Server is already running")
            return

        def _server_main():
            self.server_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.server_loop)

            async def _run_server():
                async with websockets.serve(
                    self._handle_connection,
                    host,
                    port,
                    ping_interval=30,
                    ping_timeout=60
                ):
                    self._is_running.set()
                    logger.info(f"Server started on {host}:{port}")
                    await self._stop_event.wait()

            try:
                self.server_loop.run_until_complete(_run_server())
            except Exception as e:
                logger.error(f"Server error: {str(e)}", exc_info=True)
            finally:
                self._cleanup()
                logger.info("Server stopped")

        self.server_thread = Thread(target=_server_main, daemon=True)
        self.server_thread.start()

    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New connection from {client_addr}")
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            logger.info(f"Connection closed: {client_addr}")

    def broadcast(self, data: Any) -> None:
        """Broadcast data to all connected clients."""
        if not self._is_running.is_set():
            logger.warning("Cannot broadcast - server not running")
            return

        async def _async_broadcast():
            if not self.clients:
                logger.debug("No clients connected")
                return

            try:
                message = self._serialize_data(data)
                tasks = [client.send(message) for client in self.clients]
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.debug(f"Broadcasted to {len(self.clients)} clients")
            except Exception as e:
                logger.error(f"Broadcast failed: {str(e)}", exc_info=True)

        asyncio.run_coroutine_threadsafe(_async_broadcast(), self.server_loop)

    def _serialize_data(self, data: Any) -> str:
        """Serialize data to JSON string."""
        try:
            return json.dumps(data, default=self._json_serializer)
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error: {str(e)}")
            return json.dumps({"error": "Data serialization failed"})

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return str(obj)

    def stop(self) -> None:
        """Stop the WebSocket server gracefully."""
        if not self._is_running.is_set():
            return

        async def _stop_server():
            self._stop_event.set()

        if self.server_loop is not None:
            asyncio.run_coroutine_threadsafe(_stop_server(), self.server_loop)
            self.server_thread.join(timeout=5)
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_running.clear()
        self._stop_event.clear()
        self.clients.clear()
        if self.server_loop is not None:
            self.server_loop.close()
        self.server_loop = None

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running.is_set()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.stop()
