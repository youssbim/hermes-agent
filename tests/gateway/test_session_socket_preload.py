"""The /api/ws handler is imported off the loop before READY, and a failed or hung import never holds READY."""
import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from gateway import run_api


@pytest.fixture
def unloaded(monkeypatch):
    """``tui_gateway.ws`` absent from sys.modules; record which thread imports it."""
    import importlib
    monkeypatch.delitem(sys.modules, "tui_gateway.ws", raising=False)
    seen = {}
    real = importlib.import_module

    def spy(name, *args, **kwargs):
        if name == "tui_gateway.ws":
            seen["thread"] = threading.current_thread()
            return SimpleNamespace(handle_ws=None)
        return real(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", spy)
    return seen


@pytest.mark.asyncio
async def test_preload_imports_off_the_event_loop(unloaded):
    await run_api.await_session_socket_preload(run_api.preload_session_socket())
    assert unloaded["thread"] is not threading.main_thread()


@pytest.mark.asyncio
async def test_failed_preload_never_fails_startup(caplog):
    async def boom():
        raise ImportError("broken handler")
    await run_api.await_session_socket_preload(asyncio.ensure_future(boom()))
    assert "loads on first attach" in caplog.text


@pytest.mark.asyncio
async def test_a_hung_preload_never_holds_ready(caplog):
    hang = asyncio.ensure_future(asyncio.Event().wait())
    await run_api.await_session_socket_preload(hang, timeout=0.05)
    assert "publishing READY" in caplog.text and not hang.cancelled()
    hang.cancel()


@pytest.mark.asyncio
async def test_an_attach_never_imports_the_handler_on_the_event_loop(unloaded):
    """An import still running past READY must not block the loop for the first attach."""
    await run_api.session_socket_handler()
    assert unloaded["thread"] is not threading.main_thread()


@pytest.mark.asyncio
async def test_a_preload_never_awaited_logs_no_unretrieved_exception(monkeypatch):
    """A startup that raises before awaiting the preload: its failure is not an asyncio warning."""
    import gc
    import importlib

    def boom(name, *args, **kwargs):
        raise ImportError("broken handler")
    monkeypatch.delitem(sys.modules, "tui_gateway.ws", raising=False)
    monkeypatch.setattr(importlib, "import_module", boom)
    seen = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "call_exception_handler", seen.append)
    task = run_api.preload_session_socket()
    assert task is not None
    await asyncio.wait({task})
    del task
    gc.collect()
    assert not [c for c in seen if "never retrieved" in c.get("message", "")]


@pytest.mark.asyncio
async def test_an_attach_counts_as_a_client_while_its_handler_is_still_loading(tmp_path, monkeypatch):
    """The ticket is redeemed before the handler fetch yields: an idle check in that gap must still
    see this client, or an idle-exit gateway shuts down under its first connection."""
    from gateway import run_idle_exit
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_server as web
    from hermes_cli import web_server_chat

    tickets = TicketStore('owner', [str(tmp_path)])
    ticket = tickets.mint(profile_id=str(tmp_path), subject='human', purpose='worker-adoption')
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / 'state.db'), profile_id=str(tmp_path), runner=None)
    runner = SimpleNamespace(_draining=False, session_runtime_descriptor={'state': 'ready'},
                             session_authority=authority, session_ticket_store=tickets)
    seen_while_loading = []

    async def loading_handler():
        seen_while_loading.append(run_idle_exit._client_busy(runner, window=0.0))
        async def handle(ws, **kwargs):
            await ws.close()
        return handle
    monkeypatch.setattr(run_api, 'session_socket_handler', loading_handler)
    monkeypatch.setattr(web, '_DASHBOARD_EMBEDDED_CHAT_ENABLED', True)
    monkeypatch.setattr(web_server_chat, '_ws_request_is_allowed', lambda ws: True)

    async def fallback(*args):
        pytest.fail('a native ticket must not enter the legacy fallback')
    incoming = iter([{'type': 'websocket.connect'}])
    async def receive(): return next(incoming)
    async def send(value): pass
    scope = {'type': 'websocket', 'path': '/api/ws', 'scheme': 'ws', 'query_string': b'',
             'client': ('127.0.0.1', 1234), 'server': ('127.0.0.1', 9999),
             'headers': [(b'sec-websocket-protocol', ('hermes-gateway-v1, hermes-gateway-ticket.' + ticket).encode())],
             'subprotocols': ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket]}
    await GatewayRuntimeAPI(fallback, runner, web.app)(scope, receive, send)
    assert seen_while_loading == ['1 client(s) attached']
