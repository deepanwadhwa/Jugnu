#!/usr/bin/env python3
"""tests/test_gateway_chat_tools.py — chat's tool round-trip through the gateway.

test_gateway_web.py only checks classify_reply/followup_payload/etc. as pure
functions; nothing boots chat_once/chat_stream against a real backend and
proves the tool actually gets dispatched. That matters here specifically
because the tool-execution refactor (chat now calls samosa_tools.execute_tool
via the shared registry, ctx=None, instead of the gateway's own now-deleted
execute_tool) touches exactly this wiring. Boots a fake model backend (returns
a tool call, then a final answer) and the real gateway Handler, and asserts the
web_search tool was actually invoked and its result made it back to the model.
"""

import http.client
import importlib.util
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock


class FakeBackend(BaseHTTPRequestHandler):
    """Stands in for qwen36b/llama-server: first turn asks for a tool, second
    turn (once it sees the SAMOSA_TOOL_RESULT) answers in plain text."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"ok": True})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        # The system prompt's own instructional text mentions the string
        # "SAMOSA_TOOL_RESULT" (explaining the protocol to the model), so a
        # bare substring check over every message matches on round 1 too.
        # Only a real followup carries it as the start of a user turn.
        saw_tool_result = any(
            m.get("role") == "user" and str(m.get("content", "")).startswith("SAMOSA_TOOL_RESULT ")
            for m in messages)
        if saw_tool_result:
            content = "Final answer using search results."
        else:
            content = '{"samosa_tool":"web_search","query":"pdf reading in samosa"}'
        self._json(200, {"choices": [{"message": {"content": content}}]})

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def load_gateway(home, backend_port):
    os.environ.update({
        "SAMOSA_HOME": home,
        "SAMOSA_BACKEND_PORT": str(backend_port),
        "SAMOSA_APP_HTML": str(Path(home) / "app.html"),
        "SAMOSA_APP_LOGO": str(Path(home) / "logo.png"),
        "SAMOSA_QWEN_ENGINE": str(Path(home) / "qwen36b"),
        "SAMOSA_QWEN_MODEL": str(Path(home) / "model"),
        "SAMOSA_TOKENIZER": str(Path(home) / "tokenizer.json"),
    })
    spec = importlib.util.spec_from_file_location(
        "samosa_gateway", Path(__file__).parents[1] / "tools/samosa_gateway.py")
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    return gateway


def main():
    backend = HTTPServer(("127.0.0.1", 0), FakeBackend)
    backend_port = backend.server_address[1]
    threading.Thread(target=backend.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory() as home:
        gateway = load_gateway(home, backend_port)

        search_calls = []

        def fake_search_web(query):
            search_calls.append(query)
            return [{"title": "Samosa Jobs docs", "url": "https://example.test/jobs",
                     "description": "how jobs read files"}]

        def fake_readable_page(url):
            return {"title": "Samosa Jobs docs", "url": url, "text": "PDF reading via samosa-extract.",
                    "truncated": False}

        with mock.patch.object(gateway.Supervisor, "ready", return_value=True), \
             mock.patch.object(gateway, "search_web", side_effect=fake_search_web), \
             mock.patch.object(gateway, "readable_page", side_effect=fake_readable_page):

            server = gateway.GatewayServer(("127.0.0.1", 0), gateway.Handler)
            server.handle_error = lambda request, client_address: None
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                body = json.dumps({"messages": [{"role": "user", "content": "how do I read a pdf in samosa?"}],
                                   "stream": False}).encode()
                conn.request("POST", "/v1/chat/completions", body=body,
                             headers={"Content-Type": "application/json",
                                      "Content-Length": str(len(body))})
                resp = conn.getresponse()
                data = json.loads(resp.read())
                conn.close()
            finally:
                server.shutdown()
                server.server_close()
                backend.shutdown()
                backend.server_close()

        assert resp.status == 200, f"chat/completions -> HTTP {resp.status}: {data}"
        assert search_calls == ["pdf reading in samosa"], \
            f"web_search was not dispatched via samosa_tools.execute_tool: {search_calls}"
        final_content = data["choices"][0]["message"]["content"]
        assert final_content == "Final answer using search results.", final_content
        print("test_gateway_chat_tools: OK "
              "(tool call dispatched through samosa_tools.execute_tool, round-trip completed)")


if __name__ == "__main__":
    main()
