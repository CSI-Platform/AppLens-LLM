from __future__ import annotations

import argparse
import copy
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


_THINK_BLOCK_RE = re.compile(r"(?is)<think\b[^>]*>.*?</think>\s*")


def strip_reasoning_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


def normalize_chat_completion_response(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    for choice in normalized.get("choices", []):
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = strip_reasoning_blocks(content)
    return normalized


def make_proxy_handler(upstream_base_url: str) -> type[BaseHTTPRequestHandler]:
    upstream = upstream_base_url.rstrip("/") + "/"

    class LlamaCppLmEvalProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            self._proxy()

        def log_message(self, format: str, *args: object) -> None:
            return

        def _proxy(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            target = urljoin(upstream, self.path.lstrip("/"))
            headers = {
                "Content-Type": self.headers.get("Content-Type", "application/json"),
            }
            request = Request(target, data=body or None, headers=headers, method=self.command)
            try:
                with urlopen(request, timeout=600) as response:  # noqa: S310
                    response_body = response.read()
                    status = response.status
                    response_headers = dict(response.headers)
            except HTTPError as error:
                response_body = error.read()
                status = error.code
                response_headers = dict(error.headers)

            if self.path.rstrip("/") == "/v1/chat/completions" and response_body:
                try:
                    payload = json.loads(response_body.decode("utf-8"))
                    response_body = json.dumps(
                        normalize_chat_completion_response(payload),
                        ensure_ascii=False,
                    ).encode("utf-8")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            self.send_response(status)
            self.send_header("Content-Type", response_headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    return LlamaCppLmEvalProxyHandler


def serve_proxy(*, listen_host: str, listen_port: int, upstream_base_url: str) -> None:
    handler = make_proxy_handler(upstream_base_url)
    server = ThreadingHTTPServer((listen_host, listen_port), handler)
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize llama.cpp chat responses for lm-eval API runs.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18081)
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:18080")
    args = parser.parse_args(argv)
    serve_proxy(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_base_url=args.upstream_base_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
