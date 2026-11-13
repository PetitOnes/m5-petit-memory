"""埋め込みだけを受け持つ常駐プロセス(memory-embed-server)。

記憶のサーバー(memory-mcp)は子ごと・セッションごとに別々に立ち、それぞれが埋め込みのモデルを読むと
1 つにつき 1GB を超える。モデルをここ 1 つにまとめ、各サーバーは文を送ってベクトルを受け取る。

- 127.0.0.1 だけで待つ(記憶の文が流れるので、外には出さない)
- GET  /health → {"model": モデル名, "loaded": 読み込み済みか}
- POST /embed  {"kind": "passage" | "query", "texts": [...]} → {"model": モデル名, "vectors": [[...], ...]}

使う側は環境変数 MEMORY_EMBED_SERVER(例 http://127.0.0.1:18766)。つながらなければ、自分でモデルを読む。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .embedding import E5EmbeddingFunction

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18766
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_TEXTS = 256


def make_server(
    embedding_fn: Any, model_name: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """埋め込みの関数(passage は呼び出し、query は encode_query)を HTTP で出すサーバーを作る."""
    lock = threading.Lock()  # モデルは 1 つ。同時に来たら順番に

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"model": model_name, "loaded": getattr(embedding_fn, "_model", True) is not None})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/embed":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("body size")
                req = json.loads(self.rfile.read(length))
                kind, texts = req["kind"], req["texts"]
                if kind not in ("passage", "query") or not isinstance(texts, list) or len(texts) > MAX_TEXTS:
                    raise ValueError("kind / texts")
                if not all(isinstance(t, str) for t in texts):
                    raise ValueError("texts")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {e}"})
                return
            try:
                with lock:
                    vectors = embedding_fn(texts) if kind == "passage" else embedding_fn.encode_query(texts)
            except Exception as e:  # モデルの失敗は、使う側が自分で読む道に回れるように 500 で返す
                logger.exception("embed failed")
                self._send(500, {"error": type(e).__name__})
                return
            self._send(200, {"model": model_name, "vectors": vectors})

        def log_message(self, format: str, *args: Any) -> None:  # 記憶の文も件数も、既定ではログに出さない
            logger.debug("%s " + format, self.address_string(), *args)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="記憶の埋め込みを 1 つのプロセスにまとめるサーバー")
    parser.add_argument("--host", default=os.getenv("MEMORY_EMBED_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("MEMORY_EMBED_PORT", DEFAULT_PORT)))
    parser.add_argument("--model", default=os.getenv("MEMORY_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    parser.add_argument("--lazy", action="store_true", help="モデルを、最初に頼まれたときに読む(既定は起動のとき)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    embedding_fn = E5EmbeddingFunction(args.model)  # server_url は渡さない(ここが読む側)
    if not args.lazy:
        embedding_fn._load_model()
    server = make_server(embedding_fn, args.model, args.host, args.port)
    logger.info("memory-embed-server: %s を http://%s:%d で待つ", args.model, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
