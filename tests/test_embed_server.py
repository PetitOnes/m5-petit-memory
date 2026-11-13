"""埋め込みのサーバー(memory-embed-server)と、それに頼む側のテスト。モデルは読まず、にせの埋め込みで確かめる."""

import threading

import pytest

from memory_mcp.embed_server import make_server
from memory_mcp.embedding import E5EmbeddingFunction

MODEL = "test-model"


class FakeEmbedding:
    """文の長さだけを見る、にせの埋め込み。passage と query を見分けられるようにしてある."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("passage", texts))
        return [[float(len(t)), 0.0] for t in texts]

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("query", texts))
        return [[float(len(t)), 1.0] for t in texts]


@pytest.fixture
def embed_server():
    fake = FakeEmbedding()
    server = make_server(fake, MODEL, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", fake
    server.shutdown()
    server.server_close()


def _no_local_model(fn: E5EmbeddingFunction) -> None:
    def fail() -> None:
        raise AssertionError("このプロセスでモデルを読もうとした")

    fn._load_model = fail  # type: ignore[method-assign]


def test_client_uses_server(embed_server):
    """サーバーがあれば、頼む側はモデルを読まずにベクトルを受け取る."""
    url, fake = embed_server
    fn = E5EmbeddingFunction(MODEL, server_url=url)
    _no_local_model(fn)

    assert fn(["あいう", "えお"]) == [[3.0, 0.0], [2.0, 0.0]]
    assert fn.encode_query(["かきくけ"]) == [[4.0, 1.0]]
    assert fake.calls == [("passage", ["あいう", "えお"]), ("query", ["かきくけ"])]


class _LocalModel:
    def encode(self, texts, **kwargs):
        import numpy as np

        return np.array([[9.0, 9.0] for _ in texts], dtype=np.float32)


def _with_local_model(fn: E5EmbeddingFunction) -> list[int]:
    loads: list[int] = []

    def load() -> None:
        loads.append(1)
        fn._model = _LocalModel()

    fn._load_model = load  # type: ignore[method-assign]
    return loads


def test_client_falls_back_when_server_is_down():
    """サーバーにつながらなければ、自分でモデルを読んで続ける."""
    fn = E5EmbeddingFunction(MODEL, server_url="http://127.0.0.1:9")  # だれも待っていないポート
    loads = _with_local_model(fn)

    assert fn(["あ"]) == [[9.0, 9.0]]
    assert loads == [1]


def test_client_falls_back_on_model_mismatch(embed_server):
    """サーバーのモデルが違えば、ベクトルは使わず、自分でモデルを読む."""
    url, fake = embed_server
    fn = E5EmbeddingFunction("another-model", server_url=url)
    loads = _with_local_model(fn)

    assert fn.encode_query(["あ"]) == [[9.0, 9.0]]
    assert loads == [1]


def test_client_stops_asking_after_loading_locally(embed_server):
    """一度自分でモデルを読んだら、もうサーバーには頼まない."""
    url, fake = embed_server
    fn = E5EmbeddingFunction(MODEL, server_url=url)
    fn._model = _LocalModel()

    assert fn(["あ"]) == [[9.0, 9.0]]
    assert fake.calls == []


def test_server_rejects_bad_request(embed_server):
    import json
    import urllib.error
    import urllib.request

    url, _ = embed_server
    req = urllib.request.Request(f"{url}/embed", data=json.dumps({"kind": "other", "texts": ["a"]}).encode())
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
