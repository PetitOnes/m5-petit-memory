"""カスタム埋め込み関数モジュール。

intfloat/multilingual-e5-base の SentenceTransformer ラッパー。
e5 モデルはクエリと文書で異なるプレフィックスが必要。
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

REMOTE_TIMEOUT_S = 60  # 埋め込みのサーバーを待つ長さ(サーバーがモデルを読んでいる最中のことがある)


class E5EmbeddingFunction:
    """intfloat/multilingual-e5-base 用埋め込み関数。

    e5 モデルの仕様:
    - 文書（passage）保存時: "passage: {text}" としてエンコード
    - クエリ検索時: "query: {text}" としてエンコード

    server_url を渡すと、埋め込みは memory-embed-server に頼む(このプロセスはモデルを読まない)。
    つながらない・モデル名が違う・返事がおかしいときは、自分でモデルを読んで続ける。
    一度自分で読んだら、もうサーバーには頼まない(読んだぶんのメモリは戻らないので)。

    Args:
        model_name: SentenceTransformer モデル名
        server_url: 埋め込みのサーバー(例 http://127.0.0.1:18766)。None なら自分で読む
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        server_url: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy load; actual type is SentenceTransformer
        self._server_url = server_url.rstrip("/") if server_url else None

    def _remote(self, kind: str, texts: list[str]) -> list[list[float]] | None:
        """埋め込みのサーバーに頼む。使えなければ None(呼ぶ側が自分で読む)."""
        if not self._server_url or self._model is not None:
            return None
        body = json.dumps({"kind": kind, "texts": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._server_url}/embed", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=REMOTE_TIMEOUT_S) as res:
                data = json.loads(res.read())
            vectors = data["vectors"]
            if data.get("model") != self._model_name:
                raise ValueError(f"model mismatch: {data.get('model')}")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise ValueError("vector count")
            return vectors
        except (OSError, ValueError, KeyError, TypeError) as e:  # URLError・タイムアウトは OSError
            logger.warning(
                "埋め込みのサーバー %s が使えない(%s)。このプロセスでモデルを読む", self._server_url, e
            )
            return None

    def _load_model(self) -> None:
        """モデルを遅延ロード。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                logger.info("E5EmbeddingFunction: loaded model %s", self._model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers が必要です。"
                    "`uv add sentence-transformers` を実行してください。"
                ) from e

    def __call__(self, input: list[str]) -> list[list[float]]:
        """文書保存用埋め込み（passage: プレフィックス）。

        Args:
            input: エンコードするテキストのリスト

        Returns:
            埋め込みベクトルのリスト（各要素は float のリスト）
        """
        remote = self._remote("passage", input)
        if remote is not None:
            return remote
        self._load_model()
        prefixed = [f"passage: {doc}" for doc in input]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        """クエリ検索用埋め込み（query: プレフィックス）。

        Args:
            texts: クエリテキストのリスト

        Returns:
            埋め込みベクトルのリスト
        """
        remote = self._remote("query", texts)
        if remote is not None:
            return remote
        self._load_model()
        prefixed = [f"query: {t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
