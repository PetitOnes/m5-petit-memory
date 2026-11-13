"""Configuration for Memory MCP Server."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class MemoryConfig:
    """Memory storage configuration."""

    db_path: str
    collection_name: str
    embedding_model: str = "intfloat/multilingual-e5-base"
    enable_bm25: bool = True
    embed_server: str | None = None  # 埋め込みを頼むサーバー(memory-embed-server)。無ければ自分でモデルを読む

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        default_path = str(Path.home() / ".claude" / "memories" / "memory.db")

        return cls(
            db_path=os.getenv("MEMORY_DB_PATH", default_path),
            collection_name=os.getenv("MEMORY_COLLECTION_NAME", "claude_memories"),
            embedding_model=os.getenv("MEMORY_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
            enable_bm25=os.getenv("MEMORY_ENABLE_BM25", "true").lower() != "false",
            embed_server=os.getenv("MEMORY_EMBED_SERVER") or None,
        )


@dataclass(frozen=True)
class SleepConfig:
    """Sleep (memory consolidation/forgetting) configuration."""

    min_age_days: int = 14  # 対象の最低経過日数
    # 統合と忘却は記憶を消すので、既定では動かさない(既定の整理は減衰だけ)。
    # multilingual-e5 の類似度は無関係な文どうしでも 0.85 前後になるので、
    # 統合を使うときも、ほぼ同じ文だけがまとまるように基準を高くしておく
    merge: bool = False  # 似た記憶を 1 件にまとめ、元を消す
    forget: bool = False  # 重要度 1 の古い記憶を消す
    similarity_threshold: float = 0.95  # 圧縮時の類似度閾値
    decay_retention_threshold: float = 0.4  # この保持スコア以下で減衰
    forget_min_age_days: int = 14  # 忘却の最低経過日数
    forget_max_access: int = 3  # 忘却対象の最大アクセス回数
    protected_importance: int = 4  # この値以上は絶対保護
    protected_emotions: tuple[str, ...] = ("happy", "moved", "excited", "surprised")

    @classmethod
    def from_env(cls) -> "SleepConfig":
        """環境変数から。MEMORY_SLEEP_MERGE / MEMORY_SLEEP_FORGET が true のときだけ、消す処理を動かす."""

        def on(name: str) -> bool:
            return os.getenv(name, "false").lower() in ("1", "true", "yes")

        return cls(merge=on("MEMORY_SLEEP_MERGE"), forget=on("MEMORY_SLEEP_FORGET"))


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "memory-mcp"
    version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "memory-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.1.0"),
        )
