"""
cache.py
--------
Optional disk-based embedding cache to avoid re-computing embeddings
for texts that have already been processed.

Uses a simple ``shelve`` database keyed by ``(model_name, text)``.
"""

from __future__ import annotations

import hashlib
import logging
import shelve
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Transparent disk cache for embedding vectors.

    Wrap any :class:`~sentence_embedder.SentenceEmbedder` with this class
    to skip re-embedding texts that were already processed.

    Args:
        embedder: A :class:`~sentence_embedder.SentenceEmbedder` instance.
        cache_dir (str | Path): Directory where the cache database is stored.
            Created automatically if it does not exist.

    Example:
        >>> from sentence_embedder import SentenceEmbedder
        >>> from sentence_embedder.cache import EmbeddingCache
        >>> base = SentenceEmbedder()
        >>> cached = EmbeddingCache(base, cache_dir=".cache/embeddings")
        >>> vec = cached.embed("Hello world")  # computed and stored
        >>> vec = cached.embed("Hello world")  # retrieved from cache
    """

    def __init__(self, embedder, cache_dir: str | Path = ".cache/embeddings") -> None:
        self._embedder = embedder
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = str(cache_path / "embeddings")
        logger.info("EmbeddingCache initialised at '%s'.", self._db_path)

    # ------------------------------------------------------------------
    # Public API (mirrors SentenceEmbedder)
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text, using the cache when available.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Embedding vector.
        """
        key = self._make_key(text)
        with shelve.open(self._db_path) as db:
            if key in db:
                logger.debug("Cache hit for text hash '%s'.", key)
                return db[key]

        vec = self._embedder.embed(text)
        with shelve.open(self._db_path) as db:
            db[key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts, only calling the model for cache misses.

        Args:
            texts (List[str]): Input texts.

        Returns:
            np.ndarray: 2-D array of shape ``(len(texts), dim)``.
        """
        keys = [self._make_key(t) for t in texts]
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices: List[int] = []

        with shelve.open(self._db_path) as db:
            for i, key in enumerate(keys):
                if key in db:
                    results[i] = db[key]
                else:
                    missing_indices.append(i)

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_vecs = self._embedder.embed_batch(missing_texts)
            with shelve.open(self._db_path) as db:
                for idx, vec in zip(missing_indices, new_vecs):
                    db[keys[idx]] = vec
                    results[idx] = vec

        logger.debug(
            "embed_batch: %d cached, %d computed.",
            len(texts) - len(missing_indices),
            len(missing_indices),
        )
        return np.stack(results)

    def clear(self) -> None:
        """Delete all cached embeddings."""
        import os
        for suffix in ["", ".db", ".dir", ".bak", ".dat"]:
            path = Path(self._db_path + suffix)
            if path.exists():
                os.remove(path)
        logger.info("Embedding cache cleared.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_key(self, text: str) -> str:
        payload = f"{self._embedder.model_name}::{text}"
        return hashlib.sha256(payload.encode()).hexdigest()
