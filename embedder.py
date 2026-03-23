"""
embedder.py
-----------
Core module for generating sentence/semantic embeddings.
Supports multiple backends: SentenceTransformers and OpenAI.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """
    A unified interface for generating sentence/semantic embeddings.

    Supports the following backends:
        - ``"sentencetransformers"`` — Local HuggingFace models (default, offline-friendly)
        - ``"openai"`` — OpenAI text-embedding-* API models

    Args:
        backend (str): Which embedding backend to use. One of
            ``"sentencetransformers"`` or ``"openai"``. Defaults to
            ``"sentencetransformers"``.
        model_name (str): Model identifier.
            - For SentenceTransformers: any model on HuggingFace Hub,
              e.g. ``"all-MiniLM-L6-v2"`` (default).
            - For OpenAI: e.g. ``"text-embedding-3-small"``.
        openai_api_key (str | None): API key for OpenAI backend. If ``None``,
            falls back to the ``OPENAI_API_KEY`` environment variable.
        device (str): Torch device for SentenceTransformers, e.g. ``"cpu"``
            or ``"cuda"``. Ignored for OpenAI backend.

    Example:
        >>> embedder = SentenceEmbedder()
        >>> vec = embedder.embed("Hello world")
        >>> vec.shape
        (384,)
    """

    SUPPORTED_BACKENDS = {"sentencetransformers", "openai"}

    def __init__(
        self,
        backend: str = "sentencetransformers",
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str | None = None,
        device: str = "cpu",
    ) -> None:
        backend = backend.lower()
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Choose from: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.model_name = model_name
        self.device = device
        self._model = None
        self._openai_client = None

        if backend == "openai":
            self._init_openai(openai_api_key)
        else:
            self._init_sentence_transformers()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading SentenceTransformer model '%s'…", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Model loaded.")

    def _init_openai(self, api_key: str | None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            ) from exc

        import os

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OpenAI API key not provided. Pass openai_api_key= or set "
                "the OPENAI_API_KEY environment variable."
            )
        self._openai_client = OpenAI(api_key=key)
        logger.info("OpenAI client initialised with model '%s'.", self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single sentence/document.

        Args:
            text (str): Input text to embed.

        Returns:
            np.ndarray: 1-D float32 embedding vector.

        Raises:
            ValueError: If ``text`` is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of sentences/documents.

        Args:
            texts (List[str]): List of input strings. Empty strings are
                rejected.

        Returns:
            np.ndarray: 2-D float32 array of shape ``(len(texts), dim)``.

        Raises:
            ValueError: If ``texts`` is empty or contains blank strings.
        """
        if not texts:
            raise ValueError("texts list must not be empty.")
        if any(not t or not t.strip() for t in texts):
            raise ValueError("texts list must not contain empty strings.")

        if self.backend == "sentencetransformers":
            return self._embed_st(texts)
        return self._embed_openai(texts)

    def similarity(self, a: str, b: str) -> float:
        """
        Compute cosine similarity between two sentences.

        Args:
            a (str): First sentence.
            b (str): Second sentence.

        Returns:
            float: Cosine similarity in [-1, 1].
        """
        vecs = self.embed_batch([a, b])
        return float(_cosine_similarity(vecs[0], vecs[1]))

    def most_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
    ) -> List[tuple[str, float]]:
        """
        Return the ``top_k`` most semantically similar sentences from a corpus.

        Args:
            query (str): The query sentence.
            corpus (List[str]): Pool of candidate sentences.
            top_k (int): Number of results to return.

        Returns:
            List[tuple[str, float]]: Ranked list of ``(sentence, score)``
            pairs, highest similarity first.
        """
        if not corpus:
            raise ValueError("corpus must not be empty.")
        top_k = min(top_k, len(corpus))

        query_vec = self.embed(query)
        corpus_vecs = self.embed_batch(corpus)

        scores = [
            float(_cosine_similarity(query_vec, cv)) for cv in corpus_vecs
        ]
        ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_st(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        )
        return vecs.astype(np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        response = self._openai_client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        vecs = [item.embedding for item in response.data]
        return np.array(vecs, dtype=np.float32)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
