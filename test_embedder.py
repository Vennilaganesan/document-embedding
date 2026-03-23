"""
tests/test_embedder.py
----------------------
Unit tests for SentenceEmbedder and EmbeddingCache.
Uses mocks so no model or network is required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_st_model(dim: int = 8):
    """Return a mock SentenceTransformer-like model."""
    model = MagicMock()
    model.encode = lambda texts, **kw: np.random.rand(len(texts), dim).astype(np.float32)
    return model


# ---------------------------------------------------------------------------
# SentenceEmbedder — SentenceTransformers backend
# ---------------------------------------------------------------------------

class TestSentenceEmbedderST:
    """Tests using the (mocked) SentenceTransformers backend."""

    @pytest.fixture()
    def embedder(self):
        with patch(
            "sentence_embedder.embedder.SentenceEmbedder._init_sentence_transformers"
        ):
            from sentence_embedder import SentenceEmbedder

            emb = SentenceEmbedder(backend="sentencetransformers")
            emb._model = make_mock_st_model(dim=8)
            return emb

    def test_embed_returns_1d_array(self, embedder):
        vec = embedder.embed("Hello world")
        assert vec.ndim == 1
        assert vec.shape == (8,)

    def test_embed_batch_returns_2d_array(self, embedder):
        vecs = embedder.embed_batch(["Hello", "World", "Test"])
        assert vecs.ndim == 2
        assert vecs.shape == (3, 8)

    def test_embed_empty_string_raises(self, embedder):
        with pytest.raises(ValueError, match="empty"):
            embedder.embed("")

    def test_embed_batch_empty_list_raises(self, embedder):
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_batch([])

    def test_embed_batch_blank_string_raises(self, embedder):
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_batch(["hello", "   "])

    def test_similarity_returns_float_in_range(self, embedder):
        score = embedder.similarity("cat", "dog")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_most_similar_returns_ranked_list(self, embedder):
        corpus = ["apple", "banana", "cherry", "date"]
        results = embedder.most_similar("fruit", corpus, top_k=2)
        assert len(results) == 2
        assert all(isinstance(s, str) and isinstance(sc, float) for s, sc in results)

    def test_most_similar_top_k_capped(self, embedder):
        corpus = ["a", "b"]
        results = embedder.most_similar("query", corpus, top_k=10)
        assert len(results) == 2

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            from sentence_embedder import SentenceEmbedder
            SentenceEmbedder(backend="fakegpt")


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    """Tests for the disk cache wrapper."""

    @pytest.fixture()
    def embedder(self, tmp_path):
        """A mock embedder that counts calls."""
        with patch(
            "sentence_embedder.embedder.SentenceEmbedder._init_sentence_transformers"
        ):
            from sentence_embedder import SentenceEmbedder, EmbeddingCache

            emb = SentenceEmbedder(backend="sentencetransformers")
            emb._model = make_mock_st_model(dim=8)

            cached = EmbeddingCache(emb, cache_dir=str(tmp_path / "cache"))
            # Track how many times the underlying model is called
            original_embed_batch = emb.embed_batch
            emb._call_count = 0

            def counting_embed_batch(texts):
                emb._call_count += len(texts)
                return original_embed_batch(texts)

            emb.embed_batch = counting_embed_batch
            return emb, cached

    def test_cache_miss_then_hit(self, embedder):
        base, cached = embedder
        _ = cached.embed("The sky is blue")
        assert base._call_count == 1
        _ = cached.embed("The sky is blue")
        assert base._call_count == 1  # no second call

    def test_batch_partial_cache(self, embedder):
        base, cached = embedder
        texts = ["sentence one", "sentence two", "sentence three"]
        _ = cached.embed_batch(texts[:2])  # cache first two
        base._call_count = 0              # reset counter
        _ = cached.embed_batch(texts)     # only "sentence three" should be new
        assert base._call_count == 1

    def test_clear_cache(self, embedder, tmp_path):
        base, cached = embedder
        _ = cached.embed("some text")
        cached.clear()
        base._call_count = 0
        _ = cached.embed("some text")    # must recompute after clear
        assert base._call_count == 1
