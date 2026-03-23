"""
sentence_embedder
=================
A lightweight internal library for generating sentence/semantic embeddings.

Quick start
-----------
>>> from sentence_embedder import SentenceEmbedder
>>> embedder = SentenceEmbedder()                        # uses all-MiniLM-L6-v2
>>> vec = embedder.embed("The cat sat on the mat.")
>>> vec.shape
(384,)

>>> results = embedder.most_similar("fast car", ["racing vehicle", "slow snail", "quick automobile"])
>>> results[0]
('quick automobile', 0.87...)
"""

from sentence_embedder.embedder import SentenceEmbedder
from sentence_embedder.cache import EmbeddingCache

__all__ = ["SentenceEmbedder", "EmbeddingCache"]
__version__ = "1.0.0"
