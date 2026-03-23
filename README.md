# sentence-embedder

A lightweight **internal Python library** for generating sentence/semantic embeddings with a clean, unified API.

---

## Features

| Feature | Details |
|---|---|
| **Backends** | SentenceTransformers (local, offline) · OpenAI API |
| **Single & batch** | `embed()` and `embed_batch()` |
| **Similarity** | Cosine similarity between two sentences |
| **Semantic search** | `most_similar()` over an in-memory corpus |
| **Disk cache** | `EmbeddingCache` — skip re-embedding seen texts |
| **Typed** | Full type hints, compatible with `mypy` |

---

## Installation

```bash
# Clone / copy this package into your project, then:

# SentenceTransformers backend (local, recommended)
pip install -e ".[st]"

# OpenAI backend
pip install -e ".[openai]"

# Both
pip install -e ".[all]"

# With dev tools (pytest, ruff, mypy)
pip install -e ".[all,dev]"
```

---

## Quick Start

### 1 — Basic embedding

```python
from sentence_embedder import SentenceEmbedder

embedder = SentenceEmbedder()                 # defaults: SentenceTransformers, all-MiniLM-L6-v2

vec = embedder.embed("The cat sat on the mat.")
print(vec.shape)   # (384,)
```

### 2 — Batch embedding

```python
texts = ["Hello world", "Machine learning is fun", "I love Python"]
vecs = embedder.embed_batch(texts)
print(vecs.shape)  # (3, 384)
```

### 3 — Cosine similarity

```python
score = embedder.similarity("fast car", "quick automobile")
print(score)  # ~0.85
```

### 4 — Semantic search

```python
corpus = [
    "The stock market crashed today.",
    "Scientists discover a new planet.",
    "Football team wins the championship.",
    "A new AI model beats human performance.",
]

results = embedder.most_similar("breakthrough in artificial intelligence", corpus, top_k=2)
for sentence, score in results:
    print(f"{score:.3f}  {sentence}")
```

### 5 — Disk cache (avoid re-embedding)

```python
from sentence_embedder import SentenceEmbedder, EmbeddingCache

base = SentenceEmbedder()
embedder = EmbeddingCache(base, cache_dir=".cache/embeddings")

vec = embedder.embed("Hello world")   # computed → stored on disk
vec = embedder.embed("Hello world")   # loaded from cache instantly
```

---

## OpenAI Backend

```python
from sentence_embedder import SentenceEmbedder

embedder = SentenceEmbedder(
    backend="openai",
    model_name="text-embedding-3-small",
    openai_api_key="sk-...",   # or set OPENAI_API_KEY env var
)

vec = embedder.embed("Hello from OpenAI!")
```

---

## API Reference

### `SentenceEmbedder`

| Method | Signature | Description |
|---|---|---|
| `embed` | `(text: str) → ndarray` | Embed one sentence → 1-D vector |
| `embed_batch` | `(texts: List[str]) → ndarray` | Embed many → 2-D array `(N, dim)` |
| `similarity` | `(a: str, b: str) → float` | Cosine similarity in `[-1, 1]` |
| `most_similar` | `(query, corpus, top_k=5) → List[tuple]` | Ranked `(sentence, score)` pairs |

### `EmbeddingCache`

| Method | Signature | Description |
|---|---|---|
| `embed` | `(text: str) → ndarray` | Cache-aware single embed |
| `embed_batch` | `(texts: List[str]) → ndarray` | Cache-aware batch embed |
| `clear` | `() → None` | Wipe the cache database |

---

## Running Tests

```bash
pytest
# With coverage:
pytest --cov=sentence_embedder --cov-report=term-missing
```

---

## Package Structure

```
sentence_embedder/
├── sentence_embedder/
│   ├── __init__.py       # Public exports
│   ├── embedder.py       # SentenceEmbedder (core)
│   └── cache.py          # EmbeddingCache (disk cache)
├── tests/
│   └── test_embedder.py  # Unit tests (no model/network needed)
├── pyproject.toml        # Build config & dependencies
└── README.md
```

---

## Choosing a Model

| Model | Dim | Speed | Quality | Use case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ★★★ | Default, general purpose |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | ★★★★ | Higher quality |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ⚡⚡⚡ | ★★★★ | Q&A / search |
| `text-embedding-3-small` *(OpenAI)* | 1536 | API | ★★★★★ | Best quality, needs key |
