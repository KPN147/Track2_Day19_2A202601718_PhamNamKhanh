from __future__ import annotations

import hashlib

import numpy as np
import pytest
from qdrant_client import QdrantClient

from bonus.agent import COLLECTION, HybridMemoryAgent


class FakeEmbedder:
    dim = 8

    def embed(self, texts):
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()[: self.dim]
            vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            yield vector / max(float(np.linalg.norm(vector)), 1.0)


class FakeResponse:
    def to_dict(self):
        return {
            "user_id": ["u_001"],
            "reading_speed_wpm": [240],
            "preferred_language": ["mix"],
            "topic_affinity": ["cloud"],
            "queries_last_hour": [7],
            "distinct_topics_24h": [3],
        }


class FakeFeatureStore:
    def get_online_features(self, **_kwargs):
        return FakeResponse()


@pytest.fixture
def agent():
    return HybridMemoryAgent(
        client=QdrantClient(":memory:"),
        feature_store=FakeFeatureStore(),
        embedder=FakeEmbedder(),
    )


def test_chunking_preserves_paragraphs_and_overlaps():
    short = HybridMemoryAgent.chunk("đoạn một\n\nđoạn hai")
    assert short == ["đoạn một", "đoạn hai"]

    chunks = HybridMemoryAgent.chunk(" ".join(f"t{i}" for i in range(15)), 10, 2)
    assert len(chunks) == 2
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    with pytest.raises(ValueError):
        HybridMemoryAgent.chunk("text", 10, 10)


def test_remember_uses_unique_ids(agent):
    agent.remember("same memory")
    agent.remember("same memory")
    points, _ = agent.client.scroll(COLLECTION, limit=10)
    assert len(points) == 2
    assert len({str(point.id) for point in points}) == 2


def test_rrf_is_one_based_and_honors_top_k():
    rankings = [["a", "b", "c"], ["b", "a", "d"]]
    result = HybridMemoryAgent.rrf(rankings, top_k=2, rrf_k=60)
    assert result == ["a", "b"]
    expected_a = 1 / 61 + 1 / 62
    expected_b = 1 / 62 + 1 / 61
    assert expected_a == expected_b


def test_recall_is_user_isolated_and_contains_features(agent):
    agent.remember("Kubernetes autoscaling cho cloud", user_id="u_001")
    agent.remember("private medical memory", user_id="u_002")

    context = agent.recall("Kubernetes cloud", user_id="u_001")

    assert "language=mix" in context
    assert "topic_affinity=cloud" in context
    assert "queries_last_hour=7" in context
    assert "Kubernetes autoscaling" in context
    assert "private medical memory" not in context


def test_empty_memory_and_invalid_mode_fail_clearly(agent):
    with pytest.raises(ValueError, match="must not be empty"):
        agent.remember("  \n\n  ")
    agent.remember("valid memory")
    with pytest.raises(ValueError, match="mode"):
        agent.recall("query", mode="keyword")
