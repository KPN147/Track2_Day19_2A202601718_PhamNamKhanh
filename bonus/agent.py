"""Minimal hybrid memory: Qdrant episodes + Feast user context."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

COLLECTION = "bonus_memories"
RRF_K = 60
PROFILE_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


class HybridMemoryAgent:
    """Store user-scoped episodes and combine them with online profile features."""

    def __init__(self, client=None, feature_store=None, embedder=None) -> None:
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(":memory:")
        if feature_store is None:
            from feast import FeatureStore

            repo = Path(__file__).resolve().parents[1] / "app" / "feast_repo"
            feature_store = FeatureStore(repo_path=str(repo))
        self.feature_store = feature_store
        self._memories: list[dict[str, str]] = []

        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.embedder.dim, distance=models.Distance.COSINE
                ),
            )

    @staticmethod
    def chunk(text: str, max_tokens: int = 120, overlap: int = 20) -> list[str]:
        """Split on paragraphs first, then use overlapping whitespace-token windows."""
        if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
            raise ValueError("require max_tokens > overlap >= 0")
        chunks: list[str] = []
        for paragraph in (p.strip() for p in text.split("\n\n")):
            tokens = paragraph.split()
            start = 0
            while tokens and start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunks.append(" ".join(tokens[start:end]))
                if end == len(tokens):
                    break
                start = end - overlap
        return chunks

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Chunk, embed, and synchronously upsert one episodic memory."""
        chunks = self.chunk(text)
        if not chunks:
            raise ValueError("memory text must not be empty")
        vectors = list(self.embedder.embed(chunks))
        created_at = datetime.now(timezone.utc).isoformat()
        points = []
        for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid4())
            payload = {
                "user_id": user_id,
                "text": chunk,
                "created_at": created_at,
                "chunk_index": chunk_index,
            }
            points.append(models.PointStruct(
                id=point_id, vector=vector.tolist(), payload=payload
            ))
            self._memories.append({"point_id": point_id, **payload})
        self.client.upsert(collection_name=COLLECTION, points=points)

    def _semantic_ids(self, query: str, user_id: str, limit: int) -> list[str]:
        vector = next(self.embedder.embed([query])).tolist()
        user_filter = models.Filter(must=[models.FieldCondition(
            key="user_id", match=models.MatchValue(value=user_id)
        )])
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=user_filter,
            limit=limit,
        ).points
        return [str(hit.id) for hit in hits]

    def _keyword_ids(self, query: str, user_id: str, limit: int) -> list[str]:
        rows = [m for m in self._memories if m["user_id"] == user_id]
        if not rows:
            return []
        bm25 = BM25Okapi([m["text"].lower().split() for m in rows])
        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(rows)), key=lambda i: -scores[i])[:limit]
        return [rows[i]["point_id"] for i in order]

    @staticmethod
    def rrf(rankings: list[list[str]], top_k: int, rrf_k: int = RRF_K) -> list[str]:
        scores: dict[str, float] = {}
        for ranked_ids in rankings:
            for rank, point_id in enumerate(ranked_ids, start=1):
                scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank)
        return [key for key, _ in sorted(scores.items(), key=lambda item: -item[1])[:top_k]]

    def _retrieve(self, query: str, user_id: str, mode: str, top_k: int) -> list[dict]:
        depth = max(top_k * 5, 10)
        semantic = self._semantic_ids(query, user_id, depth)
        if mode == "semantic":
            ranked = semantic[:top_k]
        elif mode == "hybrid":
            ranked = self.rrf([self._keyword_ids(query, user_id, depth), semantic], top_k)
        else:
            raise ValueError("mode must be 'semantic' or 'hybrid'")
        by_id = {m["point_id"]: m for m in self._memories}
        return [by_id[point_id] for point_id in ranked if point_id in by_id]

    def _profile(self, user_id: str) -> dict[str, object]:
        response = self.feature_store.get_online_features(
            features=PROFILE_FEATURES, entity_rows=[{"user_id": user_id}]
        ).to_dict()
        return {name.split(":")[-1]: response.get(name.split(":")[-1], [None])[0]
                for name in PROFILE_FEATURES}

    def recall(
        self,
        query: str,
        user_id: str = "u_001",
        mode: Literal["semantic", "hybrid"] = "hybrid",
    ) -> str:
        """Return profile, recent activity, and the top three user-scoped memories."""
        profile = self._profile(user_id)
        memories = self._retrieve(query, user_id, mode, top_k=3)
        memory_lines = "\n".join(f"  {i}. {m['text']}" for i, m in enumerate(memories, 1))
        return (
            f"User profile: language={profile['preferred_language']}, "
            f"reading_speed={profile['reading_speed_wpm']}wpm, "
            f"topic_affinity={profile['topic_affinity']}.\n"
            f"Recent activity: queries_last_hour={profile['queries_last_hour']}, "
            f"distinct_topics_24h={profile['distinct_topics_24h']}.\n"
            f"Top memories ({mode}):\n{memory_lines or '  (none)'}"
        )
