"""Five-query demonstration for the bonus hybrid-memory POC."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent, PROFILE_FEATURES  # noqa: E402


def _online_profile_ready(store) -> bool:
    try:
        result = store.get_online_features(
            features=PROFILE_FEATURES, entity_rows=[{"user_id": "u_001"}]
        ).to_dict()
        return result.get("preferred_language", [None])[0] is not None
    except Exception:
        return False


def bootstrap_feast():
    """Create and materialize deterministic local features when NB4 has not run."""
    from feast import FeatureStore
    from app.feast_repo.feature_views import (
        item,
        item_popularity_features,
        query_velocity_features,
        user,
        user_profile_features,
    )

    repo = ROOT / "app" / "feast_repo"
    store = FeatureStore(repo_path=str(repo))
    if _online_profile_ready(store):
        return store

    now = datetime.now(timezone.utc).replace(microsecond=0)
    data = repo / "data"
    data.mkdir(exist_ok=True)
    pl.DataFrame({
        "user_id": ["u_001"],
        "reading_speed_wpm": [240],
        "preferred_language": ["mix"],
        "topic_affinity": ["cloud"],
        "event_timestamp": [now],
    }).write_parquet(data / "user_profile.parquet")
    pl.DataFrame({
        "user_id": ["u_001"],
        "queries_last_hour": [7],
        "distinct_topics_24h": [3],
        "event_timestamp": [now],
    }).write_parquet(data / "query_velocity.parquet")
    pl.DataFrame({
        "doc_id": ["bonus_001"],
        "click_count_24h": [12],
        "ctr_7d": [0.42],
        "avg_dwell_seconds": [65.0],
        "event_timestamp": [now],
    }).write_parquet(data / "item_popularity.parquet")

    store.apply([
        user, item, user_profile_features,
        query_velocity_features, item_popularity_features,
    ])
    store.materialize(
        start_date=now - timedelta(minutes=1),
        end_date=now + timedelta(seconds=1),
    )
    assert _online_profile_ready(store), "Feast profile bootstrap failed"
    return store


def main() -> int:
    store = bootstrap_feast()
    agent = HybridMemoryAgent(feature_store=store)
    memories = [
        "Tôi đã đọc ghi chú về Kubernetes: pod, deployment và autoscaling bằng HPA.",
        "Tài liệu cloud giải thích tự động mở rộng hạ tầng theo lưu lượng người dùng.",
        "Cloud security cần least privilege, mã hóa dữ liệu và xoay vòng secrets.",
        "Bài AI gần đây nói về embedding, vector search và mô hình RAG.",
    ]
    for memory in memories:
        agent.remember(memory)

    queries = [
        ("Tôi đã đọc gì về Kubernetes?", "semantic"),
        ("Recommend đọc gì tiếp", "hybrid"),
        ("Tôi đang quan tâm gì gần đây?", "hybrid"),
        ("Tài liệu về tự động mở rộng hạ tầng?", "hybrid"),
        ("Cho tôi summary cloud security", "hybrid"),
    ]
    for number, (query, mode) in enumerate(queries, 1):
        context = agent.recall(query, mode=mode)
        assert context and "Top memories" in context
        print(f"\n[{number}] {query}\n{context}")
    print("\nBonus demo completed: 5/5 queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
