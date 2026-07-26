"""
Tests for the Prefect pipeline and model promotion logic.

Categories:
  - Fast tests (no mark): promotion logic, flow structure.
    These run in ~1s and don't train real models.
  - Slow tests (@pytest.mark.slow): full end-to-end flow.
    These train real models and take ~15s each.

Run:
    pytest tests/test_pipeline.py -v             # all tests
    pytest tests/test_pipeline.py -v -m "not slow"  # fast only
"""

import os
import sqlite3
import tempfile

import pytest

from genomics_ml.models.promote import promote_if_better

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database with the runs + metrics tables."""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            run_name TEXT NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            status TEXT DEFAULT 'running',
            model_path TEXT,
            is_champion INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)
    conn.commit()
    conn.close()
    yield db_path
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def tmp_db_with_champion(tmp_db):
    """Pre-populate the DB with a champion run (accuracy 0.85)."""
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO runs (id, model_name, run_name, model_path, is_champion) "
        "VALUES (1, 'RandomForest', 'champion_v1', 'models/champion.pkl', 1)"
    )
    conn.execute(
        "INSERT INTO metrics (run_id, metric_name, metric_value) "
        "VALUES (1, 'accuracy', 0.85)"
    )
    conn.commit()
    conn.close()
    return tmp_db


# ── Promotion tests (fast, no model training) ──────────────────────────


def test_promote_when_no_champion(tmp_db):
    """No champion exists → new model should be promoted."""
    # Insert a run into the temp DB so the update doesn't fail
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO runs (id, model_name, run_name) VALUES (1, 'test', 'new_model')"
    )
    conn.commit()
    conn.close()

    # Create a dummy model file so copy doesn't fail
    dummy_path = "/tmp/_test_dummy_model.pkl"
    with open(dummy_path, "w") as f:
        f.write("dummy")

    result = promote_if_better(
        db_path=tmp_db,
        new_model_path=dummy_path,
        new_accuracy=0.9,
        new_run_id=1,
    )
    assert result is True, "Should promote when no champion exists"

    # Verify champion file was created
    assert os.path.exists("models/champion.pkl")

    # Verify DB was updated
    conn = sqlite3.connect(tmp_db)
    cur = conn.execute("SELECT is_champion FROM runs WHERE id = 1")
    assert cur.fetchone()[0] == 1, "Run should be marked as champion"
    conn.close()

    # Cleanup
    os.remove(dummy_path)
    # Don't remove champion.pkl — let other tests clean it or leave it


def test_promote_when_better(tmp_db_with_champion):
    """New model beats champion → should promote."""
    # Insert the new run (mirrors what train_model() does before promotion)
    conn = sqlite3.connect(tmp_db_with_champion)
    conn.execute(
        "INSERT INTO runs (id, model_name, run_name) VALUES (2, 'test', 'better_model')"
    )
    conn.execute(
        "INSERT INTO metrics (run_id, metric_name, metric_value) VALUES (2, 'accuracy', 0.95)"
    )
    conn.commit()
    conn.close()

    dummy_path = "/tmp/_test_dummy_model_v2.pkl"
    with open(dummy_path, "w") as f:
        f.write("dummy_v2")

    result = promote_if_better(
        db_path=tmp_db_with_champion,
        new_model_path=dummy_path,
        new_accuracy=0.95,
        new_run_id=2,
    )
    assert result is True, "Should promote when accuracy is higher"

    # New run should be champion
    conn = sqlite3.connect(tmp_db_with_champion)
    cur = conn.execute("SELECT id, is_champion FROM runs WHERE is_champion = 1")
    champion_row = cur.fetchone()
    assert champion_row is not None, "A champion should exist"
    assert champion_row[0] == 2, "Run 2 should be the new champion"

    # Old champion should be demoted
    cur = conn.execute("SELECT is_champion FROM runs WHERE id = 1")
    assert cur.fetchone()[0] == 0, "Old champion should be demoted"
    conn.close()

    os.remove(dummy_path)


def _insert_run2(db_path):
    """Helper: insert run 2 into the temp DB (called before promote_if_better)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (id, model_name, run_name) VALUES (2, 'test', 'new_model')"
    )
    conn.commit()
    conn.close()


def test_promote_when_equal(tmp_db_with_champion, tmp_path):
    """Same accuracy as champion → no promotion."""
    _insert_run2(tmp_db_with_champion)

    dummy_path = str(tmp_path / "equal_model.pkl")
    with open(dummy_path, "w") as f:
        f.write("equal")

    result = promote_if_better(
        db_path=tmp_db_with_champion,
        new_model_path=dummy_path,
        new_accuracy=0.85,
        new_run_id=2,
    )
    assert result is False, "Should NOT promote when accuracy is equal"


def test_promote_when_worse(tmp_db_with_champion, tmp_path):
    """Lower accuracy than champion → no promotion."""
    _insert_run2(tmp_db_with_champion)

    dummy_path = str(tmp_path / "worse_model.pkl")
    with open(dummy_path, "w") as f:
        f.write("worse")

    result = promote_if_better(
        db_path=tmp_db_with_champion,
        new_model_path=dummy_path,
        new_accuracy=0.7,
        new_run_id=2,
    )
    assert result is False, "Should NOT promote when accuracy is lower"


# ── Flow tests (slow — train real models) ──────────────────────────────


@pytest.mark.slow
def test_flow_returns_metrics():
    """End-to-end: training_pipeline() returns expected metrics dict."""
    from genomics_ml.orchestration.prefect_flow import training_pipeline

    metrics = training_pipeline()

    assert isinstance(metrics, dict), "Should return a dict"
    assert "accuracy" in metrics, "Should contain accuracy"
    assert "model_path" in metrics, "Should contain model_path"
    assert "run_id" in metrics, "Should contain run_id"
    assert isinstance(metrics["accuracy"], float), "Accuracy should be float"
    assert 0 < metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"


@pytest.mark.slow
def test_flow_model_override():
    """End-to-end: passing a different model type changes the trained model."""
    from genomics_ml.orchestration.prefect_flow import training_pipeline

    metrics = training_pipeline(model_type="LogisticRegression")

    assert "model_path" in metrics
    assert "LogisticRegression" in metrics["model_path"], (
        f"Expected LogisticRegression in path, got {metrics['model_path']}"
    )
