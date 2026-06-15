-- ============================================================
-- SQLite schema for ML experiment metadata
-- TODO: Complete the CREATE TABLE statements below.
-- Refer to teaching_examples/example_create_tables.sql
-- for the full reference.
-- ============================================================

-- ── Runs ────────────────────────────────────────────────────
-- Stores one row per training run, including dataset info
-- so every run is self-describing and reproducible.
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    run_name        TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    status          TEXT    NOT NULL DEFAULT 'running',
    model_path      TEXT,
    -- dataset tracking for reproducibility
    dataset_path    TEXT,
    test_size       REAL,
    random_state    INTEGER NOT NULL DEFAULT 42,
    training_rows   INTEGER,
    testing_rows    INTEGER,
    num_features    INTEGER
);

-- ── Params ──────────────────────────────────────────────────
-- Model parameters as key-value pairs per run.
CREATE TABLE IF NOT EXISTS params (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    param_name  TEXT    NOT NULL,
    param_value TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- ── Metrics ─────────────────────────────────────────────────
-- Evaluation metrics per run (accuracy, precision, etc.).
CREATE TABLE IF NOT EXISTS metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL,
    metric_name  TEXT    NOT NULL,
    metric_value REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- ── Predictions ─────────────────────────────────────────────
-- Per-sample predictions, linked to source data by row_index.
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    split           TEXT    NOT NULL DEFAULT 'test',
    row_index       INTEGER NOT NULL,
    prediction      INTEGER NOT NULL,
    actual          INTEGER,
    probability     REAL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Indexes speed up lookups when you have hundreds of runs.
CREATE INDEX IF NOT EXISTS idx_params_run_id       ON params(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run_id      ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run_id  ON predictions(run_id);