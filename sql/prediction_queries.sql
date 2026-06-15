-- ============================================================
-- SQL deployment & integration — prediction queries
--
-- These queries represent what the Python code does
-- internally via database.py, shown here as raw SQL
-- for reference and ad-hoc inspection.
-- ============================================================

-- ── Insert predictions for a run ────────────────────────────
-- (This is what insert_predictions() does in Python)
INSERT INTO predictions (run_id, split, row_index, prediction, actual, probability)
VALUES (1, 'test', 0, 1, 1, 0.95);

-- ── Retrieve all predictions for a specific run ─────────────
SELECT * FROM predictions WHERE run_id = 1 ORDER BY row_index;

-- ── Count predictions per run ───────────────────────────────
SELECT run_id, split, COUNT(*) AS total
FROM predictions
GROUP BY run_id, split
ORDER BY run_id;

-- ── Find the latest completed run's metadata ────────────────
SELECT id, run_name, model_name, timestamp, dataset_path, test_size
FROM runs
WHERE status = 'completed'
ORDER BY timestamp DESC
LIMIT 1;

-- ── Get all runs with their best metric ─────────────────────
SELECT r.id, r.run_name, r.model_name, m.metric_name, m.metric_value
FROM runs r
JOIN metrics m ON m.run_id = r.id
WHERE m.metric_name = 'accuracy'
ORDER BY m.metric_value DESC;

-- ── Count total predictions and average confidence ──────────
SELECT
    COUNT(*)               AS total_predictions,
    ROUND(AVG(probability), 3) AS avg_confidence
FROM predictions
WHERE run_id = 1;
