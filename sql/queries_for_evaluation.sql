-- ============================================================
-- SQL model evaluation queries
--
-- Use the eval_practice.db database:
--   sqlite3 teaching_examples/eval_practice.db
--
-- Tables: predictions (run_id, split, row_index, prediction, actual, probability)
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- Exercise 1: Confusion matrix for run_id=2
-- Expected: 4 rows (TN=3, FP=3, FN=1, TP=3)
-- ────────────────────────────────────────────────────────────
SELECT actual, prediction, COUNT(*) AS count
FROM predictions
WHERE run_id = 2
GROUP BY actual, prediction
ORDER BY actual, prediction;

-- ────────────────────────────────────────────────────────────
-- Exercise 2: Accuracy for run_id=2
-- Expected: 60.0
-- ────────────────────────────────────────────────────────────
SELECT
    ROUND(100.0 *
        SUM(CASE WHEN prediction = actual THEN 1 ELSE 0 END)
        / COUNT(*), 2) AS accuracy_pct
FROM predictions
WHERE run_id = 2;

-- ────────────────────────────────────────────────────────────
-- Exercise 3: Precision & Recall for class=1, run_id=2
-- Expected: precision=50.0, recall=75.0
-- ────────────────────────────────────────────────────────────
SELECT
    ROUND(100.0 *
        SUM(CASE WHEN prediction = 1 AND actual = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END), 0), 2) AS precision_pct,
    ROUND(100.0 *
        SUM(CASE WHEN prediction = 1 AND actual = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN actual = 1 THEN 1 ELSE 0 END), 0), 2) AS recall_pct
FROM predictions
WHERE run_id = 2;

-- ────────────────────────────────────────────────────────────
-- Exercise 4: F1-score for class=1, run_id=2
-- F1 = 2 * (precision * recall) / (precision + recall)
-- Expected: 60.0
-- ────────────────────────────────────────────────────────────
SELECT
    ROUND(2 * (prec * rec) / (prec + rec), 2) AS f1_pct
FROM (
    SELECT
        100.0 *
        SUM(CASE WHEN prediction = 1 AND actual = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END), 0) AS prec,
        100.0 *
        SUM(CASE WHEN prediction = 1 AND actual = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN actual = 1 THEN 1 ELSE 0 END), 0) AS rec
    FROM predictions
    WHERE run_id = 2
);

-- ────────────────────────────────────────────────────────────
-- Exercise 5 (Bonus): Find the worst predictions
-- Rows where prediction != actual, ordered by probability ASC.
-- Expected: 2 rows (row_index 9 with 0.45, row_index 4 with 0.60)
-- ────────────────────────────────────────────────────────────
SELECT run_id, row_index, prediction, actual, probability
FROM predictions
WHERE run_id = 1 AND prediction != actual
ORDER BY probability ASC;
