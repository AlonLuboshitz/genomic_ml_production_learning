-- ============================================================
-- Model run summary query
-- Joins runs + metrics + params to compare all completed runs.
-- Uses a filtered LEFT JOIN on params to show a specific
-- parameter (n_estimators) without duplicating rows.
-- ============================================================

SELECT
    r.id              AS run_id,
    r.run_name,
    r.model_name,
    r.timestamp,
    r.dataset_path,
    r.test_size,
    r.training_rows,
    r.testing_rows,
    r.model_path,
    -- specific param (filtered join — one value per run)
    p.param_value     AS n_estimators,
    -- metrics pivoted from rows into columns
    MAX(CASE WHEN m.metric_name = 'accuracy'   THEN m.metric_value END) AS accuracy,
    MAX(CASE WHEN m.metric_name = 'precision'  THEN m.metric_value END) AS precision,
    MAX(CASE WHEN m.metric_name = 'recall'     THEN m.metric_value END) AS recall,
    MAX(CASE WHEN m.metric_name = 'f1-score'   THEN m.metric_value END) AS f1
FROM runs r
LEFT JOIN metrics m ON m.run_id = r.id
LEFT JOIN params p  ON p.run_id = r.id AND p.param_name = 'n_estimators'
WHERE r.status = 'completed'
GROUP BY r.id
ORDER BY accuracy DESC;
