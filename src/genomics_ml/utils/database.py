"""
Database utility for ML experiment tracking.

Connects the SQL schema to Python — creates the database on disk,
inserts run metadata, parameters, metrics, and predictions.

Usage:
    from genomics_ml.utils.database import init_database, insert_run

    conn = init_database("ml_metadata.db")
    run_id = insert_run(conn, model_name="RandomForest", run_name="rf-v1")
"""

import sqlite3
from typing import Any, Dict, List, Optional


def init_database(db_path: str) -> sqlite3.Connection:
    """TODO: Create the database and all tables.

    Steps:
      1. sqlite3.connect(db_path)
      2. Read sql/create_tables.sql and execute it with executescript()
      3. conn.commit()
      4. Return conn
    """
    # YOUR CODE HERE
    connection = sqlite3.connect(db_path)
    with open("sql/create_tables.sql") as f:
        connection.executescript(f.read())
    connection.commit()
    return connection


def insert_run(
    conn: sqlite3.Connection,
    model_name: str,
    run_name: str,
    status: str = "running",
    dataset_path: Optional[str] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    training_rows: Optional[int] = None,
    testing_rows: Optional[int] = None,
    num_features: Optional[int] = None,
) -> int:
    """Insert a new run and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO runs
            (model_name, run_name, status, dataset_path,
             test_size, random_state, training_rows, testing_rows, num_features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_name,
            run_name,
            status,
            dataset_path,
            test_size,
            random_state,
            training_rows,
            testing_rows,
            num_features,
        ),
    )
    last_id = cursor.lastrowid
    conn.commit()
    return last_id


def insert_params(conn: sqlite3.Connection, run_id: int, params: Dict[str, Any]):
    """TODO: Insert model parameters as key-value pairs.

    Steps:
      1. Loop over params.items()
      2. For each, conn.execute("INSERT INTO params ...", (run_id, name, str(value)))
      3. conn.commit()
    """
    # YOUR CODE HERE
    for param_name, param_value in params.items():
        conn.execute(
            "INSERT INTO params (run_id, param_name, param_value) VALUES (?, ?, ?)",
            (run_id, param_name, str(param_value)),
        )
    conn.commit()


def insert_metrics(conn: sqlite3.Connection, run_id: int, metrics: Dict[str, float]):
    """TODO: Insert evaluation metrics.

    Steps:
      1. Loop over metrics.items()
      2. For each, conn.execute("INSERT INTO metrics ...", (run_id, name, float(value)))
      3. conn.commit()
    """
    # YOUR CODE HERE
    for metric_name, metric_value in metrics.items():
        conn.execute(
            "INSERT INTO metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?)",
            (run_id, metric_name, float(metric_value)),
        )
    conn.commit()


def insert_predictions(
    conn: sqlite3.Connection,
    run_id: int,
    split: str,
    y_pred: List[int],
    y_true: Optional[List[int]] = None,
    probabilities: Optional[List[float]] = None,
):
    """TODO: Store predictions for a run.

    Args:
        conn: Database connection.
        run_id: Which run these predictions belong to.
        split: 'train', 'test', or 'val'.
        y_pred: List of predicted classes.
        y_true: Optional list of true labels.
        probabilities: Optional list of confidence scores.

    Steps:
      1. Loop over range(len(y_pred))
      2. For each i, conn.execute(...) inserting:
         (run_id, split, i, y_pred[i], y_true[i] or None, probabilities[i] or None)
      3. conn.commit()
    """
    # Convert to lists for consistent indexing (handles pandas Series/DataFrame)
    y_pred = list(y_pred)
    if y_true is not None:
        y_true = list(y_true)
    if probabilities is not None:
        probabilities = list(probabilities)

    for i in range(len(y_pred)):
        conn.execute(
            """INSERT INTO predictions
                   (run_id, split, row_index, prediction, actual, probability)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                split,
                i,
                int(y_pred[i]),
                int(y_true[i]) if y_true is not None else None,
                float(probabilities[i]) if probabilities is not None else None,
            ),
        )
    conn.commit()


def get_best_run(conn: sqlite3.Connection) -> Optional[Dict]:
    """TODO: Retrieve the run with the highest accuracy.

    Returns a dict with keys: run_id, run_name, model_name, timestamp, accuracy
    or None if no runs exist.

    Steps:
      1. Execute a SELECT that JOINs runs + metrics
      2. WHERE metric_name = 'accuracy'
      3. ORDER BY metric_value DESC LIMIT 1
      4. If row exists, return dict; otherwise return None
    """
    # YOUR CODE HERE
    cur = conn.execute("""
        SELECT r.id, r.run_name, r.model_name, r.timestamp, m.metric_value AS accuracy
        FROM runs r
        JOIN metrics m ON m.run_id = r.id
        WHERE m.metric_name = 'accuracy'
        ORDER BY m.metric_value DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "run_name": row[1],
        "model_name": row[2],
        "timestamp": row[3],
        "accuracy": row[4],
    }


def close_connection(conn, model_path, run_id):
    conn.execute(
        "UPDATE runs SET status = 'completed', model_path = ? WHERE id = ?",
        (model_path, run_id),
    )
    conn.commit()
    conn.close()
