"""Model promotion logic — compare new model vs champion, promote if better.

Usage:
    from genomics_ml.models.promote import promote_if_better

    promoted = promote_if_better(
        db_path="ml_metadata.db",
        new_model_path="models/RandomForest_20260624.pkl",
        new_accuracy=0.92,
        new_run_id=5,
    )
"""

import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from genomics_ml.utils.database import get_champion
from genomics_ml.utils.logging import get_logger

logger = get_logger("genomics_ml.models.promote")

CHAMPION_PATH = "models/champion.pkl"


def promote_if_better(
    db_path: str,
    new_model_path: str,
    new_accuracy: float,
    new_run_id: int,
) -> bool:
    """Promote the new model to champion if it beats the current best.

    Compares against the highest-accuracy run in the SQLite DB.
    If no champion exists yet, or the new model is strictly better,
    it is copied to ``models/champion.pkl``.

    Args:
        db_path: Path to the SQLite metadata database.
        new_model_path: Path to the newly trained model artifact.
        new_accuracy: Accuracy of the new model on the test set.
        new_run_id: Run ID in the database for the new model.

    Returns:
        True if the model was promoted, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    champion = get_champion(conn)

    if champion is None:
        # No champion exists — promote automatically
        logger.info("No champion found — promoting new model")
        _do_promote(new_model_path, new_run_id, conn)
        conn.close()
        return True

    if new_accuracy > champion["accuracy"]:
        logger.info(
            "New model beats champion: %.4f > %.4f — promoting",
            new_accuracy,
            champion["accuracy"],
        )
        _do_promote(new_model_path, new_run_id, conn)
        conn.close()
        return True

    logger.info(
        "Champion still best: %.4f >= %.4f — no promotion",
        champion["accuracy"],
        new_accuracy,
    )
    conn.close()
    return False


def _do_promote(model_path: str, run_id: int, conn: sqlite3.Connection):
    """Copy model to champion path and mark run in DB."""
    # Copy artifact
    Path(CHAMPION_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, CHAMPION_PATH)
    logger.info("Model copied to %s", CHAMPION_PATH)

    # Demote old champion, then mark new one
    conn.execute("UPDATE runs SET is_champion = 0 WHERE is_champion = 1")
    conn.execute("UPDATE runs SET is_champion = 1 WHERE id = ?", (run_id,))
    conn.commit()
    logger.info("Run %d promoted to champion", run_id)


def get_champion_path() -> Optional[str]:
    """Return the path to the champion model if it exists, else None."""
    path = Path(CHAMPION_PATH)
    return str(path) if path.exists() else None
