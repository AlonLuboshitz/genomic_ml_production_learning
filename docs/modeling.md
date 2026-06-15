# Modeling Documentation

## Overview

The Genomics ML pipeline trains scikit-learn classifiers on gene expression data. The workflow covers data loading, validation, preprocessing, training with MLflow tracking, SQL metadata storage, and model comparison.

---

## Pipeline

### Preprocessing

Defined in `src/genomics_ml/features/preprocessing.py`:

1. **Imputer** — fills missing values (strategy: mean, configurable)
2. **Scaler** — standardizes features (StandardScaler by default)

Both steps are wrapped in a scikit-learn `Pipeline` to prevent data leakage during train/test split.

### Training

Defined in `src/genomics_ml/models/train.py`:

1. Load data (CSV via `load_data`)
2. Split into train/test (80/20 by default, stratified)
3. Build preprocessing pipeline + classifier
4. Fit on training set
5. Evaluate on test set (accuracy, classification report)
6. Save model artifact to `models/` with `joblib`
7. Log run metadata to SQLite via `database.py`

### Supported model types

Configured in `configs/default.yaml` under `model.type`:

| Model | Class |
|---|---|
| RandomForestClassifier | `sklearn.ensemble.RandomForestClassifier` |
| LogisticRegression | `sklearn.linear_model.LogisticRegression` |
| GradientBoostingClassifier | `sklearn.ensemble.GradientBoostingClassifier` |

Add new models by extending `_get_classifier()` in `train.py`.

---

## Experiment tracking (MLflow)

Each training run is automatically logged with MLflow:

- **Parameters:** model type, hyperparameters
- **Metrics:** accuracy
- **Artifacts:** the model pipeline (if configured)

View runs:
```bash
mlflow ui
```
Opens the MLflow UI at `http://localhost:5000`.

---

## SQL metadata storage

After each run, metadata is stored in `ml_metadata.db` (SQLite):

| Table | Contents |
|---|---|
| `runs` | Model name, dataset, train/test sizes, timestamp |
| `params` | Hyperparameters (key-value) |
| `metrics` | Evaluation metrics (accuracy) |
| `predictions` | Per-row predictions, actual labels, probabilities |

Query examples are in `sql/` directory.

---

## Model comparison

Defined in `src/genomics_ml/models/model_utils.py`:

```python
from genomics_ml.models.model_utils import compare_models

ranking = compare_models(["model_a.pkl", "model_b.pkl"], X_test, y_test)
# Returns sorted list: best model first
```

The comparison loads each saved model, scores it on a holdout set, and ranks by accuracy.

---

## Configuration

Key paths in `configs/default.yaml`:

| Key | Default | Description |
|---|---|---|
| `model.type` | RandomForestClassifier | Classifier to train |
| `model.path` | models/baseline.pkl | Path to saved model artifact |
| `data.raw_path` | data/raw/genomics_data.csv | Input dataset |
| `data.test_size` | 0.2 | Train/test split ratio |
| `preprocessing.scaler` | standard | Scaling strategy |
| `preprocessing.impute_strategy` | mean | Missing value strategy |
