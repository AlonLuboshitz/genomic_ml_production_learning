"""
Prediction and model comparison module.

Thin facade over model_utils — exists for backward compatibility.
New code should import directly from genomics_ml.models.model_utils.
"""

from genomics_ml.models.model_utils import (
    compare_models,
    load_model,
    predict,
    predict_proba,
)

__all__ = ["load_model", "predict", "predict_proba", "compare_models"]
