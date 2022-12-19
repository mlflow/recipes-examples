"""
Show resolved
This module defines the following routines used by the 'train' step:
- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Dict, Any


def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    from sklearn.linear_model import SGDClassifier

    if estimator_params is None:
        estimator_params = {}
    return SGDClassifier(random_state=42, **estimator_params)
