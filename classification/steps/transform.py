"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    import sklearn

    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        ('select', 'passthrough', ['jurisdictional_classification', 'negotiating_unit', 'federal_occupational_code', 'filled_positions'])
                    ]
                ),
            )
        ]
    )

