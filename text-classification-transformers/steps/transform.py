"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    TfidfTransformer,
    CountVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from whatlies.language import HFTransformersLanguage


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    pipeline = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        (
                            "Consumer_complaint_narrative",
                            Pipeline(
                                [
                                    ("embedding", HFTransformersLanguage("facebook/bart-base")),
                                ]
                            ),
                            0,
                        )
                    ]
                ),
            )
        ],
        verbose=True,
    )
    return pipeline
