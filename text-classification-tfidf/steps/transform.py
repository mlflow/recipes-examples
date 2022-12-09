"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import (
    TfidfTransformer,
    CountVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


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
                                    (
                                        "count",
                                        CountVectorizer(
                                            stop_words="english", max_features=5000
                                        ),
                                    ),
                                    (
                                        "tfid",
                                        TfidfTransformer(
                                            sublinear_tf=True,
                                            norm="l2",
                                        ),
                                    ),
                                    (
                                        "todense",
                                        FunctionTransformer(
                                            lambda x: x.todense(), accept_sparse=True
                                        ),
                                    ),
                                    ("pca", PCA(n_components=250)),
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
