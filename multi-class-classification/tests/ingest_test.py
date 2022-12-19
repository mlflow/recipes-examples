import pytest
import os
import pandas as pd


@pytest.fixture
def sample_data():
    return pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "test_sample.parquet")
    )
