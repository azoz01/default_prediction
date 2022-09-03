import pandas as pd
import pytest

from pipelines.numerical.gaussian_like import ToGaussianTransformer
from sklearn.preprocessing import PowerTransformer


@pytest.fixture
def input_df():
    return pd.DataFrame.from_dict(
        data={
            "cat_col1": ["cat1", "cat2", "cat3"],
            "num_col1": [1, 2, 3],
            "num_col2": [0.1, 0.2, 0.3],
            "num_col3": [4, 5, -1],
            "cat_col2": ["cat1", "cat2", "cat3"],
        },
        orient="columns",
    )


@pytest.fixture
def expected_df(input_df):
    transformer = PowerTransformer()
    transformed_numerical = transformer.fit_transform(
        input_df[["num_col1", "num_col2", "num_col3"]]
    )
    expected_df = input_df.copy()
    expected_df[["num_col1", "num_col2", "num_col3"]] = transformed_numerical
    return expected_df


def test_transform(input_df, expected_df):
    transformer = ToGaussianTransformer(
        columns_to_transform=["num_col1", "num_col2", "num_col3"]
    )
    transformer.fit(input_df)
    actual_df = transformer.transform(input_df)
    assert (actual_df.columns == expected_df.columns).all()
    assert (actual_df.values == expected_df.values).all()
