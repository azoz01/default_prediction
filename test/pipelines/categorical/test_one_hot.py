import pytest
import pandas as pd
from pipelines.categorical import OneHotColumnEncoder


@pytest.fixture
def input_df():
    return pd.DataFrame.from_dict(
        data={
            "num_col1": [1, 2, 3],
            "cat_col1": ["val1", "val2", "val3"],
            "cat_col2": ["val4", "val5", "val6"],
            "cat_col3": ["val7", "val8", "val9"],
            "num_col2": [4, 5, 6],
        },
        orient="columns",
    )


@pytest.fixture
def expected_df():
    return pd.DataFrame.from_dict(
        data={
            "num_col1": [1, 2, 3],
            "num_col2": [4, 5, 6],
            "cat_col1_val1": [1, 0, 0],
            "cat_col1_val2": [0, 1, 0],
            "cat_col1_val3": [0, 0, 1],
            "cat_col2_val4": [1, 0, 0],
            "cat_col2_val5": [0, 1, 0],
            "cat_col2_val6": [0, 0, 1],
            "cat_col3_val7": [1, 0, 0],
            "cat_col3_val8": [0, 1, 0],
            "cat_col3_val9": [0, 0, 1],
        },
        orient="columns",
    )


def test_transform(input_df, expected_df):
    transformer = OneHotColumnEncoder(
        columns_to_encode=["cat_col1", "cat_col2", "cat_col3"]
    )
    transformer.fit(input_df)
    actual_df = transformer.transform(input_df)

    assert (actual_df.columns == expected_df.columns).all()
    assert (actual_df.values == expected_df.values).all()
