import pytest
import pandas as pd
import numpy as np
from pipelines.null_removal.mode_imputer import ModeImputer
from pipelines.constants import COLUMNS_TO_IMPUTE_MODE


@pytest.fixture
def input_transform_data():
    return pd.DataFrame.from_dict(
        {
            "FONDKAPREMONT_MODE": ["A", "B", "A", np.nan],
            "HOUSETYPE_MODE": ["B", "C", "C", np.nan],
            "WALLSMATERIAL_MODE": [np.nan, "B", "B", "A"],
            "EMERGENCYSTATE_MODE": ["D", np.nan, "C", "C"],
        },
        orient="columns",
    )


@pytest.fixture
def expected_transform_data():
    return pd.DataFrame.from_dict(
        {
            "FONDKAPREMONT_MODE": ["A", "B", "A", "A"],
            "HOUSETYPE_MODE": ["B", "C", "C", "C"],
            "WALLSMATERIAL_MODE": ["B", "B", "B", "A"],
            "EMERGENCYSTATE_MODE": ["D", "C", "C", "C"],
        },
        orient="columns",
    )


def test_imputer(input_transform_data, expected_transform_data):
    imputer = ModeImputer(columns_to_impute=COLUMNS_TO_IMPUTE_MODE)
    imputer.fit(input_transform_data)
    actual_df = imputer.transform(input_transform_data)
    pd.testing.assert_frame_equal(
        actual_df, expected_transform_data, check_dtype=False
    )

