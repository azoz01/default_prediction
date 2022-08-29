import pytest
import pandas as pd
from pipelines.null_removal.fixed_value_imputer import FixedValueImputer
from pipelines.constants import COLUMNS_TO_IMPUTE_0


@pytest.fixture
def input_transform_data():
    return pd.DataFrame.from_dict(
        {
            "OBS_30_CNT_SOCIAL_CIRCLE": [1, 2, 3, None],
            "DEF_30_CNT_SOCIAL_CIRCLE": [1, 2, None, 4],
            "OBS_60_CNT_SOCIAL_CIRCLE": [1, None, 3, 4],
            "DEF_60_CNT_SOCIAL_CIRCLE": [None, 2, 3, 4],
        },
        orient="columns",
    )


@pytest.fixture
def expected_transform_data():
    return pd.DataFrame.from_dict(
        {
            "OBS_30_CNT_SOCIAL_CIRCLE": [1, 2, 3, 0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [1, 2, 0, 4],
            "OBS_60_CNT_SOCIAL_CIRCLE": [1, 0, 3, 4],
            "DEF_60_CNT_SOCIAL_CIRCLE": [0, 2, 3, 4],
        },
        orient="columns",
    )


def test_imputer(input_transform_data, expected_transform_data):
    imputer = FixedValueImputer(
        columns_to_impute=COLUMNS_TO_IMPUTE_0, fill_value=0
    )
    imputer.fit(input_transform_data)
    actual_df = imputer.transform(input_transform_data)
    pd.testing.assert_frame_equal(
        actual_df, expected_transform_data, check_dtype=False
    )

