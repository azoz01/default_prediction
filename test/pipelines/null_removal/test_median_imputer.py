import pytest
import pandas as pd
import numpy as np
from pipelines.null_removal.median_imputer import MedianImputer
from pipelines.lib.constants import COLUMNS_TO_IMPUTE_MEDIAN


@pytest.fixture
def input_transform_data():
    return pd.DataFrame.from_dict(
        {
            "EXT_SOURCE_1": [1, 2, 4, np.nan],
            "EXT_SOURCE_2": [1, 2, 4, np.nan],
            "EXT_SOURCE_3": [1, 2, 4, np.nan],
            "APARTMENTS_AVG": [1, 2, 4, np.nan],
            "BASEMENTAREA_AVG": [1, 2, 4, np.nan],
            "YEARS_BEGINEXPLUATATION_AVG": [1, 2, 4, np.nan],
            "YEARS_BUILD_AVG": [1, 2, 4, np.nan],
            "COMMONAREA_AVG": [1, 2, 4, np.nan],
            "ELEVATORS_AVG": [1, 2, 4, np.nan],
            "ENTRANCES_AVG": [1, 2, 4, np.nan],
            "FLOORSMAX_AVG": [1, 2, 4, np.nan],
            "FLOORSMIN_AVG": [1, 2, 4, np.nan],
            "LANDAREA_AVG": [1, 2, 4, np.nan],
            "LIVINGAPARTMENTS_AVG": [1, 2, 4, np.nan],
            "LIVINGAREA_AVG": [1, 2, 4, np.nan],
            "NONLIVINGAPARTMENTS_AVG": [1, 2, 4, np.nan],
            "NONLIVINGAREA_AVG": [1, 2, 4, np.nan],
            "APARTMENTS_MODE": [1, 2, 4, np.nan],
            "BASEMENTAREA_MODE": [1, 2, 4, np.nan],
            "YEARS_BEGINEXPLUATATION_MODE": [1, 2, 4, np.nan],
            "YEARS_BUILD_MODE": [1, 2, 4, np.nan],
            "COMMONAREA_MODE": [1, 2, 4, np.nan],
            "ELEVATORS_MODE": [1, 2, 4, np.nan],
            "ENTRANCES_MODE": [1, 2, 4, np.nan],
            "FLOORSMAX_MODE": [1, 2, 4, np.nan],
            "FLOORSMIN_MODE": [1, 2, 4, np.nan],
            "LANDAREA_MODE": [1, 2, 4, np.nan],
            "LIVINGAPARTMENTS_MODE": [1, 2, 4, np.nan],
            "LIVINGAREA_MODE": [1, 2, 4, np.nan],
            "NONLIVINGAPARTMENTS_MODE": [1, 2, 4, np.nan],
            "NONLIVINGAREA_MODE": [1, 2, 4, np.nan],
            "APARTMENTS_MEDI": [1, 2, 4, np.nan],
            "BASEMENTAREA_MEDI": [1, 2, 4, np.nan],
            "YEARS_BEGINEXPLUATATION_MEDI": [1, 2, 4, np.nan],
            "YEARS_BUILD_MEDI": [1, 2, 4, np.nan],
            "COMMONAREA_MEDI": [1, 2, 4, np.nan],
            "ELEVATORS_MEDI": [1, 2, 4, np.nan],
            "ENTRANCES_MEDI": [1, 2, 4, np.nan],
            "FLOORSMAX_MEDI": [1, 2, 4, np.nan],
            "FLOORSMIN_MEDI": [1, 2, 4, np.nan],
            "LANDAREA_MEDI": [1, 2, 4, np.nan],
            "LIVINGAPARTMENTS_MEDI": [1, 2, 4, np.nan],
            "LIVINGAREA_MEDI": [1, 2, 4, np.nan],
            "NONLIVINGAPARTMENTS_MEDI": [1, 2, 4, np.nan],
            "NONLIVINGAREA_MEDI": [1, 2, 4, np.nan],
            "TOTALAREA_MODE": [1, 2, 4, np.nan],
            "NOT_TO_IMPUTE_COLUMN": [1, 2, 4, np.nan],
        },
        orient="columns",
    )


@pytest.fixture
def expected_transform_data():
    return pd.DataFrame.from_dict(
        {
            "EXT_SOURCE_1": [1, 2, 4, 2],
            "EXT_SOURCE_2": [1, 2, 4, 2],
            "EXT_SOURCE_3": [1, 2, 4, 2],
            "APARTMENTS_AVG": [1, 2, 4, 2],
            "BASEMENTAREA_AVG": [1, 2, 4, 2],
            "YEARS_BEGINEXPLUATATION_AVG": [1, 2, 4, 2],
            "YEARS_BUILD_AVG": [1, 2, 4, 2],
            "COMMONAREA_AVG": [1, 2, 4, 2],
            "ELEVATORS_AVG": [1, 2, 4, 2],
            "ENTRANCES_AVG": [1, 2, 4, 2],
            "FLOORSMAX_AVG": [1, 2, 4, 2],
            "FLOORSMIN_AVG": [1, 2, 4, 2],
            "LANDAREA_AVG": [1, 2, 4, 2],
            "LIVINGAPARTMENTS_AVG": [1, 2, 4, 2],
            "LIVINGAREA_AVG": [1, 2, 4, 2],
            "NONLIVINGAPARTMENTS_AVG": [1, 2, 4, 2],
            "NONLIVINGAREA_AVG": [1, 2, 4, 2],
            "APARTMENTS_MODE": [1, 2, 4, 2],
            "BASEMENTAREA_MODE": [1, 2, 4, 2],
            "YEARS_BEGINEXPLUATATION_MODE": [1, 2, 4, 2],
            "YEARS_BUILD_MODE": [1, 2, 4, 2],
            "COMMONAREA_MODE": [1, 2, 4, 2],
            "ELEVATORS_MODE": [1, 2, 4, 2],
            "ENTRANCES_MODE": [1, 2, 4, 2],
            "FLOORSMAX_MODE": [1, 2, 4, 2],
            "FLOORSMIN_MODE": [1, 2, 4, 2],
            "LANDAREA_MODE": [1, 2, 4, 2],
            "LIVINGAPARTMENTS_MODE": [1, 2, 4, 2],
            "LIVINGAREA_MODE": [1, 2, 4, 2],
            "NONLIVINGAPARTMENTS_MODE": [1, 2, 4, 2],
            "NONLIVINGAREA_MODE": [1, 2, 4, 2],
            "APARTMENTS_MEDI": [1, 2, 4, 2],
            "BASEMENTAREA_MEDI": [1, 2, 4, 2],
            "YEARS_BEGINEXPLUATATION_MEDI": [1, 2, 4, 2],
            "YEARS_BUILD_MEDI": [1, 2, 4, 2],
            "COMMONAREA_MEDI": [1, 2, 4, 2],
            "ELEVATORS_MEDI": [1, 2, 4, 2],
            "ENTRANCES_MEDI": [1, 2, 4, 2],
            "FLOORSMAX_MEDI": [1, 2, 4, 2],
            "FLOORSMIN_MEDI": [1, 2, 4, 2],
            "LANDAREA_MEDI": [1, 2, 4, 2],
            "LIVINGAPARTMENTS_MEDI": [1, 2, 4, 2],
            "LIVINGAREA_MEDI": [1, 2, 4, 2],
            "NONLIVINGAPARTMENTS_MEDI": [1, 2, 4, 2],
            "NONLIVINGAREA_MEDI": [1, 2, 4, 2],
            "TOTALAREA_MODE": [1, 2, 4, 2],
            "NOT_TO_IMPUTE_COLUMN": [1, 2, 4, np.nan],
        },
        orient="columns",
    )


def test_imputer(input_transform_data, expected_transform_data):
    imputer = MedianImputer(columns_to_impute=COLUMNS_TO_IMPUTE_MEDIAN)
    imputer.fit(input_transform_data)
    actual_df = imputer.transform(input_transform_data)
    pd.testing.assert_frame_equal(
        actual_df, expected_transform_data, check_dtype=False
    )

