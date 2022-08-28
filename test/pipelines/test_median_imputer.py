import pytest
import pandas as pd
from pipelines.median_imputer import MedianImputer


@pytest.fixture
def input_transform_data():
    return pd.DataFrame.from_dict(
        {
            "APARTMENTS_AVG": [1, 2, 4, None],
            "BASEMENTAREA_AVG": [1, 2, 4, None],
            "YEARS_BEGINEXPLUATATION_AVG": [1, 2, 4, None],
            "YEARS_BUILD_AVG": [1, 2, 4, None],
            "COMMONAREA_AVG": [1, 2, 4, None],
            "ELEVATORS_AVG": [1, 2, 4, None],
            "ENTRANCES_AVG": [1, 2, 4, None],
            "FLOORSMAX_AVG": [1, 2, 4, None],
            "FLOORSMIN_AVG": [1, 2, 4, None],
            "LANDAREA_AVG": [1, 2, 4, None],
            "LIVINGAPARTMENTS_AVG": [1, 2, 4, None],
            "LIVINGAREA_AVG": [1, 2, 4, None],
            "NONLIVINGAPARTMENTS_AVG": [1, 2, 4, None],
            "NONLIVINGAREA_AVG": [1, 2, 4, None],
            "APARTMENTS_MODE": [1, 2, 4, None],
            "BASEMENTAREA_MODE": [1, 2, 4, None],
            "YEARS_BEGINEXPLUATATION_MODE": [1, 2, 4, None],
            "YEARS_BUILD_MODE": [1, 2, 4, None],
            "COMMONAREA_MODE": [1, 2, 4, None],
            "ELEVATORS_MODE": [1, 2, 4, None],
            "ENTRANCES_MODE": [1, 2, 4, None],
            "FLOORSMAX_MODE": [1, 2, 4, None],
            "FLOORSMIN_MODE": [1, 2, 4, None],
            "LANDAREA_MODE": [1, 2, 4, None],
            "LIVINGAPARTMENTS_MODE": [1, 2, 4, None],
            "LIVINGAREA_MODE": [1, 2, 4, None],
            "NONLIVINGAPARTMENTS_MODE": [1, 2, 4, None],
            "NONLIVINGAREA_MODE": [1, 2, 4, None],
            "APARTMENTS_MEDI": [1, 2, 4, None],
            "BASEMENTAREA_MEDI": [1, 2, 4, None],
            "YEARS_BEGINEXPLUATATION_MEDI": [1, 2, 4, None],
            "YEARS_BUILD_MEDI": [1, 2, 4, None],
            "COMMONAREA_MEDI": [1, 2, 4, None],
            "ELEVATORS_MEDI": [1, 2, 4, None],
            "ENTRANCES_MEDI": [1, 2, 4, None],
            "FLOORSMAX_MEDI": [1, 2, 4, None],
            "FLOORSMIN_MEDI": [1, 2, 4, None],
            "LANDAREA_MEDI": [1, 2, 4, None],
            "LIVINGAPARTMENTS_MEDI": [1, 2, 4, None],
            "LIVINGAREA_MEDI": [1, 2, 4, None],
            "NONLIVINGAPARTMENTS_MEDI": [1, 2, 4, None],
            "NONLIVINGAREA_MEDI": [1, 2, 4, None],
            "FONDKAPREMONT_MODE": [1, 2, 4, None],
            "HOUSETYPE_MODE": [1, 2, 4, None],
            "TOTALAREA_MODE": [1, 2, 4, None],
            "WALLSMATERIAL_MODE": [1, 2, 4, None],
            "EMERGENCYSTATE_MODE": [1, 2, 4, None],
            "NOT_TO_IMPUTE_COLUMN": [1, 2, 4, None],
        },
        orient="columns",
    )


@pytest.fixture
def expected_transform_data():
    return pd.DataFrame.from_dict(
        {
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
            "FONDKAPREMONT_MODE": [1, 2, 4, 2],
            "HOUSETYPE_MODE": [1, 2, 4, 2],
            "TOTALAREA_MODE": [1, 2, 4, 2],
            "WALLSMATERIAL_MODE": [1, 2, 4, 2],
            "EMERGENCYSTATE_MODE": [1, 2, 4, 2],
            "NOT_TO_IMPUTE_COLUMN": [1, 2, 4, None],
        },
        orient="columns",
    )


def test_imputer(input_transform_data, expected_transform_data):
    imputer = MedianImputer()
    imputer.fit(input_transform_data)
    actual_df = imputer.transform(input_transform_data)
    pd.testing.assert_frame_equal(
        actual_df, expected_transform_data, check_dtype=False
    )

