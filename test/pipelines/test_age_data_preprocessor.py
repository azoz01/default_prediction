import pytest
import pandas as pd
from pipelines.age_data_preprocessor import AgeDataPreprocessor


@pytest.fixture
def df_days_birth():
    return pd.DataFrame(
        {"DAYS_BIRTH": [366, -7400, -11104, -16080, -19365, -20460, -29323]}
    )


@pytest.fixture
def expected_age_df():
    return pd.DataFrame(
        {
            "AGE": [
                "Non-positive",
                "<=25",
                "25< & <=35",
                "35< & <=45",
                "45< & <=55",
                "55< & <=65",
                "<65",
            ],
        }
    )


@pytest.fixture
def df_days_employed():
    return pd.DataFrame({"DAYS_EMPLOYED": [31, -155, -650, -1524, -2432]})


@pytest.fixture
def expected_months_employed_df():
    return pd.DataFrame(
        {
            "MONTHS_EMPLOYED": [
                "Non-positive",
                "<=6",
                "<6 & <=24",
                "<24 & <=60",
                "<60",
            ]
        }
    )


@pytest.fixture
def df_days_registration():
    return pd.DataFrame(
        {"DAYS_REGISTRATION": [1000, -824, -3459, -6524, -10240]}
    )


@pytest.fixture
def expected_years_registration_df():
    return pd.DataFrame(
        {
            "YEARS_REGISTRATION": [
                "Negative",
                "<=5",
                "<5 & <=10",
                "<10 & <=20",
                "<20",
            ]
        }
    )


@pytest.fixture
def input_transform_df():
    return pd.DataFrame.from_dict(
        {
            "DAYS_BIRTH": [366, -7400, -11104, -16080, -19365, -20460,],
            "DAYS_EMPLOYED": [31, -155, -650, -1524, -2432, 31],
            "DAYS_REGISTRATION": [1000, -824, -3459, -6524, -10240, 1000],
            "OWN_CAR_AGE": [None, -1, 2, 5, 9, 12],
            "DUMMY_COLUMN": [10, 9, 8, 7, 6, 5],
        },
        orient="columns",
    )


@pytest.fixture
def expected_transformed_df():
    return pd.DataFrame.from_dict(
        {
            "AGE": [
                "Non-positive",
                "<=25",
                "25< & <=35",
                "35< & <=45",
                "45< & <=55",
                "55< & <=65",
            ],
            "MONTHS_EMPLOYED": [
                "Non-positive",
                "<=6",
                "<6 & <=24",
                "<24 & <=60",
                "<60",
                "Non-positive",
            ],
            "YEARS_REGISTRATION": [
                "Negative",
                "<=5",
                "<5 & <=10",
                "<10 & <=20",
                "<20",
                "Negative",
            ],
            "OWN_CAR_AGE": [
                "Undefined",
                "Undefined",
                "3<=",
                "3<&<=6",
                "6<&<=9",
                "9<",
            ],
            "DUMMY_COLUMN": [10, 9, 8, 7, 6, 5],
        },
        orient="columns",
    )


@pytest.fixture
def df_own_car_age():
    return pd.DataFrame({"OWN_CAR_AGE": [None, -1, 2, 5, 9, 12]})


@pytest.fixture
def expected_own_car_age_df():
    return pd.DataFrame(
        {
            "OWN_CAR_AGE": [
                "Undefined",
                "Undefined",
                "3<=",
                "3<&<=6",
                "6<&<=9",
                "9<",
            ]
        }
    )


def test_process_days_birth(df_days_birth, expected_age_df):
    preprocessor = AgeDataPreprocessor()
    actual_df = preprocessor._process_days_birth(df_days_birth)
    assert actual_df.columns == expected_age_df.columns
    assert (actual_df.values == expected_age_df.values).all(axis=None)


def test_process_days_employed(df_days_employed, expected_months_employed_df):
    preprocessor = AgeDataPreprocessor()
    actual_df = preprocessor._process_days_employed(df_days_employed)
    assert actual_df.columns == expected_months_employed_df.columns
    assert (actual_df.values == expected_months_employed_df.values).all(
        axis=None
    )


def test_process_days_registration(
    df_days_registration, expected_years_registration_df
):
    preprocessor = AgeDataPreprocessor()
    actual_df = preprocessor._process_days_registration(df_days_registration)
    assert actual_df.columns == expected_years_registration_df.columns
    assert (actual_df.values == expected_years_registration_df.values).all(
        axis=None
    )


def test_process_own_car_age(df_own_car_age, expected_own_car_age_df):
    preprocessor = AgeDataPreprocessor()
    actual_df = preprocessor._process_own_car_age(df_own_car_age)
    assert actual_df.columns == expected_own_car_age_df.columns
    assert (actual_df.values == expected_own_car_age_df.values).all(axis=None)


def test_transform(input_transform_df, expected_transformed_df):
    preprocessor = AgeDataPreprocessor()
    actual_df = preprocessor.transform(input_transform_df)
    assert (actual_df.columns == expected_transformed_df.columns).all(
        axis=None
    )
    assert (actual_df.values == expected_transformed_df.values).all(axis=None)
