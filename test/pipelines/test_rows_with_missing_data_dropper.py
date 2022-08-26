import pytest
import pandas as pd
from pipelines.rows_with_missing_data_dropper import RowsWithMissingDataDropper


@pytest.fixture()
def application_data_X():
    return pd.DataFrame.from_dict(
        data={
            "0": {
                "SK_ID_CURR": 155054,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "N",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 180000.0,
                "AMT_CREDIT": 450000.0,
                "AMT_ANNUITY": None,
                "AMT_GOODS_PRICE": 450000.0,
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": "Commercial associate",
                "NAME_EDUCATION_TYPE": "Incomplete higher",
                "NAME_FAMILY_STATUS": "Single \\/ not married",
                "NAME_HOUSING_TYPE": "House \\/ apartment",
                "REGION_POPULATION_RELATIVE": 0.026392,
                "DAYS_BIRTH": -10668,
                "DAYS_EMPLOYED": -2523,
                "DAYS_REGISTRATION": -4946.0,
                "DAYS_ID_PUBLISH": -3238,
                "OWN_CAR_AGE": None,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 1,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,
                "FLAG_EMAIL": 0,
                "OCCUPATION_TYPE": "High skill tech staff",
                "CNT_FAM_MEMBERS": 1.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                "HOUR_APPR_PROCESS_START": 13,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 0,
                "LIVE_REGION_NOT_WORK_REGION": 0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 0,
                "LIVE_CITY_NOT_WORK_CITY": 0,
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "EXT_SOURCE_1": None,
                "EXT_SOURCE_2": 0.6268964227,
                "EXT_SOURCE_3": 0.3723336657,
                "APARTMENTS_AVG": 0.0124,
                "BASEMENTAREA_AVG": None,
                "YEARS_BEGINEXPLUATATION_AVG": 0.9662,
                "YEARS_BUILD_AVG": None,
                "COMMONAREA_AVG": None,
                "ELEVATORS_AVG": 0.0,
                "ENTRANCES_AVG": 0.0345,
                "FLOORSMAX_AVG": 0.0417,
                "FLOORSMIN_AVG": None,
                "LANDAREA_AVG": None,
                "LIVINGAPARTMENTS_AVG": None,
                "LIVINGAREA_AVG": 0.0115,
                "NONLIVINGAPARTMENTS_AVG": None,
                "NONLIVINGAREA_AVG": 0.0,
                "APARTMENTS_MODE": 0.0126,
                "BASEMENTAREA_MODE": None,
                "YEARS_BEGINEXPLUATATION_MODE": 0.9662,
                "YEARS_BUILD_MODE": None,
                "COMMONAREA_MODE": None,
                "ELEVATORS_MODE": 0.0,
                "ENTRANCES_MODE": 0.0345,
                "FLOORSMAX_MODE": 0.0417,
                "FLOORSMIN_MODE": None,
                "LANDAREA_MODE": None,
                "LIVINGAPARTMENTS_MODE": None,
                "LIVINGAREA_MODE": 0.0119,
                "NONLIVINGAPARTMENTS_MODE": None,
                "NONLIVINGAREA_MODE": 0.0,
                "APARTMENTS_MEDI": 0.0125,
                "BASEMENTAREA_MEDI": None,
                "YEARS_BEGINEXPLUATATION_MEDI": 0.9662,
                "YEARS_BUILD_MEDI": None,
                "COMMONAREA_MEDI": None,
                "ELEVATORS_MEDI": 0.0,
                "ENTRANCES_MEDI": 0.0345,
                "FLOORSMAX_MEDI": 0.0417,
                "FLOORSMIN_MEDI": None,
                "LANDAREA_MEDI": None,
                "LIVINGAPARTMENTS_MEDI": None,
                "LIVINGAREA_MEDI": 0.0117,
                "NONLIVINGAPARTMENTS_MEDI": None,
                "NONLIVINGAREA_MEDI": 0.0,
                "FONDKAPREMONT_MODE": None,
                "HOUSETYPE_MODE": "block of flats",
                "TOTALAREA_MODE": 0.009,
                "WALLSMATERIAL_MODE": "Stone, brick",
                "EMERGENCYSTATE_MODE": "No",
                "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                "DAYS_LAST_PHONE_CHANGE": -2.0,
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 0,
                "FLAG_DOCUMENT_4": 0,
                "FLAG_DOCUMENT_5": 1,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_7": 0,
                "FLAG_DOCUMENT_8": 0,
                "FLAG_DOCUMENT_9": 0,
                "FLAG_DOCUMENT_10": 0,
                "FLAG_DOCUMENT_11": 0,
                "FLAG_DOCUMENT_12": 0,
                "FLAG_DOCUMENT_13": 0,
                "FLAG_DOCUMENT_14": 0,
                "FLAG_DOCUMENT_15": 0,
                "FLAG_DOCUMENT_16": 0,
                "FLAG_DOCUMENT_17": 0,
                "FLAG_DOCUMENT_18": 0,
                "FLAG_DOCUMENT_19": 0,
                "FLAG_DOCUMENT_20": 0,
                "FLAG_DOCUMENT_21": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                "AMT_REQ_CREDIT_BUREAU_QRT": 1.0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
            },
            "1": {
                "SK_ID_CURR": 100837,
                "NAME_CONTRACT_TYPE": "Revolving loans",
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 2,
                "AMT_INCOME_TOTAL": 45000.0,
                "AMT_CREDIT": 135000.0,
                "AMT_ANNUITY": 6750.0,
                "AMT_GOODS_PRICE": None,
                "NAME_TYPE_SUITE": None,
                "NAME_INCOME_TYPE": "Commercial associate",
                "NAME_EDUCATION_TYPE": "Secondary \\/ secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House \\/ apartment",
                "REGION_POPULATION_RELATIVE": 0.02461,
                "DAYS_BIRTH": -10072,
                "DAYS_EMPLOYED": -381,
                "DAYS_REGISTRATION": -519.0,
                "DAYS_ID_PUBLISH": -1834,
                "OWN_CAR_AGE": None,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 1,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 0,
                "FLAG_EMAIL": 0,
                "OCCUPATION_TYPE": "Core staff",
                "CNT_FAM_MEMBERS": 4.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                "HOUR_APPR_PROCESS_START": 14,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 0,
                "LIVE_REGION_NOT_WORK_REGION": 0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 0,
                "LIVE_CITY_NOT_WORK_CITY": 0,
                "ORGANIZATION_TYPE": "Kindergarten",
                "EXT_SOURCE_1": 0.3585091509,
                "EXT_SOURCE_2": 0.3781742068,
                "EXT_SOURCE_3": None,
                "APARTMENTS_AVG": None,
                "BASEMENTAREA_AVG": None,
                "YEARS_BEGINEXPLUATATION_AVG": 0.9791,
                "YEARS_BUILD_AVG": None,
                "COMMONAREA_AVG": None,
                "ELEVATORS_AVG": 0.0,
                "ENTRANCES_AVG": 0.0345,
                "FLOORSMAX_AVG": 0.0417,
                "FLOORSMIN_AVG": None,
                "LANDAREA_AVG": None,
                "LIVINGAPARTMENTS_AVG": None,
                "LIVINGAREA_AVG": 0.007,
                "NONLIVINGAPARTMENTS_AVG": None,
                "NONLIVINGAREA_AVG": None,
                "APARTMENTS_MODE": None,
                "BASEMENTAREA_MODE": None,
                "YEARS_BEGINEXPLUATATION_MODE": 0.9791,
                "YEARS_BUILD_MODE": None,
                "COMMONAREA_MODE": None,
                "ELEVATORS_MODE": 0.0,
                "ENTRANCES_MODE": 0.0345,
                "FLOORSMAX_MODE": 0.0417,
                "FLOORSMIN_MODE": None,
                "LANDAREA_MODE": None,
                "LIVINGAPARTMENTS_MODE": None,
                "LIVINGAREA_MODE": 0.0073,
                "NONLIVINGAPARTMENTS_MODE": None,
                "NONLIVINGAREA_MODE": None,
                "APARTMENTS_MEDI": None,
                "BASEMENTAREA_MEDI": None,
                "YEARS_BEGINEXPLUATATION_MEDI": 0.9791,
                "YEARS_BUILD_MEDI": None,
                "COMMONAREA_MEDI": None,
                "ELEVATORS_MEDI": 0.0,
                "ENTRANCES_MEDI": 0.0345,
                "FLOORSMAX_MEDI": 0.0417,
                "FLOORSMIN_MEDI": None,
                "LANDAREA_MEDI": None,
                "LIVINGAPARTMENTS_MEDI": None,
                "LIVINGAREA_MEDI": 0.0071,
                "NONLIVINGAPARTMENTS_MEDI": None,
                "NONLIVINGAREA_MEDI": None,
                "FONDKAPREMONT_MODE": None,
                "HOUSETYPE_MODE": "block of flats",
                "TOTALAREA_MODE": 0.0079,
                "WALLSMATERIAL_MODE": "Stone, brick",
                "EMERGENCYSTATE_MODE": "No",
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 1.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 1.0,
                "DAYS_LAST_PHONE_CHANGE": -2011.0,
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 1,
                "FLAG_DOCUMENT_4": 0,
                "FLAG_DOCUMENT_5": 0,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_7": 0,
                "FLAG_DOCUMENT_8": 0,
                "FLAG_DOCUMENT_9": 0,
                "FLAG_DOCUMENT_10": 0,
                "FLAG_DOCUMENT_11": 0,
                "FLAG_DOCUMENT_12": 0,
                "FLAG_DOCUMENT_13": 0,
                "FLAG_DOCUMENT_14": 0,
                "FLAG_DOCUMENT_15": 0,
                "FLAG_DOCUMENT_16": 0,
                "FLAG_DOCUMENT_17": 0,
                "FLAG_DOCUMENT_18": 0,
                "FLAG_DOCUMENT_19": 0,
                "FLAG_DOCUMENT_20": 0,
                "FLAG_DOCUMENT_21": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": None,
                "AMT_REQ_CREDIT_BUREAU_DAY": None,
                "AMT_REQ_CREDIT_BUREAU_WEEK": None,
                "AMT_REQ_CREDIT_BUREAU_MON": None,
                "AMT_REQ_CREDIT_BUREAU_QRT": None,
                "AMT_REQ_CREDIT_BUREAU_YEAR": None,
            },
            "2": {
                "SK_ID_CURR": 148605,
                "NAME_CONTRACT_TYPE": "Revolving loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 450000.0,
                "AMT_CREDIT": 675000.0,
                "AMT_ANNUITY": 33750.0,
                "AMT_GOODS_PRICE": None,
                "NAME_TYPE_SUITE": None,
                "NAME_INCOME_TYPE": "Commercial associate",
                "NAME_EDUCATION_TYPE": "Lower secondary",
                "NAME_FAMILY_STATUS": "Unknown",
                "NAME_HOUSING_TYPE": "Municipal apartment",
                "REGION_POPULATION_RELATIVE": 0.015221,
                "DAYS_BIRTH": -12396,
                "DAYS_EMPLOYED": -1161,
                "DAYS_REGISTRATION": -3265.0,
                "DAYS_ID_PUBLISH": -4489,
                "OWN_CAR_AGE": None,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 1,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,
                "FLAG_EMAIL": 0,
                "OCCUPATION_TYPE": "Managers",
                "CNT_FAM_MEMBERS": None,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "WEEKDAY_APPR_PROCESS_START": "THURSDAY",
                "HOUR_APPR_PROCESS_START": 15,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 1,
                "LIVE_REGION_NOT_WORK_REGION": 1,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 1,
                "LIVE_CITY_NOT_WORK_CITY": 1,
                "ORGANIZATION_TYPE": "Insurance",
                "EXT_SOURCE_1": 0.6285635739,
                "EXT_SOURCE_2": 0.7006176962,
                "EXT_SOURCE_3": None,
                "APARTMENTS_AVG": None,
                "BASEMENTAREA_AVG": None,
                "YEARS_BEGINEXPLUATATION_AVG": None,
                "YEARS_BUILD_AVG": None,
                "COMMONAREA_AVG": None,
                "ELEVATORS_AVG": None,
                "ENTRANCES_AVG": None,
                "FLOORSMAX_AVG": None,
                "FLOORSMIN_AVG": None,
                "LANDAREA_AVG": None,
                "LIVINGAPARTMENTS_AVG": None,
                "LIVINGAREA_AVG": None,
                "NONLIVINGAPARTMENTS_AVG": None,
                "NONLIVINGAREA_AVG": None,
                "APARTMENTS_MODE": None,
                "BASEMENTAREA_MODE": None,
                "YEARS_BEGINEXPLUATATION_MODE": None,
                "YEARS_BUILD_MODE": None,
                "COMMONAREA_MODE": None,
                "ELEVATORS_MODE": None,
                "ENTRANCES_MODE": None,
                "FLOORSMAX_MODE": None,
                "FLOORSMIN_MODE": None,
                "LANDAREA_MODE": None,
                "LIVINGAPARTMENTS_MODE": None,
                "LIVINGAREA_MODE": None,
                "NONLIVINGAPARTMENTS_MODE": None,
                "NONLIVINGAREA_MODE": None,
                "APARTMENTS_MEDI": None,
                "BASEMENTAREA_MEDI": None,
                "YEARS_BEGINEXPLUATATION_MEDI": None,
                "YEARS_BUILD_MEDI": None,
                "COMMONAREA_MEDI": None,
                "ELEVATORS_MEDI": None,
                "ENTRANCES_MEDI": None,
                "FLOORSMAX_MEDI": None,
                "FLOORSMIN_MEDI": None,
                "LANDAREA_MEDI": None,
                "LIVINGAPARTMENTS_MEDI": None,
                "LIVINGAREA_MEDI": None,
                "NONLIVINGAPARTMENTS_MEDI": None,
                "NONLIVINGAREA_MEDI": None,
                "FONDKAPREMONT_MODE": None,
                "HOUSETYPE_MODE": None,
                "TOTALAREA_MODE": None,
                "WALLSMATERIAL_MODE": None,
                "EMERGENCYSTATE_MODE": None,
                "OBS_30_CNT_SOCIAL_CIRCLE": 3.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 3.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                "DAYS_LAST_PHONE_CHANGE": -876.0,
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 0,
                "FLAG_DOCUMENT_4": 0,
                "FLAG_DOCUMENT_5": 1,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_7": 0,
                "FLAG_DOCUMENT_8": 0,
                "FLAG_DOCUMENT_9": 0,
                "FLAG_DOCUMENT_10": 0,
                "FLAG_DOCUMENT_11": 0,
                "FLAG_DOCUMENT_12": 0,
                "FLAG_DOCUMENT_13": 0,
                "FLAG_DOCUMENT_14": 0,
                "FLAG_DOCUMENT_15": 0,
                "FLAG_DOCUMENT_16": 0,
                "FLAG_DOCUMENT_17": 0,
                "FLAG_DOCUMENT_18": 0,
                "FLAG_DOCUMENT_19": 0,
                "FLAG_DOCUMENT_20": 0,
                "FLAG_DOCUMENT_21": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": None,
                "AMT_REQ_CREDIT_BUREAU_DAY": None,
                "AMT_REQ_CREDIT_BUREAU_WEEK": None,
                "AMT_REQ_CREDIT_BUREAU_MON": None,
                "AMT_REQ_CREDIT_BUREAU_QRT": None,
                "AMT_REQ_CREDIT_BUREAU_YEAR": None,
            },
            "3": {
                "SK_ID_CURR": 100002,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary \\/ secondary special",
                "NAME_FAMILY_STATUS": "Single \\/ not married",
                "NAME_HOUSING_TYPE": "House \\/ apartment",
                "REGION_POPULATION_RELATIVE": 0.018801,
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "DAYS_REGISTRATION": -3648.0,
                "DAYS_ID_PUBLISH": -2120,
                "OWN_CAR_AGE": None,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 0,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,
                "FLAG_EMAIL": 0,
                "OCCUPATION_TYPE": "Laborers",
                "CNT_FAM_MEMBERS": 1.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                "HOUR_APPR_PROCESS_START": 10,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 0,
                "LIVE_REGION_NOT_WORK_REGION": 0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 0,
                "LIVE_CITY_NOT_WORK_CITY": 0,
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "EXT_SOURCE_1": 0.0830369674,
                "EXT_SOURCE_2": 0.2629485927,
                "EXT_SOURCE_3": 0.1393757801,
                "APARTMENTS_AVG": 0.0247,
                "BASEMENTAREA_AVG": 0.0369,
                "YEARS_BEGINEXPLUATATION_AVG": 0.9722,
                "YEARS_BUILD_AVG": 0.6192,
                "COMMONAREA_AVG": 0.0143,
                "ELEVATORS_AVG": 0.0,
                "ENTRANCES_AVG": 0.069,
                "FLOORSMAX_AVG": 0.0833,
                "FLOORSMIN_AVG": 0.125,
                "LANDAREA_AVG": 0.0369,
                "LIVINGAPARTMENTS_AVG": 0.0202,
                "LIVINGAREA_AVG": 0.019,
                "NONLIVINGAPARTMENTS_AVG": 0.0,
                "NONLIVINGAREA_AVG": 0.0,
                "APARTMENTS_MODE": 0.0252,
                "BASEMENTAREA_MODE": 0.0383,
                "YEARS_BEGINEXPLUATATION_MODE": 0.9722,
                "YEARS_BUILD_MODE": 0.6341,
                "COMMONAREA_MODE": 0.0144,
                "ELEVATORS_MODE": 0.0,
                "ENTRANCES_MODE": 0.069,
                "FLOORSMAX_MODE": 0.0833,
                "FLOORSMIN_MODE": 0.125,
                "LANDAREA_MODE": 0.0377,
                "LIVINGAPARTMENTS_MODE": 0.022,
                "LIVINGAREA_MODE": 0.0198,
                "NONLIVINGAPARTMENTS_MODE": 0.0,
                "NONLIVINGAREA_MODE": 0.0,
                "APARTMENTS_MEDI": 0.025,
                "BASEMENTAREA_MEDI": 0.0369,
                "YEARS_BEGINEXPLUATATION_MEDI": 0.9722,
                "YEARS_BUILD_MEDI": 0.6243,
                "COMMONAREA_MEDI": 0.0144,
                "ELEVATORS_MEDI": 0.0,
                "ENTRANCES_MEDI": 0.069,
                "FLOORSMAX_MEDI": 0.0833,
                "FLOORSMIN_MEDI": 0.125,
                "LANDAREA_MEDI": 0.0375,
                "LIVINGAPARTMENTS_MEDI": 0.0205,
                "LIVINGAREA_MEDI": 0.0193,
                "NONLIVINGAPARTMENTS_MEDI": 0.0,
                "NONLIVINGAREA_MEDI": 0.0,
                "FONDKAPREMONT_MODE": "reg oper account",
                "HOUSETYPE_MODE": "block of flats",
                "TOTALAREA_MODE": 0.0149,
                "WALLSMATERIAL_MODE": "Stone, brick",
                "EMERGENCYSTATE_MODE": "No",
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DAYS_LAST_PHONE_CHANGE": -1134.0,
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 1,
                "FLAG_DOCUMENT_4": 0,
                "FLAG_DOCUMENT_5": 0,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_7": 0,
                "FLAG_DOCUMENT_8": 0,
                "FLAG_DOCUMENT_9": 0,
                "FLAG_DOCUMENT_10": 0,
                "FLAG_DOCUMENT_11": 0,
                "FLAG_DOCUMENT_12": 0,
                "FLAG_DOCUMENT_13": 0,
                "FLAG_DOCUMENT_14": 0,
                "FLAG_DOCUMENT_15": 0,
                "FLAG_DOCUMENT_16": 0,
                "FLAG_DOCUMENT_17": 0,
                "FLAG_DOCUMENT_18": 0,
                "FLAG_DOCUMENT_19": 0,
                "FLAG_DOCUMENT_20": 0,
                "FLAG_DOCUMENT_21": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
            },
        },
        orient="index",
    )


@pytest.fixture()
def application_data_y():
    return pd.DataFrame.from_dict(
        {
            "0": {"TARGET": 1},
            "1": {"TARGET": 1},
            "2": {"TARGET": 0},
            "3": {"TARGET": 0},
        },
        orient="index",
    )


@pytest.fixture()
def expected_data_X():
    return pd.DataFrame.from_dict(
        {
            "0": {
                "SK_ID_CURR": 100002,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary \\/ secondary special",
                "NAME_FAMILY_STATUS": "Single \\/ not married",
                "NAME_HOUSING_TYPE": "House \\/ apartment",
                "REGION_POPULATION_RELATIVE": 0.018801,
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "DAYS_REGISTRATION": -3648.0,
                "DAYS_ID_PUBLISH": -2120,
                "OWN_CAR_AGE": None,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 0,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,
                "FLAG_EMAIL": 0,
                "OCCUPATION_TYPE": "Laborers",
                "CNT_FAM_MEMBERS": 1.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                "HOUR_APPR_PROCESS_START": 10,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 0,
                "LIVE_REGION_NOT_WORK_REGION": 0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 0,
                "LIVE_CITY_NOT_WORK_CITY": 0,
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "EXT_SOURCE_1": 0.0830369674,
                "EXT_SOURCE_2": 0.2629485927,
                "EXT_SOURCE_3": 0.1393757801,
                "APARTMENTS_AVG": 0.0247,
                "BASEMENTAREA_AVG": 0.0369,
                "YEARS_BEGINEXPLUATATION_AVG": 0.9722,
                "YEARS_BUILD_AVG": 0.6192,
                "COMMONAREA_AVG": 0.0143,
                "ELEVATORS_AVG": 0.0,
                "ENTRANCES_AVG": 0.069,
                "FLOORSMAX_AVG": 0.0833,
                "FLOORSMIN_AVG": 0.125,
                "LANDAREA_AVG": 0.0369,
                "LIVINGAPARTMENTS_AVG": 0.0202,
                "LIVINGAREA_AVG": 0.019,
                "NONLIVINGAPARTMENTS_AVG": 0.0,
                "NONLIVINGAREA_AVG": 0.0,
                "APARTMENTS_MODE": 0.0252,
                "BASEMENTAREA_MODE": 0.0383,
                "YEARS_BEGINEXPLUATATION_MODE": 0.9722,
                "YEARS_BUILD_MODE": 0.6341,
                "COMMONAREA_MODE": 0.0144,
                "ELEVATORS_MODE": 0.0,
                "ENTRANCES_MODE": 0.069,
                "FLOORSMAX_MODE": 0.0833,
                "FLOORSMIN_MODE": 0.125,
                "LANDAREA_MODE": 0.0377,
                "LIVINGAPARTMENTS_MODE": 0.022,
                "LIVINGAREA_MODE": 0.0198,
                "NONLIVINGAPARTMENTS_MODE": 0.0,
                "NONLIVINGAREA_MODE": 0.0,
                "APARTMENTS_MEDI": 0.025,
                "BASEMENTAREA_MEDI": 0.0369,
                "YEARS_BEGINEXPLUATATION_MEDI": 0.9722,
                "YEARS_BUILD_MEDI": 0.6243,
                "COMMONAREA_MEDI": 0.0144,
                "ELEVATORS_MEDI": 0.0,
                "ENTRANCES_MEDI": 0.069,
                "FLOORSMAX_MEDI": 0.0833,
                "FLOORSMIN_MEDI": 0.125,
                "LANDAREA_MEDI": 0.0375,
                "LIVINGAPARTMENTS_MEDI": 0.0205,
                "LIVINGAREA_MEDI": 0.0193,
                "NONLIVINGAPARTMENTS_MEDI": 0.0,
                "NONLIVINGAREA_MEDI": 0.0,
                "FONDKAPREMONT_MODE": "reg oper account",
                "HOUSETYPE_MODE": "block of flats",
                "TOTALAREA_MODE": 0.0149,
                "WALLSMATERIAL_MODE": "Stone, brick",
                "EMERGENCYSTATE_MODE": "No",
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DAYS_LAST_PHONE_CHANGE": -1134.0,
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 1,
                "FLAG_DOCUMENT_4": 0,
                "FLAG_DOCUMENT_5": 0,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_7": 0,
                "FLAG_DOCUMENT_8": 0,
                "FLAG_DOCUMENT_9": 0,
                "FLAG_DOCUMENT_10": 0,
                "FLAG_DOCUMENT_11": 0,
                "FLAG_DOCUMENT_12": 0,
                "FLAG_DOCUMENT_13": 0,
                "FLAG_DOCUMENT_14": 0,
                "FLAG_DOCUMENT_15": 0,
                "FLAG_DOCUMENT_16": 0,
                "FLAG_DOCUMENT_17": 0,
                "FLAG_DOCUMENT_18": 0,
                "FLAG_DOCUMENT_19": 0,
                "FLAG_DOCUMENT_20": 0,
                "FLAG_DOCUMENT_21": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
            },
        },
        orient="index",
    ).reset_index(drop=True)


@pytest.fixture()
def expected_data_y():
    return pd.DataFrame.from_dict(
        {"0": {"TARGET": 0}}, orient="index"
    ).reset_index(drop=True)


def test_transform(
    application_data_X, application_data_y, expected_data_X, expected_data_y
):
    transformer = RowsWithMissingDataDropper()
    actual_data_X, actual_data_y = transformer.transform(
        X=application_data_X, y=application_data_y
    )
    pd.testing.assert_frame_equal(actual_data_X, expected_data_X)
    pd.testing.assert_frame_equal(actual_data_y, expected_data_y)

