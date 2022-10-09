from typing import List, Union
from pydantic import BaseModel


class ClientData(BaseModel):
    SK_ID_CURR: Union[int, None] = None
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: Union[str, None] = None
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_TYPE_SUITE: Union[str, None] = None
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: Union[int, None] = None
    OWN_CAR_AGE: float
    FLAG_MOBIL: int
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: Union[str, None] = None
    HOUR_APPR_PROCESS_START: Union[int, None] = None
    REG_REGION_NOT_LIVE_REGION: int
    REG_REGION_NOT_WORK_REGION: int
    LIVE_REGION_NOT_WORK_REGION: int
    REG_CITY_NOT_LIVE_CITY: int
    REG_CITY_NOT_WORK_CITY: int
    LIVE_CITY_NOT_WORK_CITY: int
    ORGANIZATION_TYPE: str
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    APARTMENTS_AVG: float
    BASEMENTAREA_AVG: float
    YEARS_BEGINEXPLUATATION_AVG: float
    YEARS_BUILD_AVG: float
    COMMONAREA_AVG: float
    ELEVATORS_AVG: float
    ENTRANCES_AVG: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    LANDAREA_AVG: float
    LIVINGAPARTMENTS_AVG: float
    LIVINGAREA_AVG: float
    NONLIVINGAPARTMENTS_AVG: float
    NONLIVINGAREA_AVG: float
    APARTMENTS_MODE: float
    BASEMENTAREA_MODE: float
    YEARS_BEGINEXPLUATATION_MODE: float
    YEARS_BUILD_MODE: float
    COMMONAREA_MODE: float
    ELEVATORS_MODE: float
    ENTRANCES_MODE: float
    FLOORSMAX_MODE: float
    FLOORSMIN_MODE: float
    LANDAREA_MODE: float
    LIVINGAPARTMENTS_MODE: float
    LIVINGAREA_MODE: float
    NONLIVINGAPARTMENTS_MODE: float
    NONLIVINGAREA_MODE: float
    APARTMENTS_MEDI: float
    BASEMENTAREA_MEDI: float
    YEARS_BEGINEXPLUATATION_MEDI: float
    YEARS_BUILD_MEDI: float
    COMMONAREA_MEDI: float
    ELEVATORS_MEDI: float
    ENTRANCES_MEDI: float
    FLOORSMAX_MEDI: float
    FLOORSMIN_MEDI: float
    LANDAREA_MEDI: float
    LIVINGAPARTMENTS_MEDI: float
    LIVINGAREA_MEDI: float
    NONLIVINGAPARTMENTS_MEDI: float
    NONLIVINGAREA_MEDI: float
    FONDKAPREMONT_MODE: str
    HOUSETYPE_MODE: str
    TOTALAREA_MODE: float
    WALLSMATERIAL_MODE: str
    EMERGENCYSTATE_MODE: str
    OBS_30_CNT_SOCIAL_CIRCLE: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: Union[float, None] = None
    FLAG_DOCUMENT_2: int
    FLAG_DOCUMENT_3: int
    FLAG_DOCUMENT_4: int
    FLAG_DOCUMENT_5: int
    FLAG_DOCUMENT_6: int
    FLAG_DOCUMENT_7: int
    FLAG_DOCUMENT_8: int
    FLAG_DOCUMENT_9: int
    FLAG_DOCUMENT_10: int
    FLAG_DOCUMENT_11: int
    FLAG_DOCUMENT_12: int
    FLAG_DOCUMENT_13: int
    FLAG_DOCUMENT_14: int
    FLAG_DOCUMENT_15: int
    FLAG_DOCUMENT_16: int
    FLAG_DOCUMENT_17: int
    FLAG_DOCUMENT_18: int
    FLAG_DOCUMENT_19: int
    FLAG_DOCUMENT_20: int
    FLAG_DOCUMENT_21: int
    AMT_REQ_CREDIT_BUREAU_HOUR: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_DAY: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_WEEK: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_MON: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_QRT: Union[float, None] = None
    AMT_REQ_CREDIT_BUREAU_YEAR: Union[float, None] = None


class ClientDataList(BaseModel):
    data: List[ClientData]
