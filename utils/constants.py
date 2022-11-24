TARGET_COLUMN = "loan_status"
FULLY_PAID_STATUS = "Fully Paid"
CHARGED_OFF_STATUS = "Charged Off"
NON_ETHICAL_COLUMNS = ["address"]
IMBALANCED_COLUMNS = ["application_type"]
COLUMNS_HIGHLY_CORRELATED_TO_DROP = ["installment"]
REDUNDANT_COLUMNS = ["sub_grade"]
COLUMNS_FROM_FUTURE = ["issue_d"]
COLUMNS_TO_REINTRODUCE = ["title"]
COLUMNS_TO_DROP = (
    NON_ETHICAL_COLUMNS
    + IMBALANCED_COLUMNS
    + COLUMNS_HIGHLY_CORRELATED_TO_DROP
    + REDUNDANT_COLUMNS
    + COLUMNS_TO_REINTRODUCE
    + COLUMNS_FROM_FUTURE
)
COLUMNS_TO_BINARIZE = ["pub_rec", "pub_rec_bankruptcies"]
DATE_COLUMNS_TO_SPLIT = ["earliest_cr_line"]
COLUMNS_TO_ORDINAL_ENCODING = ["term", "grade", "emp_length"]
COLUMNS_TO_IMPUTE_MISSING_CATEGORY = ["emp_title"]
COLUMNS_TO_IMPUTE_0 = ["pub_rec_bankruptcies", "emp_length"]
COLUMNS_TO_FREQUENCY_ENCODE = ["emp_title"]
COLUMNS_TO_MICE_IMPUTE = ["emp_length", "revol_util", "mort_acc"]
COLUMNS_TO_CYCLICAL_ENCODING = ["earliest_cr_line_month"]
SKEWED_COLUMNS = [
    "loan_amnt",
    "loan_amnt",
    "loan_amnt",
    "annual_inc",
    "dti",
    "open_acc",
]
COLUMNS_TO_STANDARDIZE = []
COLUMNS_TO_SCALE = [
    "term",
    "grade",
    "emp_length",
    "earliest_cr_line_year",
    "emp_title",
    "loan_amnt",
    "annual_inc",
    "dti",
    "open_acc",
    "mort_acc",
    "int_rate",
    "revol_bal",
    "revol_util",
    "total_acc",
    "earliest_cr_line_month_sine",
    "earliest_cr_line_month_cosine",
]
COLUMNS_TO_DROP_OUTLIERS = [
    "annual_inc",
    "dti",
    "open_acc",
    "mort_acc",
]
STANDARD_CAT_COLUMNS = [
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
]
