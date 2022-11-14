TARGET_COLUMN = "loan_status"
FULLY_PAID_STATUS = "Fully Paid"
CHARGED_OFF_STATUS = "Charged Off"

CONSTANT_COLUMNS = [
    "out_prncp",
    "out_prncp_inv",
    "policy_code",
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "pymnt_plan",
]

NULL_COLUMNS = [
    "next_pymnt_d",
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "open_acc_6m",
    "open_il_6m",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "next_pymnt_d",
]

HIGH_CARDINALITY_COLUMNS = ["url", "id", "member_id"]

# For ethical purposes
NOT_ALLOWED_COLUMNS = ["zip_code", "addr_state"]

# Let it now be dropped
# Needs further investigartion and application
# of NLP methods
DESCRIPTION_COLUMNS = ["desc", "title"]

REDUNDANT_COLUMNS = ["sub_grade"]
HIGHLY_IMBALANCED_COLUMNS = ["home_ownership", "application_type"]

DATA_TO_DROP = (
    CONSTANT_COLUMNS
    + NULL_COLUMNS
    + HIGH_CARDINALITY_COLUMNS
    + NOT_ALLOWED_COLUMNS
    + DESCRIPTION_COLUMNS
    + REDUNDANT_COLUMNS
    + HIGHLY_IMBALANCED_COLUMNS
)

COLUMNS_TO_CATEGORIZE = [
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
]

COLUMNS_TO_IMPUTE_MISSING_CATEGORY = ["emp_length"]

COLUMNS_TO_IMPUTE_0 = [
    "collections_12_mths_ex_med",
    "tot_coll_amt",
    "tot_cur_bal",
]

COLUMNS_TO_IMPUTE_MEAN = ["revol_util"]

COLUMNS_TO_EXCLUDE_FROM_MICE_IMPUTATION = [
    "emp_title",
    "earliest_cr_line",
    "last_credit_pull_d",
]
COLUMNS_TO_IMPUTE_WITH_MICE = ["total_rev_hi_lim"]

DATE_COLUMNS_TO_SPLIT = [
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "last_credit_pull_d",
]

MONTHS_TO_NUMS_MAPPING = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

COLUMNS_TO_GAUSSIAN_TRANSFORM = [
    "loan_amnt",
    "dti",
    "funded_amnt",
    "installment",
    "total_acc",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "last_pymnt_amnt",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

COLUMNS_TO_CUT_OUTLIERS = [
    "annual_inc",
    "dti",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "last_pymnt_amnt",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

COLUMNS_TO_GAUSSIAN_TRANSFORM = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "annual_inc",
    "revol_bal",
    "total_pymnt",
    "dti",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

COLUMNS_TO_Z_SCORE_STANDARDIZE = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "issue_d_month",
]

COLUMNS_TO_MIN_MAX_SCALE = [
    "issue_d_year",
    "earliest_cr_line_year",
    "last_pymnt_d_year",
    "last_credit_pull_d_year",
    "grade",
]

NUMERICAL_COLUMNS = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc",
]

CYCLICAL_CATEGORICAL_COLUMNS = [
    "earliest_cr_line_month",
    "last_pymnt_d_month",
    "last_credit_pull_d_month",
]

COLUMNS_TO_ORDINAL_ENCODING = ["grade"]

COLUMNS_TO_FREQUENCY_ENCODING = ["emp_title"]

CATEGORICAL_COLUMNS_TO_STANDARD_ENCODING = [
    "term",
    "emp_length",
    "verification_status",
    "purpose",
    "initial_list_status",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
]
