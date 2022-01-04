FLAG_COLUMN_NAME = "flag"
ID_COLUMN_NAME = "id"
DLC_COLUMN_NAME = "dlc"
DATA_0_COLUMN_NAME = "data[0]"
DATA_1_COLUMN_NAME = "data[1]"
DATA_2_COLUMN_NAME = "data[2]"
DATA_3_COLUMN_NAME = "data[3]"
DATA_4_COLUMN_NAME = "data[4]"
DATA_5_COLUMN_NAME = "data[5]"
DATA_6_COLUMN_NAME = "data[6]"
DATA_7_COLUMN_NAME = "data[7]"

COLUMNS_NAMES = ["timestamp",
                 ID_COLUMN_NAME,
                 DLC_COLUMN_NAME,
                 DATA_0_COLUMN_NAME,
                 DATA_1_COLUMN_NAME,
                 DATA_2_COLUMN_NAME,
                 DATA_3_COLUMN_NAME,
                 DATA_4_COLUMN_NAME,
                 DATA_5_COLUMN_NAME,
                 DATA_6_COLUMN_NAME,
                 DATA_7_COLUMN_NAME,
                 FLAG_COLUMN_NAME]

COLUMNS_TO_CONVERT = [
                ID_COLUMN_NAME,
                DATA_0_COLUMN_NAME,
                DATA_1_COLUMN_NAME,
                DATA_2_COLUMN_NAME,
                DATA_3_COLUMN_NAME,
                DATA_4_COLUMN_NAME,
                DATA_5_COLUMN_NAME,
                DATA_6_COLUMN_NAME,
                DATA_7_COLUMN_NAME
]

METRICS_COLUMNS = [
                "accuracy",
                "precision",
                "recall",
                "f1 score"
                ]

# if regular -> REGULAR_FLAG = 0
# if anomalous -> REGULAR_FLAG = 1
REGULAR_FLAG_INT = 0
INJECTED_FLAG_INT = 1

REGULAR_FLAG_STR = "R"
INJECTED_FLAG_STR = "T"

CONCATENATED_FEATURES_COLUMN_NAME = "concatenated_features"

NUMBER_OF_TRAINING_SAMPLES = 250000
NUMBER_OF_REGULAR_VALIDATION_SAMPLES = 25000
NUMBER_OF_INJECTED_VALIDATION_SAMPLES = NUMBER_OF_REGULAR_VALIDATION_SAMPLES

ROOT_PATH = "/home/luigiluz/Documents/cin/github/in1054-project"
NORMAL_RUN_DATA_TXT_PATH = ROOT_PATH + "/data/normal_run_data.txt"
NORMAL_RUN_DATA_CSV_PATH = ROOT_PATH + "/data/normal_run_data.csv"
NORMAL_RUN_DATA_JOINED_PATH = ROOT_PATH + "/data/normal_run_data_joined.csv"
DOS_DATA_CSV_PATH = ROOT_PATH + "/data/DoS_dataset.csv"
