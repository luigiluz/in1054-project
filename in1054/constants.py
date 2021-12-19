FLAG_COLUMN_NAME = "flag"

COLUMNS_NAMES = ["timestamp",
                 "id",
                 "dlc",
                 "data[0]",
                 "data[1]",
                 "data[2]",
                 "data[3]",
                 "data[4]",
                 "data[5]",
                 "data[6]",
                 "data[7]",
                 FLAG_COLUMN_NAME]

# if regular -> REGULAR_FLAG = 0
# if anomalous -> REGULAR_FLAG = 1
REGULAR_FLAG = 0

ROOT_PATH = "/home/luigiluz/Documents/cin/github/in1054-project"
NORMAL_RUN_DATA_TXT_PATH = ROOT_PATH + "/data/normal_run_data.txt"
NORMAL_RUN_DATA_CSV_PATH = ROOT_PATH + "/data/normal_run_data.csv"
