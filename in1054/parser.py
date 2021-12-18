import pandas as pd

import in1054.constants as consts

def load_txt_file(filename):
  file = open(filename, 'r')
  contents = file.readlines()

  return contents


def convert_line_to_frame_vector(line):
  splitted_line = line.split()
  n_of_values = len(splitted_line)

  timestamp = splitted_line[1]
  id = splitted_line[3]
  dlc = splitted_line[6]
  data = ["00", "00", "00", "00", "00", "00", "00", "00"]
  for index in range(7, n_of_values):
    data[index - 7] = "0x" + splitted_line[index]

  frame_vector = [timestamp,
                  int(id, 16),
                  dlc,
                  int(data[0], 16),
                  int(data[1], 16),
                  int(data[2], 16),
                  int(data[3], 16),
                  int(data[4], 16),
                  int(data[5], 16),
                  int(data[6], 16),
                  int(data[7], 16),
                  consts.REGULAR_FLAG]

  return frame_vector


def parse(filename, output_filepath):
  contents = load_txt_file(filename)
  contents_df = pd.DataFrame(columns=consts.COLUMNS_NAMES)

  for line in contents:
    frame_vector = convert_line_to_frame_vector(line)
    frame_df = pd.DataFrame([frame_vector], columns=consts.COLUMNS_NAMES)
    contents_df = contents_df.append(frame_df, ignore_index=True)

  contents_df.to_csv(output_filepath, index=False)