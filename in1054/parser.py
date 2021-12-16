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
    data[index - 7] = splitted_line[index]

  frame_vector = [timestamp,
                  id,
                  dlc,
                  data[0],
                  data[1],
                  data[2],
                  data[3],
                  data[4],
                  data[5],
                  data[6],
                  data[7],
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