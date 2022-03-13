import pybloomfilter
import pybloom_live as pybloom

import in1054.constants as consts

def _convert_bloom_filter_results(results):
	# Logical inversion of results
	# 0 means that is not in bloom filter
	# 1 means that it is in bloom filter
	inverted_results = [not elem for elem in results]
	# Now:
	# 0 means that is a regular frame
	# 1 means that is an injected frame

	# Convert boolean list to int list
	integer_map = map(int, inverted_results)
	results_as_int = list(integer_map)

	return results_as_int


def povoate_bloom_filter(normal_dataframe, bf_filename=None):
	tmp_df = normal_dataframe.copy()
	concatenated_features_array = tmp_df.loc[:, consts.CONCATENATED_FEATURES_COLUMN_NAME].values

	n_of_rows = concatenated_features_array.shape[0]
	bf_capacity = int(n_of_rows + 0.1 * n_of_rows)

	# TODO: Update filter capacity based on the amount of data used to create
	# normal state representation
	fs_bloomfilter = pybloomfilter.BloomFilter(capacity=bf_capacity, error_rate=0.01, filename=bf_filename)

	for feature in concatenated_features_array:
		fs_bloomfilter.add(feature.encode())

	return fs_bloomfilter


def check_bloomfilter(eval_dataframe, bloomfilter):
	tmp_df = eval_dataframe.copy()

	words_array = tmp_df.loc[:, consts.CONCATENATED_FEATURES_COLUMN_NAME].values
	words_array = words_array.tolist()

	bf_results = []

	for word in words_array:
		single_result = word.encode() in bloomfilter
		bf_results.append(single_result)

	results_as_int = _convert_bloom_filter_results(bf_results)

	return results_as_int


def povoate_pybloom_filter(normal_dataframe):
	concatenated_features_df = normal_dataframe.copy()
	concatenated_features_array = concatenated_features_df.loc[:, consts.CONCATENATED_FEATURES_COLUMN_NAME].values

	n_of_rows = concatenated_features_array.shape[0]
	bf_capacity = int(n_of_rows + 0.1 * n_of_rows)

	pybloom_bloomfilter = pybloom.ScalableBloomFilter(mode=pybloom.ScalableBloomFilter.SMALL_SET_GROWTH, error_rate=0.001)#BloomFilter(capacity=1000000, error_rate=0.01)

	for feature in concatenated_features_array:
		pybloom_bloomfilter.add(feature)

	return pybloom_bloomfilter


def check_pybloomfilter(eval_dataframe, pybloomfilter):
	tmp_df = eval_dataframe.copy()

	words_array = tmp_df.loc[:, consts.CONCATENATED_FEATURES_COLUMN_NAME].values
	words_array = words_array.tolist()

	bf_results = []

	for word in words_array:
		single_result = word in pybloomfilter
		bf_results.append(single_result)

	results_as_int = _convert_bloom_filter_results(bf_results)

	return results_as_int


def check_pybloomfilter_single(single_value, pybloomfilter):

	word = single_value.values
	word = word.tolist()

	bf_results = []

	single_result = word in pybloomfilter
	bf_results.append(single_result)

	results_as_int = _convert_bloom_filter_results(bf_results)

	return results_as_int


class BloomFilterDict():
  def __init__(self):
    self._bf_dict = dict()


  def __is__key__in__dict(self, my_key):
    return my_key in self._bf_dict.keys()


  def __add__key__value__pair(self, current_pair):
    current_key = current_pair[0]
    current_value = current_pair[1]

    if self.__is__key__in__dict(current_key):
      #print("Key already in dict")
      self._bf_dict[current_key].add(current_value)
    else:
      #print("Adding {} key to the dict".format(current_key))
      self._bf_dict[current_key] = pybloom.ScalableBloomFilter(mode=pybloom.ScalableBloomFilter.SMALL_SET_GROWTH)
      self._bf_dict[current_key].add(current_value)


  def povoate_dict(self, input_list):
    for list_index in range(len(input_list)):
      current_pair = input_list[list_index]
      self.__add__key__value__pair(current_pair)


  def single_query(self, query_pair):
    query_key = query_pair[0]
    query_value = query_pair[1]

    if self.__is__key__in__dict(query_key):
      if query_value in self._bf_dict[query_key]:
		# TODO: Tá na logica de que 0 é normal e 1 é ataque
        return 0
      else:
        return 1
    else:
      return 1


  def multiple_query(self, query_list):
    results_list = []

    for list_index in range(len(query_list)):
      current_query_pair = query_list[list_index]
      current_result = self.single_query(current_query_pair)
      results_list.append(current_result)

    return results_list


  def get_memory_consumption(self):
    bits_count = 0
    for my_key in self._bf_dict.keys():
      for filter in self._bf_dict[my_key].filters:
        current_filter_len = len(filter.bitarray)
        bits_count += current_filter_len

    total_bytes = bits_count / 8
    total_kbytes = total_bytes / 1024

    return total_kbytes