import in1054.parser as parser

def main():
	root_path = "/home/luigiluz/Documents/cin/github/in1054-project"
	filename = root_path + "/data/normal_run_data.txt"
	output_filepath = root_path + "/data/normal_run_data.csv"
	parser.parse(filename, output_filepath)

if __name__ == "__main__":
	main()