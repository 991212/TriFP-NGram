import glob

def merge_csv(output_file, input_files):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in input_files:
            with open(fname, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

# get all csv files
input_files = glob.glob('data*.csv')

# merge files
merge_csv('./merged_data.csv', sorted(input_files))