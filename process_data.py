import pandas as pd

def process_data(input_file, output_file):

    df = pd.read_csv(input_file, sep='::')
    df = df.sort_values('timestamp')
    df.to_csv(output_file, index=False, header=False)

if __name__ == '__main__':

    input_file = 'ml-1m/ratings.dat'
    output_file = 'ml-1m/ratings_new.dat'
    process_data(input_file, output_file)
