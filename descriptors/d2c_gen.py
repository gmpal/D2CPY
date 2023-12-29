import argparse
import sys
import pickle


# Adjust sys.path
sys.path.append("..")
sys.path.append("../d2c/")
from d2c.d2c import D2C

def process_d2c(input_file, output_file, n_jobs=1):
    # Load data from the input file
    with open(input_file, 'rb') as f:
        observations, dags, updated_dags = pickle.load(f)

    # Create a D2C object and initialize it
    d2c = D2C(dags, observations, n_jobs=n_jobs)
    d2c.initialize()

    # Save the descriptors to the output file
    d2c.save_descriptors_df(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process D2C for given input and save descriptors to output')
    parser.add_argument('--input_file', type=str, default='../data/ts3.pkl', help='Path to the input pickle file')
    parser.add_argument('--output_file', type=str, default='../data/ts_limited_descriptors.csv', help='Path to save the descriptors CSV file')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs for parallel processing')

    args = parser.parse_args()

    process_d2c(args.input_file, args.output_file, args.n_jobs)
