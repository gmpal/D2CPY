#!/bin/bash

# Navigate to the 'generation' directory from the project root
cd generation

# Run the first Python script with specified arguments
python ./generateTS.py --n_series=10 --n_observations=150 --name=deleteme

# Run the second Python script with specified arguments
python ./d2c_past_gen.py --name=deleteme --maxlags=3 --n_jobs=10

# Navigate to the 'competitors' directory from the project root
cd ../competitors

# Run the third Python script with specified arguments
python test.py --name=deleteme --maxlags=3 --n_jobs=10
