#!/bin/bash

# Navigate to the 'generation' directory from the project root
cd generation

# Run the first Python script with specified arguments
python ./generateTS.py --n_series=1000 --n_observations=150 --name=mixed --n_jobs=50 --function_types sigmoid linear quadratic exponential tanh polynomial


# Run the second Python script with specified arguments
python ./d2c_past_gen.py --name=mixed --maxlags=3 --n_jobs=50

# Navigate to the 'competitors' directory from the project root
cd ../competitors

# Run the third Python script with specified arguments
python test.py --name=mixed --maxlags=3 --n_jobs=50


# Navigate to the 'generation' directory from the project root
cd ../generation

# Run the first Python script with specified arguments
python ./generateTS.py --n_series=1000 --n_observations=150 --name=linear --n_jobs=50 --function_types linear


# Run the second Python script with specified arguments
python ./d2c_past_gen.py --name=linear --maxlags=3 --n_jobs=50

# Navigate to the 'competitors' directory from the project root
cd ../competitors

# Run the third Python script with specified arguments
python test.py --name=linear --maxlags=3 --n_jobs=50

# Navigate to the 'generation' directory from the project root
cd ../generation

# Run the first Python script with specified arguments
python ./generateTS.py --n_series=1000 --n_observations=150 --name=quadratic --n_jobs=50 --function_types quadratic


# Run the second Python script with specified arguments
python ./d2c_past_gen.py --name=quadratic --maxlags=3 --n_jobs=50

# Navigate to the 'competitors' directory from the project root
cd ../competitors

# Run the third Python script with specified arguments
python test.py --name=quadratic --maxlags=3 --n_jobs=50


# Navigate to the 'generation' directory from the project root
cd ../generation

# Run the first Python script with specified arguments
python ./generateTS.py --n_series=1000 --n_observations=150 --name=polynomial --n_jobs=50 --function_types polynomial


# Run the second Python script with specified arguments
python ./d2c_past_gen.py --name=polynomial --maxlags=3 --n_jobs=50

# Navigate to the 'competitors' directory from the project root
cd ../competitors

# Run the third Python script with specified arguments
python test.py --name=polynomial --maxlags=3 --n_jobs=50

# Navigate to the 'generation' directory from the project root
cd ../generation

# Run the first Python script with specified arguments
python ./generateTS.py --n_series=1000 --n_observations=150 --name=tanh --n_jobs=50 --function_types tanh


# Run the second Python script with specified arguments
python ./d2c_past_gen.py --name=tanh --maxlags=3 --n_jobs=50

# Navigate to the 'competitors' directory from the project root
cd ../competitors

# Run the third Python script with specified arguments
python test.py --name=tanh --maxlags=3 --n_jobs=50