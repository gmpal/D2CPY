import subprocess
import os

# Navigate to the 'generation' directory from the project root
os.chdir('generation')

# # Run the first Python script with specified arguments
subprocess.run(['python', 'generateTS.py', '--n_series=10', '--n_observations=150', '--name=deleteme', '--n_jobs=10'])

# # Run the second Python script with specified arguments
subprocess.run(['python', 'd2c_past_gen.py', '--name=deleteme', '--maxlags=3', '--n_jobs=10'])

# Navigate to the 'competitors' directory from the project root
os.chdir('../competitors')

# Run the third Python script with specified arguments
subprocess.run(['python', 'test.py', '--name=deleteme', '--maxlags=3', '--n_jobs=10'])
