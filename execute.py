import subprocess
import os

# Navigate to the 'generation' directory from the project root
os.chdir('generation')

# # # Run the first Python script with specified arguments
# # Example function_types list
# function_types_list = ['sigmoid', 'linear', 'quadratic', 'exponential', 'tanh', 'polynomial']

# # Prepare the command with function types
# command = [
#     'python', 'generateTS.py',
#     '--n_series', '10',
#     '--n_observations', '150',
#     '--name', 'mix',
#     '--n_jobs', '10',
#     '--function_types'
# ] + function_types_list

# subprocess.run(command)

# # # Run the second Python script with specified arguments
# subprocess.run(['python', 'd2c_past_gen.py', '--name=mix', '--maxlags=3', '--n_jobs=10'])

# Navigate to the 'competitors' directory from the project root
os.chdir('../competitors')

# Run the third Python script with specified arguments
subprocess.run(['python', 'test.py', '--name=mix', '--maxlags=3', '--n_jobs=50'])

##############################################################################################################
##############################################################################################################
##############################################################################################################

# for function in function_types_list:
#     os.chdir('../generation')

#     # # Run the first Python script with specified arguments
#     # Example function_types list
    

#     # Prepare the command with function types
#     command = [
#         'python', 'generateTS.py',
#         '--n_series', '10',
#         '--n_observations', '150',
#         '--name', 'only_'+function,
#         '--n_jobs', '10',
#         '--function_types'
#     ] + [function]

#     subprocess.run(command)

#     # # Run the second Python script with specified arguments
#     subprocess.run(['python', 'd2c_past_gen.py', '--name=only_'+function, '--maxlags=3', '--n_jobs=10'])

#     # Navigate to the 'competitors' directory from the project root
#     os.chdir('../competitors')

#     # Run the third Python script with specified arguments
#     subprocess.run(['python', 'test.py', '--name=only_'+function, '--maxlags=3', '--n_jobs=10'])
