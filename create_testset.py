import numpy as np 

DAGs_names = None 
savefile = True
njobs = 1 

DAGS_number = 100
filename =  "./data/testDAGs_" + str(DAGS_number)+".RData"

if (njobs > 1):
    # TODO
    pass 

for current_DAG in range(DAGS_number):
    range_number_nodes = np.arange(10, 200)

    range_number_samples = 500

    DAGS_number_test = 1000

    std = [0.2, 1]

    if current_DAG%2 == 0:
        testDAG = SimulatedDAG(nodes_number = range_number_nodes,
                               observations_number = range_number_samples,
                               noise_std = std,
                               maxpar_pc = 0.5,
                               quantize = True,
                               additive = True,
                               weights = True,
                               maxV = 10,
                               verbose = False,
                               n_jobs = njobs,
                               NDAG = DAGS_number_test,
                               seed = current_DAG)
    else:
        testDAG = SimulatedTS(nodes_number = range_number_nodes,
                               observations_number = range_number_samples,
                               noise_std = std,
                               maxpar_pc = 0.5,
                               quantize = True,
                               additive = True,
                               weights = True,
                               maxV = 10,
                               verbose = False,
                               n_jobs = njobs,
                               NDAG = DAGS_number_test,
                               seed = current_DAG)

 # TODO: save 