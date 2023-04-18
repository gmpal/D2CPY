# TODO: this file is redundant 

from temp_methods import is_mb, is_parent, is_child, is_descendant, is_ancestor, make_model
import random 
import numpy as np 

from simulatedDAG import SimulatedDAG
from simulatedTS import SimulatedTS
from d2c_descriptor import D2Cdescriptor
from d2c import D2C

def is_what(iDAG, i, j, type):
    if type == "is_mb":
        return int(is_mb(iDAG, i, j))
    elif type == "is_parent":
        return int(is_parent(iDAG, i, j))
    elif type == "is_child":
        return int(is_child(iDAG, i, j))
    elif type == "is_descendant":
        return int(is_descendant(iDAG, i, j))
    elif type == "is_ancestor":
        return int(is_ancestor(iDAG, i, j))

random.seed(0)
number_nodes = 20

number_samples = 150

number_features = 20

max_s = 10 #TODO: rename

n_jobs = 1 

number_DAGS = 100 
number_DAGS_per_iteration = 5 

if n_jobs > 1:
    # TODO
    pass

iteration_counter = 0 
rep = 1 #TODO: rename and understand 

while iteration_counter < number_DAGS: #TODO: check < or <=
    random.seed(iteration_counter)

    if rep%2 == 0:
        train_DAG = SimulatedDAG() #TODO: check default parameters 
    else:
        train_DAG = SimulatedTS()

    descriptor = D2Cdescriptor()
    d2c = D2C()

    if iteration_counter==1:
        all_D2C = d2c #TODO: check copy
    else: 
        all_D2C = np.concatenate((all_D2C, d2c), axis=0) 
        #TODO: is it necessary to store the attributes separately? 

    iteration_counter += number_DAGS_per_iteration

    rep += 1 #TODO: what for? 
    
trained_d2c = make_model(all_D2C, classifier='RF', EErep=2) #TODO: implement
#TODO: save 


