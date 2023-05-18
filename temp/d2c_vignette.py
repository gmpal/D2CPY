import random 
from simulatedDAG import SimulatedDAG
from d2c_descriptor import D2CDescriptor
from d2c import D2C

from temp_methods import make_model

random.seed(0)

nodes_number = [50,100]
std_noise = [0.2, 0.5]

DAGS_number_train = 20
DAGS_number_test = 10

trainDAG = SimulatedDAG(DAGS_number_train)

descriptor_example = D2CDescriptor()

trainD2C = D2C()

trainedD2C = make_model(trainD2C, classifier='RF', EErep=2)

testDAG = SimulatedDAG(DAGS_number_test)

d2c_Y_hat = None 
iamb_Y_hat = None
gs_Y_hat = None
Y_test = None

for i in range(DAGS_number_test):
    random.seed(i)
    