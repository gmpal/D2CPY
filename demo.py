import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn import datasets, linear_model

import numpy as np
from d2c import simulatedDAG

np.random.seed(0)

noNodes = [5, 20]
N = [50, 100]
sd_noise = [0.1, 0.25]
NDAG = 15
type = "is.parent"

trainDAG = simulatedDAG(
    NDAG=NDAG,
    N=N,
    noNodes=noNodes,
    functionType=["linear", "quadratic", "sigmoid"],
    seed=1,
    sdn=0,
    additive=[True, False],
    verbose=True,
    maxV=3,
    weights=[0.5, 1]
)
