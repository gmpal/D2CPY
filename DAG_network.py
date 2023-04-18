from igraph import Graph
from random import seed, uniform, choice
from numpy.random import normal

class DAG_network:
    def __init__(self, network, sdn=0.5, exosdn=1, sigma=None, H=None, additive=True, weights=[0.8, 2], maxV=5, seed=None):
        self.network = network
        self.additive = additive
        self.maxV = maxV
        self.exosdn = exosdn
        
        if not isinstance(self.network, Graph):
            raise TypeError("network should be of class igraph.Graph")
        else:
            self.network.vs["bias"] = 0
            self.network.vs["sigma"] = sigma if sigma is not None else lambda x: normal(0, sdn)
            self.network.vs["seed"] = [None] * self.network.vcount()
            self.network.es["H"] = H if H is not None else lambda x: x
        
        for i in range(self.network.vcount()):
            if seed is None:
                seed_val = None
            else:
                seed_val = seed
                seed(seed_val)
            
            self.network.vs[i]["seed"] = uniform(1, 10000)
            
            def sigma_func(x):
                return normal(0, uniform(0.9*sdn, sdn))
            
            self.network.vs[i]["sigma"] = sigma_func if sigma is None else sigma
            
        for edge in self.network.es():
            weight = uniform(weights[0], weights[1]) * choice([-1, 1])
            edge["weight"] = weight
            
            if H is not None:
                Hi = choice(H)
            else:
                Hi = H
            
            edge["H"] = Hi() if Hi is not None else None


















net = DAG_network(Graph(), sdn=0.5, exosdn=1, sigma=None, H=[lambda x: H_Rn(1), lambda x: H_Rn(2)],
                  additive=True, weights=[0.8, 2], maxV=5, seed=None)
