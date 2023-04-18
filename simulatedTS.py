class SimulatedTS:
    def __init__(self, nodes_number, observations_number, noise_std, maxpar_pc, quantize, additive, weights, maxV, verbose, n_jobs, NDAG, seed):
        self.nodes_number = nodes_number
        self.observations_number = observations_number
        self.noise_std = noise_std
        self.maxpar_pc = maxpar_pc
        self.quantize = quantize
        self.additive = additive
        self.weights = weights
        self.maxV = maxV
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.NDAG = NDAG
        self.seed = seed

        self.create_testset()

    def create_testset(self):
        if self.n_jobs > 1:
            # TODO: parallelize
            pass 
    

        pass
            