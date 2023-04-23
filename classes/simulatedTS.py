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
            




# setGeneric("update", def=function(object,...) {standardGeneric("update")})

# #' update of a "simulatedDAG" with a list of DAGs and associated observations
# #' @param object :  simulatedDAG to be updated
# #' @param list.DAGs : list of stored DAGs
# #' @param list.observationsDAGs : list of observed datasets, each sampled from the corresponding member of list.DAGs
# #' @export
# setMethod(f="update",
#           signature="simulatedDAG",
#           definition=function(object,list.DAGs,list.observationsDAGs) {
#             if (length(list.DAGs)!=length(list.observationsDAGs))
#               stop("Lists with different lengths !")
#             object@list.DAGs=c(object@list.DAGs,list.DAGs)
#             object@list.observationsDAGs=c(object@list.observationsDAGs,list.observationsDAGs)
#             object@NDAG=length(object@list.DAGs)
#             object
            
#           }
# )

