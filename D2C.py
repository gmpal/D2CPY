
class D2C:
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.orig_X = None
        self.orig_Y = None

    

    # def is_what(self, iDAG, i, j, type):
    #     if type == "is_mb":
    #         return int(is_mb(iDAG, i, j))
    #     elif type == "is_parent":
    #         return int(is_parent(iDAG, i, j))
    #     elif type == "is_child":
    #         return int(is_child(iDAG, i, j))
    #     elif type == "is_descendant":
    #         return int(is_descendant(iDAG, i, j))
    #     elif type == "is_ancestor":
    #         return int(is_ancestor(iDAG, i, j))

    # def create_trainset(self):
    #     random.seed(self.iteration_counter)
    #     self.train_DAG = SimulatedDAG()
    #     self.train_DAG.generate(self.number_nodes, self.number_samples, self.number_features, self.max_s)
    #     self.train_DAG.generate_data()
    #     self.train_DAG.generate_DAG()