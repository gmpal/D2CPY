class D2CDescriptor:
    def __init__(self, lin=True, acc=True, struct=False, pq=[0.1, 0.25, 0.5, 0.75, 0.9],
                 bivariate=False, ns=4, boot="rank", maxs=20, diff=False, residual=False, stabD=False):
        self.lin = lin
        self.acc = acc
        self.struct = struct
        self.pq = pq
        self.bivariate = bivariate
        self.ns = ns
        self.maxs = maxs
        self.boot = boot
        self.residual = residual
        self.diff = diff
        self.stabD = stabD
