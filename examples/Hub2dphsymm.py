import libdmet.utils.logger as log
import libdmet.dmet.HubPhSymm as dmet 
from libdmet.solver import block
import numpy as np
from copy import deepcopy

block.Block.nproc = 4
log.verbose = "INFO"

U = 4
LatSize = [36, 36]
ImpSize = [4,4]
MaxIter = 10
M = 400

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.InitGuess(ImpSize, U, 1.)

for iter in range(MaxIter):
    rho, mu = dmet.HartreeFock(Lat, vcor, U)
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    onepdm, E = dmet.SolveImpHam(ImpHam, basis, 300)
    log.verbose = "DEBUG2"
    vcor_new = deepcopy(vcor)
    vcor_new, _ = dmet.slater.FitVcorEmb(onepdm, Lat, basis, vcor_new, np.inf)
    log.verbose = "INFO"
    vcor = vcor_new
