import libdmet.utils.logger as log
import libdmet.dmet.HubPhSymm as dmet 
from libdmet.solver import block
from copy import deepcopy
import numpy as np

block.Block.nproc = 4
log.verbose = "INFO"

U = 4
LatSize = [36, 36]
ImpSize = [2,2]
MaxIter = 1
M = 400

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.InitGuess(ImpSize, U, 1.)

for iter in range(MaxIter):
    log.result("Vcor =\n%s", vcor.get())
    rho, mu = dmet.HartreeFock(Lat, vcor, U)
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    onepdm, E = dmet.SolveImpHam(ImpHam, basis, M)
    vcor_new, err = dmet.FitVcor(onepdm, Lat, basis, vcor, np.inf)
    vcor = vcor_new
