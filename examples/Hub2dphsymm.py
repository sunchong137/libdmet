import libdmet.utils.logger as log
import libdmet.dmet.HubPhSymm as dmet 
from libdmet.solver import block
import numpy as np
from copy import deepcopy

block.Block.nproc = 4
log.verbose = "INFO"

U = 4
LatSize = [20, 20]
ImpSize = [4,4]
MaxIter = 1
M = 400

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

pdm = block.readpdm("/tmp/BLOCKwHVpr4/onepdm.0.0.txt")
onepdm = np.empty((2,32,32))
onepdm[0] = pdm[::2, ::2]
onepdm[1] = pdm[1::2, 1::2]

vcor = dmet.InitGuess(ImpSize, U, 1.)

for iter in range(MaxIter):
    log.result("Vcor =\n%s", vcor.get())
    rho, mu = dmet.HartreeFock(Lat, vcor, U)
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    #onepdm, E = dmet.SolveImpHam(ImpHam, basis, M)
    log.verbose = "DEBUG2"
    vcor_new = deepcopy(vcor)
    vcor_new, _ = dmet.slater.FitVcorEmb(onepdm, Lat, basis, vcor_new, np.inf)
    vcor_new, _ = dmet.slater.FitVcorFull(onepdm, Lat, basis, vcor_new, np.inf)
    log.verbose = "INFO"
    vcor = vcor_new
