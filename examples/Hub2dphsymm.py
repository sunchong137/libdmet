import libdmet.utils.logger as log
import libdmet.dmet.HubPhSymm as dmet 
from libdmet.solver import block
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from libdmet.routine.diis import FDiisContext

block.Block.nproc = 4
log.verbose = "RESULT"

U = 4
LatSize = [16, 16]
ImpSize = [2,2]
MaxIter = 20
M = 400
DiisDim = 4

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.InitGuess(ImpSize, U, 1.)
dc = FDiisContext(DiisDim)

conv = False

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)
    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    rho, mu = dmet.HartreeFock(Lat, vcor, U)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb = dmet.SolveImpHam(ImpHam, basis, M)
    rhoImp, EnergyImp, nelecImp = dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, vcor, np.inf)
    if la.norm(vcor.get() - vcor_new.get()) < 1e-4:
        conv = True
        break
    # DIIS
    if not conv:
        skipDiis = (iter < 4) and (la.norm(vcor_new.param - vcor.param) > 0.01)
        pvcor, _, _ = dc.Apply(vcor_new.param, vcor_new.param - vcor.param, Skip = skipDiis)
        vcor.update(pvcor)
