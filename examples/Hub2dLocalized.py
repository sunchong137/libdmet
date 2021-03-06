import libdmet.utils.logger as log
import libdmet.dmet.HubPhSymm as dmet 
import numpy as np
import numpy.linalg as la

log.verbose = "INFO"

U = 4
LatSize = [72, 72]
ImpSize = [4,4]
MaxIter = 1
M = 600
DiisStart = 4
DiisDim = 4

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.InitGuess(ImpSize, U, 1.)
dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

block = dmet.impurity_solver.Block(nproc = 8, nnode = 1, \
        reorder = True, tol = 1e-7)
solver = dmet.impurity_solver.DrmgCI(ncas = 32, nelecas = 32, \
        splitloc = True, cisolver = block)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    rho, mu = dmet.HartreeFock(Lat, vcor, U)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching = True)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb = solver.run(ImpHam, ci_args = {'M':400}, \
            guess = dmet.foldRho(rho, Lat, basis), basis = basis)
    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, vcor, np.inf, MaxIter2 = 0)
    history.update(EnergyImp, err, nelecImp, \
            np.max(abs(vcor.get() - vcor_new.get())), dc)

    if np.max(abs(vcor.get() - vcor_new.get())) < 1e-5:
        conv = True
        break
    # DIIS
    if not conv:
        skipDiis = (iter < DiisStart) and (la.norm(vcor_new.param - vcor.param) > 0.01)
        pvcor, _, _ = dc.Apply(vcor_new.param, \
                vcor_new.param - vcor.param, Skip = skipDiis)
        vcor.update(pvcor)

solver.cleanup()

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
