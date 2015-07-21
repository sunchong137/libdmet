import libdmet.utils.logger as log
import libdmet.dmet.Hubbard as dmet
from libdmet.solver import block
import numpy as np
import numpy.linalg as la

log.verbose = "INFO"

U = 4
LatSize = [36, 36]
ImpSize = [2, 2]
Filling = 0.875/2
MaxIter = 20
M = 400
DiisStart = 4
TraceStart = 2
DiisDim = 4

ntotal = Filling * np.product(LatSize)
if ntotal - int(ntotal) > 1e-5:
    log.warning("rounded total number of electrons to integer %d", int(ntotal))
    Filling = float(int(ntotal)) / np.product(LatSize)

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.AFInitGuess(ImpSize, U, Filling)
Mu = U * Filling
dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

log.section("\nsolving mean-field problem\n")
log.result("Vcor =\n%s", vcor.get())
log.result("Mu (guess) = %20.12f", Mu)
rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)

solver = dmet.impurity_solver.Block(nproc = 2, nnode = 1, tol = 1e-6)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)

    rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)
    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args = {"M": 400})
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
            vcor, np.inf, Filling, MaxIter2 = 0)

    log.section("\nfitting chemical potential\n")
    rho, Mu_new = dmet.HartreeFock(Lat, vcor_new, Filling, Mu)
    log.result("dMu = %20.12f", Mu_new - Mu)
    if iter >= TraceStart: # we want to avoid spiral increase of vcor and mu
        log.result("dMu applied to vcor")
        vcor_new = dmet.addDiag(vcor_new, Mu - Mu_new)
    else:
        Mu = Mu_new

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
