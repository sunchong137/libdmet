import libdmet.utils.logger as log
import libdmet.dmet.HubbardBCS as dmet
import numpy as np
import numpy.linalg as la

log.verbose = "DEBUG2"

U = 4
LatSize = [36, 36]
ImpSize = [2, 2]
Filling = 0.875/2
MaxIter = 20
M = 400
DiisStart = 4
TraceStart = 2
DiisDim = 4

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand = 0.001)
Mu = U * Filling
dc = dmet.FDiisContext(DiisDim) # I don't know yet whether diis needs to be changed

conv = False

history = dmet.IterHistory()

solver = dmet.impurity_solver.Block(nproc = 4, nnode = 1, \
        bcs = True, reorder = True, tol = 1e-6)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section ("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H_energy, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu)
    log.section("\nsolving impurity problem\n")
    GRhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args = {"M": 200}, delta = 0.02, step = 0.05)
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    GRhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(GRhoEmb, EnergyEmb, basis, ImpHam, H_energy)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
            MaxIter1 = 200, MaxIter2 = 0)

    log.section("\nfitting chemical potential\n")
    GRho, Mu_new = dmet.HartreeFockBogoliubov(Lat, vcor_new, Filling, Mu)
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
