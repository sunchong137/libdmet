import libdmet.utils.logger as log
import libdmet.dmet.Hubbard as dmet
from libdmet.solver import block
import numpy as np
import numpy.linalg as la

block.Block.set_nproc(4)
log.verbose = "WARNING"

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

vcor = dmet.InitGuess(ImpSize, U, Filling)
Mu = U * Filling
dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

log.section("\nsolving mean-field problem\n")
log.result("Vcor =\n%s", vcor.get())
log.result("Mu (guess) = %20.12f", Mu)
rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    if iter >= TraceStart: # we want to avoid spiral increase of vcor and mu
        vcor = dmet.addDiag(vcor, Mu - Mu_new)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb, ImpHam, dmu = dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, M)
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    rhoImp, EnergyImp, nelecImp = dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, vcor, np.inf, Filling, MaxIter2 = 0)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor_new.get())
    log.result("Mu (guess) = %20.12f", Mu)
    rho, Mu_new = dmet.HartreeFock(Lat, vcor_new, Filling, Mu)
    if iter >= TraceStart: # we want to avoid spiral increase of vcor and mu
        vcor_new = dmet.addDiag(vcor_new, Mu - Mu_new)
    else:
        Mu = Mu_new

    history.update(EnergyImp, err, nelecImp, np.max(abs(vcor.get() - vcor_new.get())), dc)

    if np.max(abs(vcor.get() - vcor_new.get())) < 1e-5:
        conv = True
        break
    # DIIS
    if not conv:
        skipDiis = (iter < DiisStart) and (la.norm(vcor_new.param - vcor.param) > 0.01)
        pvcor, _, _ = dc.Apply(vcor_new.param, vcor_new.param - vcor.param, Skip = skipDiis)
        vcor.update(pvcor)

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
