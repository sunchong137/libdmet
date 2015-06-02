import libdmet.utils.logger as log
import libdmet.dmet.Hubbard as dmet
from libdmet.solver import block
import numpy as np
import numpy.linalg as la
from libdmet.routine import slater

block.Block.set_nproc(4)
dmet.solver.createTmp()
log.verbose = "WARNING"

U = 4
LatSize = [12, 12]
ImpSize = [4, 4]
Filling = 0.4
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

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)

    rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)
    log.section("\nconstructing impurity problem\n")
    #ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    log.result("Making embedding basis")
    basis = slater.embBasis(Lat, rho, local = False)
    basis[:,:,:,:16] = dmet.basisMatching(basis[:,:,:,:16])
    basis[:,:,:,16:] = dmet.basisMatching(basis[:,:,:,16:])
    log.result("Constructing impurity Hamiltonian")
    ImpHam, H1e = slater.embHam(Lat, basis, vcor, local = False)

    H2 = ImpHam.H2["ccdd"][0] + ImpHam.H2["ccdd"][1] + 2 * ImpHam.H2["ccdd"][2]
    from libdmet.routine.localizer import Localizer
    Loc = Localizer(H2[:16,:16,:16,:16])
    Loc.optimize(1e-6)
    for i in range(16):
        print Loc.Int2e[i,i,i,i]
    #basis[:,:,:,4:] = np.tensordot(basis[:,:,:,:4], Loc.coefs, axes = (3, 1))
    #print basis[0,:,:,0].ravel()[:20]
    #print basis[0,:,:,1].ravel()[:20]
    #print basis[0,:,:,2].ravel()[:20]
    #print basis[0,:,:,3].ravel()[:20]
    #print basis[0,:,:,4].ravel()[:20]
    #print basis[0,:,:,5].ravel()[:20]
    #print basis[0,:,:,6].ravel()[:20]
    assert(0)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb, ImpHam, dmu = dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, M)
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    rhoImp, EnergyImp, nelecImp = dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, vcor, np.inf, Filling, MaxIter2 = 0)

    log.section("\nfitting chemical potential\n")
    rho, Mu_new = dmet.HartreeFock(Lat, vcor_new, Filling, Mu)
    log.result("dMu = %20.12f", Mu_new - Mu)
    if iter >= TraceStart: # we want to avoid spiral increase of vcor and mu
        log.result("dMu applied to vcor")
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
