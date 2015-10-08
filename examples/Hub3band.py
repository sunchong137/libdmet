import libdmet.utils.logger as log
import libdmet.dmet.abinitioBCS as dmet
import numpy as np
import numpy.linalg as la
import os

log.verbose = "INFO"

Ud = 15.8 / 1.64
ed = -5.1 / 1.64
tpd = -1.
LatSize = [6, 6]
ImpSize = [2, 2]
doping = 0.
Filling = (5.-doping) / 6
MaxIter = 20
M = 400
DiisStart = 6
TraceStart = 4
DiisDim = 4

Lat = dmet.Square3Band(*(LatSize + ImpSize))
Ham = dmet.Hubbard3band(Lat, Ud, 0., ed, tpd, 0.)
Lat.setHam(Ham)
vcor = dmet.VcorLocal(False, True, Lat.supercell.nsites)
fvcor = "vcor.npy"

AFidx = [[0, 9], [3, 6]]
PMidx = []
if os.path.exists(fvcor):
    log.result("Read vcor from disk")
    vcor.update(np.load(fvcor))
else:
    log.result("Antiferromagnetic initial guess of vcor")
    dmet.AFInitGuessIdx(vcor, Lat.supercell.nsites, AFidx, PMidx, \
            shift = Ud/4., polar = Ud/4., bogoliubov = True, rand = 0.001)

Mu = Ud * Filling * 0.5 + ed
dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

block = dmet.impurity_solver.Block(nproc = 4, nnode = 1, maxM = 400, \
        reorder = True, tol = 1e-7)

solver = dmet.impurity_solver.BCSDmrgCI(ncas = 24, \
        cisolver = block, splitloc = True)

log.section("\nfitting chemical potential\n")
_, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, None, Mu)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu)
    log.section("\nsolving impurity problem\n")
    GRhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis})
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
            MaxIter1 = 200, MaxIter2 = 0)

    if iter >= TraceStart:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        ddiagV = np.average(np.diagonal(\
                (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        vcor_new = dmet.addDiag(vcor_new, -ddiagV)

    log.section("\nfitting chemical potential\n")
    _, Mu_new = dmet.HartreeFockBogoliubov(Lat, vcor_new, Filling, Mu)
    log.result("dMu = %20.12f", Mu_new - Mu)
    
    history.update(EnergyImp, err, nelecImp, \
            np.max(abs(vcor.get() - vcor_new.get())), dc)

    if np.max(abs(vcor.get() - vcor_new.get())) < 1e-5:
        conv = True
        break
    # DIIS
    if not conv:
        skipDiis = (iter < DiisStart) and (la.norm(vcor_new.param - vcor.param) > 0.01)
        pvcor, dpvcor, _ = dc.Apply( \
                np.hstack((vcor_new.param, Mu_new)), \
                np.hstack((vcor_new.param - vcor.param, Mu_new - Mu)), \
                Skip = skipDiis)
        vcor.update(pvcor[:-1])
        Mu = pvcor[-1]

solver.cleanup()

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
