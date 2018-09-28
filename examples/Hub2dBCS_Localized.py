import libdmet.utils.logger as log
import libdmet.dmet.HubbardBCS as dmet
import numpy as np
import numpy.linalg as la


#log.verbose = "INFO"
log.verbose = "DEBUG0"

U = 4
LatSize = [36, 36]
ImpSize = [2, 2]
#Filling = 0.875/2.0
Filling = 0.8/2.0
MaxIter = 50
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

block = dmet.impurity_solver.StackBlock(nproc = 1, nthread = 28, nnode = 1, \
        bcs = True, tol = 1e-7, maxM = 400)

solver = dmet.impurity_solver.BCSDmrgCI(ncas = 8, \
        cisolver = block, splitloc = True, algo = "energy", doscf = True)

log.section("\nfitting chemical potential\n")
_, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section ("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    GRho, Mu, Res = dmet.HartreeFockBogoliubov_full(Lat, vcor, None, Mu)


    log.section("\nconstructing impurity problem\n")
    ImpHam, H_energy, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu)
    log.section("\nsolving impurity problem\n")
    
    # scf
    GRhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis})
    
    #print Res["e"].shape
    #print Res["coef"].shape
    #GRhoEmb, EnergyEmb, ImpHam, dmu = \
    #        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
    #        solver_args = {"guess": (Res["e"], Res["coef"]), "basis": basis})
    
            
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    GRhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, H_energy, dmu)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
            MaxIter1 = max(len(vcor.param) * 20, 3000), MaxIter2 = 0)

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

    if np.max(abs(vcor.get() - vcor_new.get())) < 1.0e-4:
        conv = True
        break

    if not conv:
        #skipDiis = (iter < DiisStart) and (np.max(np.abs(vcor_new.param - vcor.param)) > 0.01)
        skipDiis = (iter < DiisStart) and (np.linalg.norm(vcor_new.param - vcor.param) > len(vcor.param) * 0.01)
        pvcor, dpvcor, _ = dc.Apply( \
                np.hstack((vcor_new.param, Mu_new)), \
                np.hstack((vcor_new.param - vcor.param, Mu_new - Mu)), \
                Skip = skipDiis)
        vcor.update(pvcor[:-1])
        Mu = pvcor[-1]

#solver.cleanup()

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
