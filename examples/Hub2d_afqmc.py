import libdmet.utils.logger as log
import libdmet.dmet.HubPhSymm as dmet 
import numpy as np
import numpy.linalg as la

log.verbose = "DEBUG1"

U = 4
LatSize = [72, 72]
ImpSize = [2,2]
MaxIter = 20
DiisStart = 4
DiisDim = 4

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

vcor = dmet.InitGuess(ImpSize, U, 1.)
dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

dmet.impurity_solver.AFQMC.settings["meas_sweep"] = 200
solver = dmet.impurity_solver.AFQMC(nproc = 4, nnode = 1, \
        TmpDir = "/scratch/boxiao/DMETTemp")

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    rho, mu = dmet.HartreeFock(Lat, vcor, U)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb = solver.run(ImpHam)
    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)
    nImp = np.product(ImpSize)
    log.result("Spin Correlation Function (impurity):\n%s", \
            solver.spin_corr()[:nImp, :nImp])

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, vcor, np.inf)
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
