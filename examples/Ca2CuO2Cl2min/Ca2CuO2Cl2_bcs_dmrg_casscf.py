import libdmet.utils.logger as log
import libdmet.dmet.abinitioBCS as dmet
import numpy as np
import numpy.linalg as la
import itertools as it
from libdmet.system import integral
import os

log.verbose = "INFO"

# control variables
MaxIter = 30
DiisStart = 6
TraceStart = 4
DiisDim = 8

# first define lattice shape, atoms and basis
Dim = 2
Cell = np.asarray([[3.869*2 , 0], [0, 3.869*2]])
Atoms = [
    (np.asarray([ 0.00000,  0.00000]), "Cu"),
    (np.asarray([ 0.25000,  0.00000]), "O" ),
    (np.asarray([ 0.00000,  0.25000]), "O" ),
    (np.asarray([ 0.00000,  0.50000]), "Cu"),
    (np.asarray([ 0.25000,  0.50000]), "O" ),
    (np.asarray([ 0.00000,  0.75000]), "O" ),
    (np.asarray([ 0.50000,  0.00000]), "Cu"),
    (np.asarray([ 0.75000,  0.00000]), "O" ),
    (np.asarray([ 0.50000,  0.25000]), "O" ),
    (np.asarray([ 0.50000,  0.50000]), "Cu"),
    (np.asarray([ 0.75000,  0.50000]), "O" ),
    (np.asarray([ 0.50000,  0.75000]), "O" ),
]
AtomicBasis = {
    "Cu": ["3d5", "4s", "4d5"],
    "O": ["2p3"],
}

# for the form of vcor
AForbs = [["Cu1_3d5", "Cu4_3d5"], ["Cu2_3d5", "Cu3_3d5"]]
PMorbs = []
doping = 0.
Filling = (2.5-0.5*doping)/5

# then lattice and impurity sizes
LatSize = np.asarray([10, 10])
ImpSize = np.asarray([1, 1])

Lat = dmet.buildLattice(LatSize, ImpSize, Cell, Atoms, AtomicBasis)
Ham = dmet.buildHamiltonian("integrals", Lat)
Lat.setHam(Ham)

vcor = dmet.VcorLocal(False, True, Lat.supercell.nsites)
fvcor = "vcor.npy"
if os.path.exists(fvcor) and 0:
    log.result("Read vcor from disk")
    with open(fvcor, "r") as f:
        vcor.update(np.load(f))
else:
    log.result("Antiferromagnetic initial guess of vcor")
    dmet.AFInitGuessOrbs(vcor, Lat, AForbs, PMorbs, shift = 0.3, polar = 0.3, \
            bogoliubov = True, rand = 0.001)
Mu = 0. # I don't know what is Mu

dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

block = dmet.impurity_solver.Block(nproc = 4, maxM = 300, bcs = True)
solver = dmet.impurity_solver.CASSCF(ncas = 12, nelecas = None, bogoliubov = True, \
        fcisolver = "DMRG", settings = {"fcisolver": block, "splitloc": True, \
        "mom_reorder": True})

log.section("\nfitting chemical potential\n")
_, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, None, Mu)
    dmet.reportOccupation(Lat, GRho[0])

    log.section("\nconstructing impurity problem\n")
    ImpHam, H_energy, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu)
    log.section("\nsolving impurity problem\n")
    GRhoEmb, EnergyEmb, ImpHam, dmu = dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "mcscf_args": {"basis": basis}})
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    GRhoImp, EnergyImp, nelecImp = dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, H_energy, dmu)
    dmet.reportOccupation(Lat, GRhoImp)
    
    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, MaxIter1 = 200, MaxIter2 = 0)

    if iter >= TraceStart:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        ddiagV = np.average(np.diagonal(\
                (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        vcor_new = dmet.addDiag(vcor_new, -ddiagV)

    log.section("\nfitting chemical potential\n")
    _, Mu_new = dmet.HartreeFockBogoliubov(Lat, vcor_new, Filling, Mu)
    log.result("dMu = %20.12f", Mu_new - Mu)

    history.update(EnergyImp, err, nelecImp, np.max(abs(vcor.get() - vcor_new.get())), dc)    
    if np.max(abs(vcor.get() - vcor_new.get())) < 1e-5:
        conv = True
        break

    if not conv:
        skipDiis = (iter < DiisStart) and (la.norm(vcor_new.param - vcor.param) > 0.01)
        pvcor, dpvcor, _ = dc.Apply( \
                np.hstack((vcor_new.param, Mu_new)), \
                np.hstack((vcor_new.param - vcor.param, Mu_new - Mu)), \
                Skip = skipDiis)
        vcor.update(pvcor[:-1])
        Mu = pvcor[-1]

    with open("vcor.npy", "w") as f:
        np.save(f, vcor_new.param)

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
