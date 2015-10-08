import libdmet.utils.logger as log
import libdmet.dmet.abinitio as dmet
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

vcor = dmet.VcorLocal(False, False, Lat.supercell.nsites)
if os.path.exists("vcor.npy"):
    log.result("Read vcor from disk")
    with open("vcor.npy", "r") as f:
        vcor.update(np.load(f))
else:
    log.result("Antiferromagnetic initial guess of vcor")
    dmet.AFInitGuessOrbs(vcor, Lat, AForbs, PMorbs, shift = 0.3, polar = 0.3)
Mu = 0. # I don't know what is Mu

dc = dmet.FDiisContext(DiisDim)

conv = False

history = dmet.IterHistory()

solver = dmet.impurity_solver.CASSCF(ncas = 12, nelecas = 10)

for iter in range(MaxIter):

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)
    dmet.reportOccupation(Lat, rho[:,0])

    log.section("\nDMET Iteration %d\n", iter)
    log.section("\nconstructing impurity problem\n")
    log.result("Making embedding basis")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching = True)
    log.section("\nsolving impurity problem\n")
    rhoEmb, EnergyEmb, ImpHam, dmu = dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
          solver_args = {"guess": dmet.foldRho(rho, Lat, basis)})
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    rhoImp, EnergyImp, nelecImp = dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e)
    dmet.reportOccupation(Lat, rhoImp)
    
    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, vcor, np.inf, Filling, MaxIter1 = 200, MaxIter2 = 0)

    history.update(EnergyImp, err, nelecImp, np.max(abs(vcor.get() - vcor_new.get())), dc)    
    if np.max(abs(vcor.get() - vcor_new.get())) < 1e-5:
        conv = True
        break

    if not conv:
        skipDiis = (iter < DiisStart) and (la.norm(vcor_new.param - vcor.param) > 0.01)
        pvcor, _, _ = dc.Apply(vcor_new.param, vcor_new.param - vcor.param, Skip = skipDiis)
        vcor.update(pvcor)

    with open("vcor.npy", "w") as f:
        np.save(f, vcor_new.param)

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
