import libdmet.utils.logger as log
import libdmet.dmet.HubbardBCS as dmet
from libdmet.system import integral
from libdmet.routine.vcor import Vcor
import numpy as np
import numpy.linalg as la
import os

log.verbose = "DEBUG1"

U = 8
LatSize = [24, 24]
ImpSize = [2, 2]
# Pi-phase shift on x-direction
DoubleImp = [4, 2]
Filling = 0.875/2
MaxIter = 40
M = 400
DiisStart = 4
TraceStart = 4
DiisDim = 4

LatMf = dmet.SquareLattice(*(LatSize + DoubleImp))
HamMf = dmet.Ham(LatMf, U)
LatMf.setHam(HamMf)

LatImp = dmet.SquareLattice(*(LatSize + ImpSize))
HamImp = dmet.Ham(LatImp, U)
LatImp.setHam(HamImp)

Mf2Imp = [LatImp.sitedict[tuple(site)] for site in LatMf.sites]
Imp2Mf = [LatMf.sitedict[tuple(site)] for site in LatImp.sites]

vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand = 0.001)

class VcorWrapper(Vcor):
    def __init__(self, vcor):
        self.vcor = vcor
        self.grad = None
        self.param = vcor.param

    def update(self, param):
        self.vcor.update(param)
        self.param = self.vcor.param

    def islocal(self):
        return True

    def get(self, i = 0, kspace = True):
        if kspace or i == 0:
            return self.evaluate()
        else:
            return np.zeros_like(self.evalute())

    def evaluate(self):
        va, vb, vd = self.vcor.value
        nImp = va.shape[0]
        v = np.zeros((3, nImp*2, nImp*2))
        v[0, :nImp, :nImp] = va
        v[0, nImp:, nImp:] = va
        v[1, :nImp, :nImp] = vb
        v[1, nImp:, nImp:] = vb
        v[2, :nImp, :nImp] = vd
        v[2, nImp:, nImp:] = vd
        #v[0, :nImp, :nImp] = va
        #v[0, nImp:, nImp:] = vb
        #v[1, :nImp, :nImp] = vb
        #v[1, nImp:, nImp:] = va
        #v[2, :nImp, :nImp] = vd
        #v[2, nImp:, nImp:] = -vd.T
        return v

    def gradient(self):
        if self.grad is None:
            gImp = self.vcor.gradient()
            nImp = gImp.shape[-1]
            g = np.zeros((self.length(), 3, nImp*2, nImp*2))
            g[:, :, :nImp, :nImp] = gImp
            g[:, :, nImp:, nImp:] = gImp
            #g[:, 0, nImp:, nImp:] = gImp[:, 1]
            #g[:, 1, nImp:, nImp:] = gImp[:, 0]
            #g[:, 2, nImp:, nImp:] = -np.transpose(gImp[:, 2], (0, 2, 1))
            self.grad = g
        return self.grad

    def length(self):
        return self.vcor.length()
    
    def assign(self, v0):
        log.error("Not implemented")

def __embHam1e(latticeImp, basisImp, vcorImp, \
        latticeMf, basisMf, vcorMf, mu, **kwargs):
    import libdmet.routine.bcs_helper as helper
    from copy import deepcopy
    nbasis = basisImp.shape[-1]
    spin = 2
    H0 = 0.
    H1 = {
            "cd": np.empty((2, nbasis, nbasis)), 
            "cc": np.empty((1, nbasis, nbasis))
    }
    H0energy = 0.
    H1energy = {
            "cd": np.empty((2, nbasis, nbasis)), 
            "cc": np.empty((1, nbasis, nbasis))
    }

    log.debug(1, "transform Fock")
    H1["cd"], H1["cc"][0], H0 = helper.transform_trans_inv_sparse(basisImp, latticeImp, \
            latticeImp.getFock(kspace = False))
    log.debug(1, "transform Vcor")
    v = deepcopy(vcorMf.get())
    v[0] -= mu * np.eye(latticeMf.supercell.nsites)
    v[1] -= mu * np.eye(latticeMf.supercell.nsites)
    tempCD, tempCC, tempH0 = helper.transform_local(basisMf, latticeMf, v)
    H1["cd"] += tempCD
    H1["cc"][0] += tempCC
    H0 += tempH0

    if not "fitting" in kwargs or not kwargs["fitting"]:
        tempCD, tempCC, tempH0 = helper.transform_imp(basisImp, latticeImp, vcorImp.get())
        H1["cd"] -= tempCD
        H1["cc"][0] -= tempCC
        H0 -= tempH0


    if latticeImp.getImpJK() is not None:
        log.debug(1, "transform impurity JK")
        tempCD, tempCC, tempH0 = helper.transform_imp(basisImp, latticeImp, latticeImp.getImpJK())
        H1["cd"] -= tempCD
        H1["cc"][0] -= tempCC
        H0 -= tempH0

    log.debug(1, "transform native H1")
    H1energy["cd"], H1energy["cc"][0], H0energy = helper.transform_imp_env(basisImp, \
            latticeImp, latticeImp.getH1(kspace = False))
    return (H1, H0), (H1energy, H0energy)

fvcor = "vcor_ps.npy"
if os.path.exists(fvcor):
    log.result("Read vcor from disk")
    vcor.update(np.load(fvcor))

vcorMf = VcorWrapper(vcor)

Mu = U * Filling
dc = dmet.FDiisContext(DiisDim) # I don't know yet whether diis needs to be changed

conv = False

history = dmet.IterHistory()

solver = dmet.impurity_solver.StackBlock(nproc = 1, nthread = 2, nnode = 1, \
        bcs = True, reorder = True, tol = 1e-7, maxM = M)

log.section("\nfitting chemical potential\n")
_, Mu = dmet.HartreeFockBogoliubov(LatMf, vcorMf, Filling, Mu)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section ("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcorMf.get())
    log.result("Mu (guess) = %20.12f", Mu)
    GRho, Mu = dmet.HartreeFockBogoliubov(LatMf, vcorMf, None, Mu)
    log.section("\nconstructing impurity problem\n")
    log.result("Making embedding basis")
    basis = dmet.bcs.embBasis(LatMf, GRho, local = True, sites = range(np.product(ImpSize)))
    log.result("Rotate bath orbitals to match alpha and beta basis")
    nbasis = basis.shape[-1]
    basis[:, :, :, nbasis/2:] = dmet.basisMatching(basis[:, :, :, nbasis/2:])
    basisV = basis[:, :, :LatMf.supercell.nsites].reshape(2, \
            LatMf.nsites, nbasis)[:, Imp2Mf].reshape(2, \
            LatImp.ncells, LatImp.supercell.nsites, nbasis)
    basisU = basis[:, :, LatMf.supercell.nsites:].reshape(2, \
            LatMf.nsites, nbasis)[:, Imp2Mf].reshape(2, \
            LatImp.ncells, LatImp.supercell.nsites, nbasis)
    basisImp = np.zeros((2, LatImp.ncells, LatImp.supercell.nsites*2, nbasis))
    basisImp[:, :, :LatImp.supercell.nsites] = basisV
    basisImp[:, :, LatImp.supercell.nsites:] = basisU
    log.result("Constructing impurity Hamiltonian")
    log.info("One-body part")
    (Int1e, H0_from1e), (Int1e_energy, H0_energy_from1e) = \
            __embHam1e(LatImp, basisImp, vcor, LatMf, basis, vcorMf, Mu)
    log.info("Two-body part")
    Int2e, Int1e_from2e, H0_from2e = \
            dmet.bcs.__embHam2e(LatImp, basisImp, vcor, True)
    H0 = H0_from1e + H0_from2e
    H0_energy = H0_energy_from1e + H0_from2e
    if Int1e_from2e is not None:
        Int1e["cd"] += Int1e_from2e["cd"]
        Int1e["cc"] += Int1e_from2e["cc"]
        Int1e_energy["cd"] += Int1e_from2e["cd"]
        Int1e_energy["cc"] += Int1e_from2e["cc"]
    ImpHam, H_energy = integral.Integral(nbasis, False, True, H0, Int1e, Int2e), \
            (Int1e_energy, H0_energy)
    #np.save("basis.ps.npy", basisImp)
    #integral.dumpFCIDUMP("FCIDUMP.ps", ImpHam)

    log.section("\nsolving impurity problem\n")
    GRhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(LatImp, Filling, ImpHam, basisImp, solver, step = 0.3, delta = 0.2)
    Mu += dmu
    vcor = dmet.addDiag(vcor, dmu)
    vcorMf.update(vcor.param)

    GRhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(GRhoEmb, EnergyEmb, LatImp, basisImp, ImpHam, H_energy, dmu)

    log.section("\nfitting correlation potential\n")
    vcorMf_new, err = dmet.FitVcor(GRhoEmb, LatMf, basis, vcorMf, Mu, \
            MaxIter1 = 400, MaxIter2 = 0)
    
    vcor_new = vcorMf_new.vcor
    if iter >= TraceStart:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        ddiagV = np.average(np.diagonal(\
                (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        vcor_new = dmet.addDiag(vcor_new, -ddiagV)

    vcorMf_new = VcorWrapper(vcor_new)
    log.section("\nfitting chemical potential\n")
    GRho, Mu_new = dmet.HartreeFockBogoliubov(LatMf, vcorMf_new, Filling, Mu)
    log.result("dMu = %20.12f", Mu_new - Mu)

    history.update(EnergyImp, err, nelecImp, \
            np.max(abs(vcor.get() - vcor_new.get())), dc)

    if np.max(abs(vcor.get() - vcor_new.get())) < 1e-5:
        conv = True
        break

    skipDiis = (iter < DiisStart) and (la.norm(vcor_new.param - vcor.param) > 0.01)
    pvcor, dpvcor, _ = dc.Apply( \
            np.hstack((vcor_new.param, Mu_new)), \
            np.hstack((vcor_new.param - vcor.param, Mu_new - Mu)), \
            Skip = skipDiis)
    vcorMf_new.update(pvcor[:-1])
    Mu = pvcor[-1]

    vcor = vcor_new
    vcorMf = vcorMf_new
    np.save(fvcor, vcor.param)

solver.cleanup()

if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")
