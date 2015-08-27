from tempfile import mkdtemp
import itertools as it
import os
import subprocess as sub
import numpy as np
import numpy.linalg as la
from libdmet.solver import block, scf
from libdmet.system import integral
import libdmet.utils.logger as log
from libdmet.utils.misc import mdot
from libdmet.routine.localizer import Localizer
from libdmet.utils.munkres import Munkres, make_cost_matrix
from libdmet.routine.bcs_helper import extractRdm
from libdmet.integral.integral_emb_casci import transform

def cas_from_1pdm(rho, ncas, nelecas, nelec):
    assert(nelecas <= nelec)
    natocc, natorb = la.eigh(rho)
    log.debug(1, "Natural orbital occupations:\n%s", natocc)
    norbs = natocc.shape[0]
    
    ncore = nelec - nelecas
    nvirt = norbs - ncore - ncas
    log.info("ncore = %d nvirt = %d ncas = %d", ncore, nvirt, ncas)
    log.info("core orbital occupation cut-off = %20.12f", \
            natocc[-ncore] if ncore > 0 else 1)
    log.info("virt orbital occupation cut-off = %20.12f", \
            natocc[nvirt-1] if nvirt > 0 else 0)
    
    if ncore == 0:
        casocc = natocc[nvirt:]
    else:
        casocc = natocc[nvirt:-ncore]
    _nvirt = np.sum(casocc < 0.3)
    _ncore = np.sum(casocc > 0.7)
    _npart = np.sum((casocc >= 0.3) * (casocc <= 0.7))
    log.info("In CAS:\n"
            "Occupied (n>0.7): %d\n""Virtual  (n<0.3): %d\n"
            "Partial Occupied: %d\n", _ncore, _nvirt, _npart)
    
    core = natorb[:, norbs-ncore:]
    cas = natorb[:, nvirt:norbs-ncore:-1]
    virt = natorb[:, :nvirt]
    return core, cas, virt, (_ncore, _npart, _nvirt)

def cas_from_energy(mo, mo_energy, ncas, nelecas, nelec):
    assert(nelecas <= nelec)
    log.debug(1, "Orbital energies:\n%s", mo_energy)
    norbs = mo_energy.shape[0]
    
    ncore = nelec - nelecas
    nvirt = norbs - ncore - ncas
    log.info("ncore = %d nvirt = %d ncas = %d", ncore, nvirt, ncas)
    log.info("core orbital energy cut-off = %20.12f", \
            mo_energy[ncore-1] if ncore > 0 else float("Inf"))
    log.info("virt orbital eneryg cut-off = %20.12f", \
            mo_energy[-nvirt] if nvirt > 0 else -float("Inf"))
    
    if nvirt == 0:
        casenergy = mo_energy[ncore:]
    else:
        casenergy = mo_energy[ncore:-nvirt]
    mu = 0.5 * (casenergy[nelecas-1] + casenergy[nelecas])
    log.debug(0, "HF gap = %20.12f", casenergy[nelecas] - casenergy[nelecas-1])
    _nvirt = np.sum(casenergy > mu+1e-4)
    _ncore = np.sum(casenergy < mu-1e-4)
    _npart = np.sum((casenergy >= mu-1e-4) * (casenergy <= mu+1e-4))
    log.info("In CAS:\n"
            "Occupied (e<mu): %d\n""Virtual  (e>mu): %d\n"
            "Partial Occupied: %d\n", _ncore, _nvirt, _npart)
    
    core = mo[:, :ncore]
    cas = mo[:, ncore:norbs-nvirt]
    virt = mo[:, norbs-nvirt:]
    return core, cas, virt, (_ncore, _npart, _nvirt)

def get_orbs(casci, Ham, guess, nelec):
    spin = Ham.H1["cd"].shape[0]

    casci.scfsolver.set_system(nelec, 0, False, spin == 1)
    casci.scfsolver.set_integral(Ham)
    
    E_HF, rhoHF = casci.scfsolver.HF(tol = 1e-5, MaxIter = 30, InitGuess = guess)

    if casci.MP2natorb:
        E_MP2, rhoMP2 = casci.scfsolver.MP2()
        log.result("MP2 energy = %20.12f", E_HF + E_MP2)

    if casci.spinAverage:
        if casci.MP2natorb:
            rho0 = rhoMP2
        else:
            rho0 = rhoHF
        core, cas, virt, casinfo = cas_from_1pdm(0.5*(rho0[0]+rho0[1]), \
                casci.ncas, casci.nelecas/2, nelec/2)
    else:
        core = [None, None]
        cas = [None, None]
        virt = [None, None]
        casinfo = [None, None]
        if casci.MP2natorb:
            for s in range(spin):
                log.info("Spin %d", s)
                core[s], cas[s], virt[s], casinfo[s] = cas_from_1pdm(rhoMP2[s], \
                        casci.ncas, casci.nelecas/2, nelec/2)
        else:
            # use hartree-fock orbitals, we need orbital energy or order in this case
            mo = casci.scfsolver.get_mo()
            mo_energy = casci.scfsolver.get_mo_energy()
            for s in range(spin):
                log.info("Spin %d", s)
                core[s], cas[s], virt[s], casinfo[s] = cas_from_energy(mo[s], \
                        mo_energy[s], casci.ncas, casci.nelecas/2, nelec/2)
        core = np.asarray(core)
        cas = np.asarray(cas)
        virt = np.asarray(virt)
    return core, cas, virt, casinfo

def buildCASHamiltonian(Ham, core, cas):
    spin = Ham.H1["cd"].shape[0]
    if len(core.shape) == 2:
        core = np.asarray([core, core])
        cas = np.asarray([cas, cas])

    coreRdm = np.asarray([np.dot(core[0], core[0].T), np.dot(core[1], core[1].T)])
    # zero-energy
    H0 = Ham.H0
    # core-core one-body
    H0 += np.sum(coreRdm[0] * Ham.H1["cd"][0] + coreRdm[1] * Ham.H1["cd"][1])
    # core-fock
    vj00 = np.tensordot(coreRdm[0], Ham.H2["ccdd"][0], ((0,1), (0,1)))
    vj11 = np.tensordot(coreRdm[1], Ham.H2["ccdd"][1], ((0,1), (0,1)))
    vj10 = np.tensordot(coreRdm[0], Ham.H2["ccdd"][2], ((0,1), (0,1)))
    vj01 = np.tensordot(coreRdm[1], Ham.H2["ccdd"][2], ((1,0), (3,2)))
    vk00 = np.tensordot(coreRdm[0], Ham.H2["ccdd"][0], ((0,1), (0,3)))
    vk11 = np.tensordot(coreRdm[1], Ham.H2["ccdd"][1], ((0,1), (0,3)))
    v = np.asarray([vj00 + vj01 - vk00, vj11 + vj10 - vk11])
    # core-core two-body
    H0 += 0.5 * np.sum(coreRdm[0]*v[0] + coreRdm[1]*v[1])
    H1 = {
        "cd": np.asarray([
            mdot(cas[0].T, Ham.H1["cd"][0]+v[0], cas[0]),
            mdot(cas[1].T, Ham.H1["cd"][1]+v[1], cas[1])])
    }
    H2 = {
        "ccdd": scf.incore_transform(Ham.H2["ccdd"], \
            (cas, cas, cas, cas))
    }
    return integral.Integral(cas.shape[2], False, False, H0, H1, H2)

def split_localize(orbs, info, Ham, basis = None):
    spin = Ham.H1["cd"].shape[0]
    norbs = Ham.H1["cd"].shape[1]
    localorbs = np.zeros_like(orbs) # with respect to original embedding basis
    rotmat = np.zeros_like(Ham.H1["cd"]) # with respect to active orbitals
    for s in range(spin):
        occ, part, virt = info[s]
        if occ > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    :occ, :occ, :occ, :occ])
            log.info("Localization: Spin %d, occupied", s)
            localizer.optimize()
            occ_coefs = localizer.coefs.T
            localorbs[s, :, :occ] = np.dot(orbs[s,:,:occ], occ_coefs)
            rotmat[s, :occ, :occ] = occ_coefs
        if virt > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    -virt:, -virt:, -virt:, -virt:])
            log.info("Localization: Spin %d, virtual", s)
            localizer.optimize()
            virt_coefs = localizer.coefs.T
            localorbs[s, :, -virt:] = np.dot(orbs[s,:,-virt:], virt_coefs)
            rotmat[s, -virt:, -virt:] = virt_coefs
        if part > 0:
            localizer = Localizer(Han.H2["ccdd"][s, occ:norbs-virt, \
                occ:norbs-virt, occ:norbs-virt, occ:nobrs-virt])
            log.info("Localization: Spin %d, partially occupied:", s)
            localizer.optimize()
            part_coefs = localizer.ceofs.T
            localorbs[s, :, occ:norbs-virt] = \
                    np.dot(orbs[s,:,occ:norbs-virt], part_coefs)
            rotmat[s, occ:norbs-virt, occ:norbs-virt] = part_coefs

    if basis is not None:
        # match alpha, beta basis
        localbasis = np.asarray([
                np.tensordot(basis[0], localorbs[0], (2, 0)),
                np.tensordot(basis[1], localorbs[1], (2, 0))
        ])
        ovlp = np.tensordot(np.abs(localbasis[0]), np.abs(localbasis[1]), ((0,1), (0,1)))
        ovlp_sq = ovlp ** 2
        cost_matrix = make_cost_matrix(ovlp_sq, lambda cost: 1. - cost)
        m = Munkres()
        indexes = m.compute(cost_matrix)
        indexes = sorted(indexes, key = lambda idx: idx[0])
        vals = map(lambda idx: ovlp_sq[idx], indexes)
        log.debug(1, "Orbital pairs and their overlap:")
        for i in range(norbs):
            log.debug(1, "(%2d, %2d) -> %12.6f", indexes[i][0], indexes[i][1], vals[i])
        log.info("Match localized orbitals: max %5.2f min %5.2f ave %5.2f", \
                np.max(vals), np.min(vals), np.average(vals))

        # update localorbs and rotmat
        orderb = map(lambda idx: idx[1], indexes)
        localorbs[1] = localorbs[1][:,orderb]
        rotmat[1] = rotmat[1][:,orderb]

        localbasis[1] = localbasis[1][:,:,orderb]
        # make spin up and down basis have the same sign, i.e.
        # inner product larger than 1
        for i in range(norbs):
            if np.sum(localbasis[0,:,:,i] * localbasis[1,:,:,i]) < 0:
                localorbs[1,:,i] *= -1.
                rotmat[1,:,i] *= -1.

    H1 = {
        "cd":np.asarray([
                mdot(rotmat[0].T, Ham.H1["cd"][0], rotmat[0]),
                mdot(rotmat[1].T, Ham.H1["cd"][1], rotmat[1])
    ])}
    H2 = {
        "ccdd": scf.incore_transform(Ham.H2["ccdd"], \
                (rotmat, rotmat, rotmat, rotmat))
    }
    HamLocal = integral.Integral(norbs, False, False, Ham.H0, H1, H2)
    return HamLocal, localorbs, rotmat

def gaopt(Ham, tmp = "/tmp"):
    norbs = Ham.norb
    # build K matrix
    K = np.empty((norbs, norbs))
    Int2e = Ham.H2["ccdd"]
    for i, j in it.product(range(norbs), repeat = 2):
        K[i,j] = 0.5*abs(Int2e[0,i,j,i,j]) + 0.5*abs(Int2e[1,i,j,i,j]) + abs(Int2e[2,i,j,i,j])
        K[i,j] += 1e-7 * (abs(Ham.H1["cd"][0,i,j])+abs(Ham.H1["cd"][1,i,j]))

    # write K matrix
    wd = mkdtemp(prefix = "GAOpt", dir = tmp)
    log.debug(0, "gaopt temporary file: %s", wd)
    with open(os.path.join(wd, "Kmat"), "w") as f:
        f.write("%d\n" % norbs)
        for i in range(norbs):
            for j in range(norbs):
                f.write(" %24.16f" % K[i,j])
            f.write("\n")

    # write configure file
    with open(os.path.join(wd, "ga.conf"), "w") as f:
        f.write("maxcomm 32\n")
        f.write("maxgen 20000\n")
        f.write("maxcell %d\n" % (2*norbs))
        f.write("cloning 0.90\n")
        f.write("mutation 0.10\n")
        f.write("elite 1\n")
        f.write("scale 1.0\n")
        f.write("method gauss\n")

    executable = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), \
            "../block/genetic/gaopt"))
    log.debug(0, "gaopt executable: %s", executable)

    log.debug(0, "call gaopt")
    with open(os.path.join(wd, "output"), "w") as f:
        if block.Block.env_slurm:
            sub.check_call(" ".join(["srun", \
                    executable, "-s", "-config", os.path.join(wd, "ga.conf"), \
                    "-integral", os.path.join(wd, "Kmat")]), stdout = f, shell = True)
        else:
            sub.check_call(["mpirun", "-np", "4", \
                    executable, "-s", "-config", os.path.join(wd, "ga.conf"), \
                    "-integral", os.path.join(wd, "Kmat")], stdout = f)

    with open(os.path.join(wd, "output"), "r") as f:
        result = f.readlines()[-1]
        log.debug(1, "gaopt result: %s", result)
        reorder = map(lambda i: int(i)-1, result.split(','))

    sub.check_call(["rm", "-rf", wd])

    return reorder
    
def momopt(old_basis, new_basis):
    # use Hungarian algorithm to match the basis
    ovlp = 0.5 * np.tensordot(np.abs(old_basis), np.abs(new_basis), ((0,1,2), (0,1,2)))
    ovlp_sq = ovlp ** 2
    cost_matrix = make_cost_matrix(ovlp_sq, lambda cost: 1. - cost)

    m = Munkres()
    indexes = m.compute(cost_matrix)
    indexes = sorted(indexes, key = lambda idx: idx[0])
    vals = map(lambda idx: ovlp_sq[idx], indexes)
    log.info("MOM reorder quality: max %5.2f min %5.2f ave %5.2f", \
            np.max(vals), np.min(vals), np.average(vals))

    reorder = [idx[1] for idx in indexes]
    return reorder, np.average(vals)

def reorder(order, Ham, orbs, rot = None):
    # order 4 1 3 2 means 4 to 1, 1 to 2, 3 to 3, 2 to 4
    # reorder in place
    orbs = orbs[:, :, order]
    Ham.H1["cd"] = Ham.H1["cd"][:, order, :]
    Ham.H1["cd"] = Ham.H1["cd"][:, :, order]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, order, :, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, order, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, order, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, :, order]
    if rot is not None:
        rot = rot[:, :, order]
        return Ham, orbs, rot
    else:
        return Ham, orbs

class DmrgCI(object):
    def __init__(self, ncas, nelecas, MP2natorb = False, spinAverage = False, \
            splitloc = True, cisolver = None, mom_reorder = True, tmpDir = "/tmp"):
        log.eassert(ncas * 2 >= nelecas, \
                "CAS size not compatible with number of electrons")
        self.ncas = ncas
        self.nelecas = nelecas # alpha and beta
        self.MP2natorb = MP2natorb
        self.spinAverage = spinAverage
        self.splitloc = splitloc
        log.eassert(cisolver is not None, "No default ci solver is available" \
                " with CASCI, you have to use Block")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()

        # reorder scheme for restart block calculations
        if mom_reorder:
            if block.Block.reorder:
                log.warning("Using maximal overlap method (MOM) to reorder localized "\
                        "orbitals, turning off Block reorder option")
                block.Block.reoder = False

        self.mom_reorder = mom_reorder
        self.localized_cas = None
        self.tmpDir = tmpDir

    def run(self, Ham, ci_args = {}, guess = None, nelec = None, basis = None, similar = False): 
        # ci_args is a list or dict for ci solver, or None
        spin = Ham.H1["cd"].shape[0]
        log.eassert(spin == 2, \
                "spin-restricted CASCI solver is not implemented")
        if nelec is None:
            nelec = Ham.norb

        core, cas, virt, casinfo = get_orbs(self, Ham, guess, nelec)
        coreRho = np.asarray([np.dot(core[0], core[0].T), \
                np.dot(core[1], core[1].T)])
        
        casHam = buildCASHamiltonian(Ham, core, cas)

        if self.splitloc:
            casHam, cas, _ = \
                    split_localize(cas, casinfo, casHam, basis = basis)
        
        if self.mom_reorder:
            log.eassert(basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            if self.localized_cas is None:
                order = gaopt(casHam, tmp = self.tmpDir)
            else:
                # define cas_basis
                cas_basis = np.asarray([
                    np.tensordot(basis[0], cas[0], (2,0)),
                    np.tensordot(basis[1], cas[1], (2,0))
                ])
                # cas_basis and self.localized_cas are both in
                # atomic representation now
                order, q = momopt(self.localized_cas, cas_basis)
                # if the quality of mom is too bad, we reorder the orbitals
                # using genetic algorithm
                # FIXME seems larger q is a better choice
                if q < 0.7:
                    order = gaopt(casHam, tmp = self.tmpDir)

            log.info("Orbital order: %s", order)
            # reorder casHam and cas
            casHam, cas = reorder(order, casHam, cas)
            # store cas in atomic basis
            self.localized_cas = np.asarray([
                np.tensordot(basis[0], cas[0], (2,0)),
                np.tensordot(basis[1], cas[1], (2,0))
            ])

        casRho, E = self.cisolver.run(casHam, nelec = self.nelecas, **ci_args)
        
        rho = np.asarray([mdot(cas[0], casRho[0], cas[0].T), \
                mdot(cas[1], casRho[1], cas[1].T)]) + coreRho

        return rho, E

    def cleanup(self):
        self.cisolver.cleanup()

def get_qps(casci, Ham, guess):
    # get quasiparticles by solving scf
    casci.scfsolver.set_system(None, 0, True, False)
    casci.scfsolver.set_integral(Ham)

    E_HFB, GRho_HFB = casci.scfsolver.HFB(Mu = 0, tol = 1e-5, \
            MaxIter = 30, InitGuess = guess)

    mo = casci.scfsolver.get_mo()
    mo_energy = casci.scfsolver.get_mo_energy()
    norb = mo_energy.size / 2
    ncas = casci.ncas
    core = mo[:, :norb - ncas]
    virt = mo[:, norb + ncas:]

    cas0 = mo[:, norb - ncas: norb + ncas]
    cas_energy = mo_energy[norb - ncas: norb + ncas]
    cas = [{"o": [], "v": [], "p": []}, {"o": [], "v": [], "p": []}]
    for i in range(ncas*2):
        v, u = la.norm(cas0[:norb, i]), la.norm(cas0[norb:, i])
        if v > u:
            if cas_energy[i] < -1e-4:
                cas[0]['o'].append(cas0[:, i])
            elif cas_energy[i] > 1e-4:
                cas[0]['v'].append(cas0[:, i])
            else:
                cas[0]['p'].append(cas0[:, i])
        else:
            core = np.hstack((core, cas0[:, i:i+1]))
            if cas_energy[i] < -1e-4:
                cas[1]['v'].append(np.concatenate(\
                        (cas0[norb:, i], cas0[:norb, i]), 0))
            elif cas_energy[i] > 1e-4:
                cas[1]['o'].append(np.concatenate(\
                        (cas0[norb:, i], cas0[:norb, i]), 0))
            else:
                cas[1]['p'].append(np.concatenate(\
                        (cas0[norb:, i], cas0[:norb, i]), 0))

    casinfo = map(lambda i: (len(cas[i]['o']), len(cas[i]['p']), len(cas[i]['v'])), \
            range(2))
    cas = map(lambda i: np.asarray(cas[i]['o'] + cas[i]['p'] + cas[i]['v']).T, \
            range(2))
    log.eassert(cas[0].shape[1] == cas[1].shape[1], \
            "Number of alpha and beta quasiparticles are not equal")
    return core, np.asarray(cas), virt, casinfo

def buildBCSCasHamiltonian(Ham, core, cas):
    norb = core.shape[0] / 2
    coreGRdm = np.dot(core, core.T)
    cRhoA, cRhoB, cKappaBA = extractRdm(coreGRdm)
    # zero-energy
    _H0 = Ham.H0
    # core-core one-body
    _H0 += np.sum(cRhoA * Ham.H1["cd"][0] + cRhoB * Ham.H1["cd"][1] + \
            2 * cKappaBA.T * Ham.H1["cc"][0])
    # core-fock
    assert(Ham.H2["cccd"] is None or la.norm(Ham.H2["cccd"]) == 0)
    assert(Ham.H2["cccc"] is None or la.norm(Ham.H2["cccc"]) == 0)
    _eriA, _eriB, _eriAB = Ham.H2["ccdd"]

    vj00 = np.tensordot(cRhoA, _eriA, ((0,1), (0,1)))
    vj11 = np.tensordot(cRhoB, _eriB, ((0,1), (0,1)))
    vj10 = np.tensordot(cRhoA, _eriAB, ((0,1), (0,1)))
    vj01 = np.tensordot(_eriAB, cRhoB, ((2,3), (0,1)))
    vk00 = np.tensordot(cRhoA, _eriA, ((0,1), (0,3)))
    vk11 = np.tensordot(cRhoB, _eriB, ((0,1), (0,3)))
    vl10 = np.tensordot(cKappaBA, _eriAB, ((0,1), (0,2)))# wrt kappa_ba
    v = np.asarray([vj00+vj01-vk00, vj11+vj10-vk11, vl10.T])
    # core-core two-body
    _H0 += 0.5 * np.sum(cRhoA * v[0] + cRhoB * v[1] + 2 * cKappaBA.T * v[2])

    VA, VB, UA, UB = cas[0,:norb], cas[1,:norb], cas[1,norb:], cas[0,norb:]
    H0, CD, CC, CCDD, CCCD, CCCC = transform(VA, VB, UA, UB, _H0, \
            Ham.H1["cd"][0] + v[0], Ham.H1["cd"][1] + v[1], Ham.H1["cc"][0] + \
            v[2][0], Ham.H2["ccdd"][0], Ham.H2["ccdd"][1], Ham.H2["ccdd"][2])
    return integral.Integral(cas.shape[2], False, True, H0, {"cd": CD, "cc": CC}, \
            {"ccdd": CCDD, "cccd": CCCD, "cccc": CCCC})

class BCSDmrgCI(object):
    def __init__(self, ncas, splitloc = False, cisolver = None, \
            mom_reorder = False, tmpDir = "/tmp"):
        self.ncas = ncas
        self.splitloc = splitloc
        log.eassert(cisolver is not None, "No default ci solver is available" \
                " with CASCI, you have to use Block")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()

        # reorder scheme for restart block calculations
        if mom_reorder:
            if block.Block.reorder:
                log.warning("Using maximal overlap method (MOM) to reorder localized "\
                        "orbitals, turning off Block reorder option")
                block.Block.reoder = False

        self.mom_reorder = mom_reorder
        self.localized_cas = None
        self.tmpDir = tmpDir

    def run(self, Ham, ci_args = {}, guess = None, basis = None, similar = False):
        # ci_args is a list or dict for ci solver, or None

        # FIXME think about choosing number of electron/hole freely
        core, cas, virt, casinfo = get_qps(self, Ham, guess)
        coreGRho = np.dot(core, core.T)

        BCSCasHam = buildBCSCasHamiltonian(Ham, core, cas)
        #if self.splitloc:
        #    casHam, cas, _ = \
        #            split_localize(cas, casinfo, casHam, basis = basis)
        #
        #if self.mom_reorder:
        #    log.eassert(basis is not None, \
        #            "maximum overlap method (MOM) requires embedding basis")
        #    if self.localized_cas is None:
        #        order = gaopt(casHam, tmp = self.tmpDir)
        #    else:
        #        # define cas_basis
        #        cas_basis = np.asarray([
        #            np.tensordot(basis[0], cas[0], (2,0)),
        #            np.tensordot(basis[1], cas[1], (2,0))
        #        ])
        #        # cas_basis and self.localized_cas are both in
        #        # atomic representation now
        #        order, q = momopt(self.localized_cas, cas_basis)
        #        # if the quality of mom is too bad, we reorder the orbitals
        #        # using genetic algorithm
        #        # FIXME seems larger q is a better choice
        #        if q < 0.7:
        #            order = gaopt(casHam, tmp = self.tmpDir)

        #    log.info("Orbital order: %s", order)
        #    # reorder casHam and cas
        #    casHam, cas = reorder(order, casHam, cas)
        #    # store cas in atomic basis
        #    self.localized_cas = np.asarray([
        #        np.tensordot(basis[0], cas[0], (2,0)),
        #        np.tensordot(basis[1], cas[1], (2,0))
        #    ])

        casGRho, E = self.cisolver.run(BCSCasHam, **ci_args)
        norb = core.shape[0] / 2

        cas1 = np.empty((norb*2, self.ncas * 2))
        cas1[:, :self.ncas] = cas[0]
        cas1[:norb, self.ncas:] = cas[1, norb:]
        cas1[norb:, self.ncas:] = cas[1, :norb]
        GRho = mdot(cas1, casGRho, cas1.T) + coreGRho

        return GRho, E

    def cleanup(self):
        self.cisolver.cleanup()

