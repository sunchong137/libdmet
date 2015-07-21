from libdmet.solver import block, scf, casscf
from libdmet.system import integral
import libdmet.utils.logger as log
import numpy as np
import numpy.linalg as la
from libdmet.utils.misc import mdot
from libdmet.routine.localizer import Localizer
from libdmet.utils.munkres import Munkres, make_cost_matrix
from copy import deepcopy

class AFQMC(object):
    pass

class Block(object):
    def __init__(self, nproc, nnode = 1, TmpDir = "/tmp", SharedDir = None, \
            reorder = False, minM = 100, tol = 1e-7, spinAdapted = False):
        log.eassert(nnode == 1 or SharedDir is not None, \
                "Running on multiple nodes (nnod = %d), must specify shared directory", \
                nnode)
        self.cisolver = block.Block()
        block.Block.set_nproc(nproc, nnode)
        self.cisolver.createTmp(tmp = TmpDir, shared = SharedDir)
        block.Block.reorder = reorder
        self.schedule = block.Schedule(sweeptol = tol)
        self.minM = minM
        self.spinAdapted = spinAdapted

    def run(self, Ham, M, nelec = None):
        if nelec is None:
            nelec = Ham.norb
        if not self.cisolver.sys_initialized:
            if self.spinAdapted:
                self.cisolver.set_system(nelec, 0, True, False, True)
            else:
                self.cisolver.set_system(nelec, 0, False, False, False)
        if not self.cisolver.optimized:
            self.schedule.gen_initial(minM = self.minM, maxM = M)
        else:
            self.schedule.max_iter = 16
            self.schedule.gen_restart(M)
        self.cisolver.set_schedule(self.schedule)
        self.cisolver.set_integral(Ham)

        truncation, energy, onepdm = self.cisolver.optimize()
        return onepdm, energy

    def cleanup(self):
        # FIXME first copy and save restart files
        self.cisolver.cleanup()

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
            localizer.optimize()
            occ_coefs = localizer.coefs.T
            localorbs[s, :, :occ] = np.dot(orbs[s,:,:occ], occ_coefs)
            rotmat[s, :occ, :occ] = occ_coefs
        if virt > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    -virt:, -virt:, -virt:, -virt:])
            localizer.optimize()
            virt_coefs = localizer.coefs.T
            localorbs[s, :, -virt:] = np.dot(orbs[s,:,-virt:], virt_coefs)
            rotmat[s, -virt:, -virt:] = virt_coefs
        if part > 0:
            localizer = Localizer(Han.H2["ccdd"][s, occ:norbs-virt, \
                occ:norbs-virt, occ:norbs-virt, occ:nobrs-virt])
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

class CASCI(object):
    def __init__(self, ncas, nelecas, MP2natorb = False, spinAverage = False, \
            splitloc = False, cisolver = None):
        log.eassert(ncas * 2 >= nelecas, \
                "CAS size not compatible with number of electrons")
        self.ncas = ncas
        self.nelecas = nelecas # alpha and beta
        self.MP2natorb = MP2natorb
        self.spinAverage = spinAverage
        self.splitloc = splitloc
        log.eassert(cisolver is not None, "No default ci solver is available" \
                " with CASCI")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()
    
    def run(self, Ham, ci_args = {}, guess = None, nelec = None, basis = None): 
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
            casHam, cas, rotmat = \
                    split_localize(cas, casinfo, casHam, basis = basis)
        
        casRho, E = self.cisolver.run(casHam, nelec = self.nelecas, **ci_args)
        
        rho = np.asarray([mdot(cas[0], casRho[0], cas[0].T), \
                mdot(cas[1], casRho[1], cas[1].T)]) + coreRho

        return rho, E

class CASSCF(object):
    # CASSCF with FCI solver only, not DMRG-SCF

    options = {
        "max_orb_stepsize": 0.03,
        "max_cycle_macro": 50,
        "max_cycle_micro": 2, # micro_cycle
        "max_cycle_micro_inner": 8,
        "conv_tol": 1e-5, # energy convergence
        "conv_tol_grad": 5e-3, # orb grad convergence
        # for augmented hessian
        "ah_level_shift": 1e-4,
        "ah_conv_tol": 1e-12, # augmented hessian accuracy
        "ah_max_cycle": 30,
        "ah_lindep": 1e-14,
        "ah_start_tol": 1e-5, # augmented hessian accuracy
        "ah_start_cycle": 2,
        "ah_grad_trust_region": 1.5,
        "ah_guess_space": 0,
        "ah_decay_rate": 0.5, # augmented hessian decay
        "dynamic_micro_step": True,
    }

    def __init__(self, ncas, nelecas, MP2natorb = False, spinAverage = False, \
            cisolver = None):
        log.eassert(ncas * 2 >= nelecas, \
                "CAS size not compatible with number of electrons")
        self.ncas = ncas
        self.nelecas = nelecas # alpha and beta
        self.MP2natorb = MP2natorb
        self.spinAverage = spinAverage
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()
        self.mo_coef = None

    def apply_options(self, mc):
        for key, val in CASSCF.options.items():
            setattr(mc, key, val)

    def run(self, Ham, mcscf_args = {}, guess = None, nelec = None):
        spin = Ham.H1["cd"].shape[0]
        norbs = Ham.H1["cd"].shape[1]
        if nelec is None:
            nelec = Ham.norb
        log.eassert(spin == 2, \
                "spin-restricted CASSCF solver is not implemented")

        if self.mo_coef is None: # restart from previous orbitals
            core, cas, virt, casinfo = get_orbs(self, Ham, guess, nelec)
            self.mo_coef = np.empty((2, norbs, norbs))
            self.mo_coef[:, :, :core.shape[2]] = core
            self.mo_coef[:, :, core.shape[2]:core.shape[2]+cas.shape[2]] = cas
            self.mo_coef[:, :, core.shape[2]+cas.shape[2]:] = virt

        nelecasAB = (self.nelecas/2, self.nelecas/2)
        mc = casscf.CASSCF(self.scfsolver.mf, self.ncas, nelecasAB)
        # apply options
        self.apply_options(mc)
        # run
        E, _, _, self.mo_coef = mc.mc1step(mo_coeff = self.mo_coef)
        rho = np.asarray(mc.make_rdm1s())

        return rho, E
