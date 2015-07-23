from libdmet.solver import block, scf, casscf
from libdmet.solver.dmrgci import DmrgCI, get_orbs
import libdmet.utils.logger as log
import numpy as np
import numpy.linalg as la

__all__ = ["AFQMC", "Block", "DmrgCI", "CASSCF"]

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

    def twopdm(self):
        log.info("Compute 2pdm")
        return self.cisolver.twopdm()

    def cleanup(self):
        # FIXME first copy and save restart files
        self.cisolver.cleanup()

class CASSCF(object):
    # CASSCF with FCI solver only, not DMRG-SCF

    options = {
        "max_orb_stepsize": 0.04,
        "max_cycle_macro": 50,
        "max_cycle_micro": 2, # micro_cycle
        "max_cycle_micro_inner": 8,
        "conv_tol": 1e-5, # energy convergence
        "conv_tol_grad": 1e-3, # orb grad convergence
        # for augmented hessian
        "ah_level_shift": 1e-4,
        "ah_conv_tol": 1e-12, # augmented hessian accuracy
        "ah_max_cycle": 30,
        "ah_lindep": 1e-14,
        "ah_start_tol": 0.1,
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
        
        # FIXME restart from last CASSCF calculation seems not working
        if self.mo_coef is None or 1: # restart from previous orbitals
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

    def cleanup(self):
        pass
