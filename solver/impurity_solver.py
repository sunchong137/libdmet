from libdmet.solver import block, scf, casscf
from libdmet.solver.afqmc import AFQMC
from libdmet.solver.dmrgci import DmrgCI, BCSDmrgCI, get_orbs
from libdmet.system import integral
import libdmet.utils.logger as log
import numpy as np
import numpy.linalg as la

__all__ = ["AFQMC", "Block", "DmrgCI", "CASSCF", "BCSDmrgCI"]

class Block(object):
    def __init__(self, nproc, nnode = 1, TmpDir = "/tmp", SharedDir = None, \
            reorder = False, minM = 100, maxM = None, tol = 1e-6, spinAdapted = False, \
            bcs = False):
        log.eassert(nnode == 1 or SharedDir is not None, \
                "Running on multiple nodes (nnod = %d), must specify shared directory", \
                nnode)
        self.cisolver = block.Block()
        block.Block.set_nproc(nproc, nnode)
        self.cisolver.createTmp(tmp = TmpDir, shared = SharedDir)
        block.Block.reorder = reorder
        self.schedule = block.Schedule(sweeptol = tol)
        self.minM = minM
        self.maxM = maxM
        self.spinAdapted = spinAdapted
        self.bcs = bcs

    def run(self, Ham, M = None, nelec = None, schedule = None, similar = False, restart = True):
        if M is None:
            M = self.maxM
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                nelec = Ham.norb
        if not self.cisolver.sys_initialized:
            self.cisolver.set_system(nelec, 0, self.spinAdapted, \
                    self.bcs, self.spinAdapted)

        if schedule is None:
            schedule = self.schedule
            if self.cisolver.optimized and restart:
                schedule.maxiter = 16
                schedule.gen_restart(M)
            else:
                self.cisolver.optimized = False
                schedule.gen_initial(minM = self.minM, maxM = M)

        self.cisolver.set_schedule(schedule)
        self.cisolver.set_integral(Ham)

        truncation, energy, onepdm = self.cisolver.optimize()
        return onepdm, energy

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.cisolver.onepdm()

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.cisolver.twopdm()

    def cleanup(self):
        # FIXME first copy and save restart files
        self.cisolver.cleanup()


class CASSCF(object):

    settings = {
        "max_stepsize": 0.04,
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
        "ah_decay_rate": 0.7, # augmented hessian decay
        "ci_repsonse_space": 3,
        "dynamic_micro_step": False,
    }

    def __init__(self, ncas, nelecas, MP2natorb = False, spinAverage = False, \
            fcisolver = "FCI", settings = {}):
        log.eassert(ncas * 2 >= nelecas, \
                "CAS size not compatible with number of electrons")
        self.ncas = ncas
        self.nelecas = nelecas # alpha and beta
        self.MP2natorb = MP2natorb
        self.spinAverage = spinAverage
        self.scfsolver = scf.SCF()
        self.mo_coef = None

        # mcscf options, these will be used for first installation of
        # mcscf class: casscf.CASSCF or casscf.DMRGSCF
        self.settings = settings
        if fcisolver.upper() == "FCI":
            self.solver_cls = casscf.CASSCF
        elif fcisolver.upper() == "DMRG":
            log.eassert(self.settings.has_key("fcisolver"), \
                    "When using DMRG-CASSCF, must specify "
                    "the key 'fcisolver' with a BLOCK solver instance")
            self.solver_cls = casscf.DMRGSCF
        else:
            log.error("FCI solver %s is not valid.", fcisolver.upper())
        # solver instance, initialized at first run
        self.solver = None

    def apply_options(self, mc, options):
        for key, val in options.items():
            setattr(mc, key, val)

    def run(self, Ham, mcscf_args = {}, guess = None, nelec = None, \
            similar = False):
        spin = Ham.H1["cd"].shape[0]
        norbs = Ham.H1["cd"].shape[1]
        if nelec is None:
            nelec = Ham.norb
        log.eassert(spin == 2, \
                "spin-restricted CASSCF solver is not implemented")

        nelecasAB = (self.nelecas/2, self.nelecas/2)
        if self.mo_coef is None or not similar: 
            # not restart from previous orbitals
            core, cas, virt, casinfo = get_orbs(self, Ham, guess, nelec)
            self.mo_coef = np.empty((2, norbs, norbs))
            self.mo_coef[:, :, :core.shape[2]] = core
            self.mo_coef[:, :, core.shape[2]:core.shape[2]+cas.shape[2]] = cas
            self.mo_coef[:, :, core.shape[2]+cas.shape[2]:] = virt

        if self.solver is None:
            self.solver = self.solver_cls(self.scfsolver.mf, \
                    self.ncas, nelecasAB, **self.settings)
        else:
            self.solver.refresh(self.scfsolver.mf, self.ncas, nelecasAB)

        # apply options, these options are limited to the CASSCF convergence
        # settings specified as static members
        self.apply_options(self.solver, CASSCF.settings)

        E, _, _, self.mo_coef = self.solver.mc1step(mo_coeff = self.mo_coef, \
                **mcscf_args)
        rho = np.asarray(self.solver.make_rdm1s())

        return rho, E

    def cleanup(self):
        pass
