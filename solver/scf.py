import numpy as np
import numpy.linalg as la
import pyscf
from pyscf import gto, ao2mo
from pyscf.scf.uhf import UHF
from pyscf.scf import hf
from pyscf.mp.mp2 import MP2
import pyscf.lib.logger as pyscflogger
import libdmet.utils.logger as log
from libdmet.system import integral
from libdmet.utils.misc import mdot


class flush(object):
    def __init__(self, keywords):
        self.keywords = set(keywords)

    def addkey(self, key):
        self.keywords.add(key)

    def addkeys(self, keys):
        self.keywords.union(keys)

    def has_keyword(self, args):
        for arg in map(str, args):
            for key in self.keywords:
                if key in arg:
                    return True
        return False

    def __call__(self, object, *args):
        if self.has_keyword(args):
            log.result(*args)

pyscflogger.flush = flush([""])
pyscflogger.QUIET = 10

class UIHF(UHF):
    """
    a routine for unrestricted HF with integrals different for two spin species
    """
    def __init__(self, mol, DiisDim = 12, MaxIter = 30):
        UHF.__init__(self, mol)
        self._keys = self._keys.union(['h1e', 'ovlp'])
        self.direct_scf = False
        self.diis_space = DiisDim
        self.max_cycle = MaxIter
        self.h1e = None
        self.ovlp = None

    def get_veff(self, mol, dm, dm_last = 0, vhf_last = 0, hermi = 1):
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array([dm*0.5, dm*0.5])
        if self._eri is not None:
            vj00 = np.tensordot(dm[0], self._eri[0], ((0,1), (0,1)))
            vj11 = np.tensordot(dm[1], self._eri[1], ((0,1), (0,1)))
            vj10 = np.tensordot(dm[0], self._eri[2], ((0,1), (0,1)))
            vj01 = np.tensordot(dm[1], self._eri[2], ((1,0), (3,2)))
            vk00 = np.tensordot(dm[0], self._eri[0], ((0,1), (0,3)))
            vk11 = np.tensordot(dm[1], self._eri[1], ((0,1), (0,3)))
            va = vj00 + vj01 - vk00
            vb = vj11 + vj10 - vk11
            vhf = np.asarray([va, vb])
        else:
            log.error("Direct SCF not implemented")
        return vhf

    def get_fock(self, h1e, s1e, vhf, dm, cycle = -1, adiis = None, *args):
        f = h1e + vhf
        if 0 <= cycle < self.diis_start_cycle - 1:
            f = np.asarray([
                hf.damping(s1e, dm[0], f[0], self.damp_factor),
                hf.damping(s1e, dm[1], f[1], self.damp_factor),
            ])
            f = np.asarray([
                hf.level_shift(s1e, dm[0], f[0], self.level_shift_factor),
                hf.level_shift(s1e, dm[1], f[1], self.level_shift_factor),
            ])
        elif 0 <= cycle:
            fac = self.level_shift_factor * np.exp(self.diis_start_cycle - cycle - 1)
            f = np.asarray([
                hf.level_shift(s1e, dm[0], f[0], fac),
                hf.level_shift(s1e, dm[1], f[1], fac)
            ])
        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dm, np.array(f))
        return f

    def energy_elec(self, dm, h1e, vhf):
        e1 = np.sum(h1e * dm)
        e_coul = 0.5 * np.sum(vhf * dm)
        log.debug(1, "E_coul = %.15f", e_coul)
        return e1 + e_coul, e_coul

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

def incore_transform(eri_, c):
    log.debug(1, "Int2e component 0")
    eriA = np.tensordot(c[0][0], eri_[0], (0, 0))
    eriA = np.swapaxes(np.tensordot(c[1][0], eriA, (0, 1)), 0, 1)
    eriA = np.swapaxes(np.tensordot(eriA, c[2][0], (2, 0)), 2, 3)
    eriA = np.tensordot(eriA, c[3][0], (3, 0))
    log.debug(1, "Int2e component 1")
    eriB = np.tensordot(c[0][1], eri_[1], (0, 0))
    eriB = np.swapaxes(np.tensordot(c[1][1], eriB, (0, 1)), 0, 1)
    eriB = np.swapaxes(np.tensordot(eriB, c[2][1], (2, 0)), 2, 3)
    eriB = np.tensordot(eriB, c[3][1], (3, 0))
    log.debug(1, "Int2e component 2")
    eriAB = np.tensordot(c[0][0], eri_[2], (0, 0))
    eriAB = np.swapaxes(np.tensordot(c[1][0], eriAB, (0, 1)), 0, 1)
    eriAB = np.swapaxes(np.tensordot(eriAB, c[2][1], (2, 0)), 2, 3)
    eriAB = np.tensordot(eriAB, c[3][1], (3, 0))
    return np.asarray([eriA, eriB, eriAB])


def kernel(mp, mo_coeff, mo_energy, nocc):
    log.debug(0, "transforming integral for MP2")
    ovov = mp.ao2mo(mo_coeff, nocc)
    nmo = mo_coeff[0].shape[1]
    nvir = (nmo - nocc[0], nmo - nocc[1])
    epsilon_ia = np.asarray([
        mo_energy[0][:nocc[0], None] - mo_energy[0][None, nocc[0]:],
        mo_energy[1][:nocc[1], None] - mo_energy[1][None, nocc[1]:]
    ])
    t2 = np.asarray([
        np.empty((nocc[0], nocc[0], nvir[0], nvir[0])),
        np.empty((nocc[1], nocc[1], nvir[1], nvir[1])),
        np.empty((nocc[0], nocc[1], nvir[1], nvir[0]))
    ])
    E = 0.
    for s in range(2): # for 2 pure spins
        log.debug(0, "computing amplitudes: spin component %d of 2", s)
        for i in range(nocc[s]):
            # three-index intermediate
            d_jba = (epsilon_ia[s].reshape(-1,1) + epsilon_ia[s,i].reshape(1,-1)).ravel()
            g_i = ovov[s,i].transpose(1,2,0)
            t2[s,i] = (g_i.ravel()/d_jba).reshape(nocc[s], nvir[s], nvir[s])
            theta = g_i - np.swapaxes(g_i, 1, 2)
            E += 0.25 * np.tensordot(t2[s,i], theta, ((0,1,2), (0,1,2)))

    # alpha-beta part
    for i in range(nocc[0]):
        d_jba = (epsilon_ia[1].reshape(-1,1) + epsilon_ia[0][i].reshape(1,-1)).ravel()
        g_i = ovov[2,i].transpose(1,2,0)
        t2[2,i] = (g_i.ravel()/d_jba).reshape(nocc[1], nvir[0], nvir[1])
        E += 0.5 * np.tensordot(t2[2,i], g_i, ((0,1,2), (0,1,2)))

    return E, t2

class UMP2(MP2):
    def __init__(self, mf):
        MP2.__init__(self, mf)

    def run(self, mo = None, mo_energy = None, nocc = None):
        if mo is None:
            mo = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if nocc is None:
            nocc = (self._scf.nelectron_alpha, self._scf.mol.nelectron - self._scf.nelectron_alpha)

        self.E, self.t2 = kernel(self, mo, mo_energy, nocc)

        log.info("MP2 correlation energy = %20.12f", self.E)
        return self.E, self.t2

    def ao2mo(self, mo_coeff, nocc):
        nmo = mo_coeff[0].shape[1]
        nvir = (nmo - nocc[0], nmo - nocc[1])
        co = np.asarray([mo_coeff[0][:, :nocc[0]], mo_coeff[1][:, :nocc[1]]])
        cv = np.asarray([mo_coeff[0][:, nocc[0]:], mo_coeff[1][:, nocc[1]:]])
        if self._scf._eri is not None:
            eri = incore_transform(self._scf._eri, (co, cv, co, cv))
        else:
            log.error("on-disk calculation not implemented")
        return eri

    def onepdm(self):
        rdm_v_a = 0.5 * (np.tensordot(self.t2[0], self.t2[0], ((0,1,2), (0,1,2))) +
            np.tensordot(self.t2[2], self.t2[2], ((0,1,2), (0,1,2))))
        rdm_v_b = 0.5 * (np.tensordot(self.t2[1], self.t2[1], ((0,1,2), (0,1,2))) +
            np.tensordot(self.t2[2], self.t2[2], ((0,1,3), (0,1,3))))
        rdm_c_a = -0.5 * (np.tensordot(self.t2[0], self.t2[0], ((1,2,3), (1,2,3))) +
            np.tensordot(self.t2[2], self.t2[2], ((1,2,3), (1,2,3))))
        rdm_c_b = -0.5 * (np.tensordot(self.t2[1], self.t2[1], ((1,2,3), (1,2,3))) +
            np.tensordot(self.t2[2], self.t2[2], ((0,2,3), (0,2,3))))
        nocc = (rdm_c_a.shape[0], rdm_c_b.shape[0])
        nmo = rdm_c_a.shape[0] + rdm_v_a.shape[0]
        rdm_a = np.zeros((nmo, nmo))
        rdm_b = np.zeros((nmo, nmo))
        rdm_a[:nocc[0], :nocc[0]] = np.eye(nocc[0]) + rdm_c_a
        rdm_a[nocc[0]:, nocc[0]:] = rdm_v_a
        rdm_b[:nocc[1], :nocc[1]] = np.eye(nocc[1]) + rdm_c_b
        rdm_b[nocc[1]:, nocc[1]:] = rdm_v_b
        return np.asarray([rdm_a, rdm_b])

class SCF(object):
    def __init__(self, tmp = "/tmp"):
        self.sys_initialized = False
        self.integral_initialized = False
        self.doneHF = False
        log.debug(0, "Using pyscf version %s", pyscf.__version__)

    def set_system(self, nelec, spin, bogoliubov, spinRestricted):
        log.eassert(not bogoliubov, "Hartree-Fock-Bogoliubov and MP2 not implemented yet")
        log.eassert(not spinRestricted, "Only spin-unrestricted version is implemented")
        self.nelec = nelec
        self.spin = spin
        self.bogoliubov = bogoliubov
        self.spinRestricted = spinRestricted
        self.mol = gto.Mole()
        self.mol.dump_input = lambda *args: 0 # do not dump input file
        if log.Level[log.verbose] >= log.Level["RESULT"]:
            self.mol.build(verbose = 4)
        else:
            self.mol.build(verbose = 2)
        if log.Level[log.verbose] <= log.Level["INFO"]:
            pyscflogger.flush = flush(["cycle="])
            
        self.mol.nelectron = self.nelec
        self.mol.spin = self.spin
        self.sys_initialized = True

    def set_integral(self, *args):
        log.eassert(self.sys_initialized, "set_integral() should be used after initializing set_system()")
        if len(args) == 1:
            self.integral = args[0]
        elif len(args) == 4:
            self.integral = integral.Integral(args[0], self.spinRestricted, self.bogoliubov, *args[1:])
        else:
            log.error("input either an integral object, or (norb, H0, H1, H2)")
        self.integral_initialized = True
        self.mol.nuclear_repulsion = lambda *args: self.integral.H0

    def HF(self, DiisDim = 12, MaxIter = 30, InitGuess = None, tol = 1e-6):
        log.eassert(self.sys_initialized and self.integral_initialized, \
            "components for Hartree-Fock calculation are not ready\nsys_init = %s\nint_init = %s", \
            self.sys_initialized, self.integral_initialized)
        if not self.spinRestricted:
            log.result("Unrestricted Hartree-Fock with pyscf")
            self.mf = UIHF(self.mol, DiisDim = DiisDim, MaxIter = MaxIter)
            self.mf.h1e = self.integral.H1["cd"]
            self.mf.ovlp = np.eye(self.integral.norb)
            self.mf._eri = self.integral.H2["ccdd"] #vaa, vbb, vab
            self.mf.conv_tol = tol
            if InitGuess is not None:
                E = self.mf.scf(InitGuess)
            else:
                E = self.mf.scf(np.zeros((2, self.integral.norb, self.integral.norb)))

            coefs = self.mf.mo_coeff
            occs = self.mf.mo_occ
            rho = np.asarray([mdot(coefs[0], np.diag(occs[0]), coefs[0].T), \
                mdot(coefs[1], np.diag(occs[1]), coefs[1].T)])
        else:
            log.error("Restricted Hartree-Fock not interfaced yet")

        log.result("Hartree-Fock convergence: %s", self.mf.converged)
        log.result("Hartree-Fock energy = %20.12f", E)
        self.doneHF = True
        return E, rho

    def MP2(self, mo = False):
        if not self.spinRestricted:
            log.result("Unrestricted MP2 with pyscf")
            if not self.doneHF:
                log.warning("running HF first with default settings")
                self.HF()
            log.check(self.mf.converged, "Hartree-Fock calculation has not converged")
            self.mp = UMP2(self.mf)
            E, t2 = self.mp.run()
            rho = self.mp.onepdm() # this is rho in mo basis
            if not mo:
                coefs = self.mf.mo_coeff
                rho = np.asarray([mdot(coefs[0], rho[0], coefs[0].T),
                    mdot(coefs[1], rho[1], coefs[1].T)])
        else:
            log.error("Restricted MP2 not interfaced yet")
        return E, rho

    def get_mo(self):
        log.eassert(self.doneHF, "Hartree-Fock calculation is not done")
        return np.asarray(self.mf.mo_coeff)

if __name__ == "__main__":
    log.verbose = "INFO"
    Int1e = -np.eye(8, k = 1)
    Int1e[0, 7] = -1
    Int1e += Int1e.T
    Int1e = np.asarray([Int1e, Int1e])
    Int2e = np.zeros((3,8,8,8,8))

    for i in range(8):
        Int2e[0,i,i,i,i] = 4
        Int2e[1,i,i,i,i] = 4
        Int2e[2,i,i,i,i] = 4

    scf = SCF()
    scf.set_system(8, 0, False, False)
    scf.set_integral(8, 0, {"cd": Int1e}, {"ccdd": Int2e})
    _, rhoHF = scf.HF(MaxIter = 100, tol = 1e-3, \
        InitGuess = (np.diag([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]), np.diag([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])))
    log.result("HF density matrix:\n%s\n%s", rhoHF[0], rhoHF[1])
    _, rhoMP = scf.MP2()
    log.result("MP2 density matrix:\n%s\n%s", rhoMP[0], rhoMP[1])

