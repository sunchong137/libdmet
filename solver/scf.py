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
from libdmet.routine.bcs_helper import extractRdm, extractH1
from libdmet import settings

# logger wrapper for pyscf
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

pyscflogger.flush = flush([])
pyscflogger.QUIET = 10

# simple 2e integral transformation

def incore_transform(eri_, c):
    eriA = np.tensordot(c[0][0], eri_[0], (0, 0))
    eriA = np.tensordot(c[1][0], eriA, (0, 1))
    eriA = np.tensordot(eriA, c[3][0], (3, 0))
    eriA = np.tensordot(eriA, c[2][0], (2, 0))
    eriB = np.tensordot(c[0][1], eri_[1], (0, 0))
    eriB = np.tensordot(c[1][1], eriB, (0, 1))
    eriB = np.tensordot(eriB, c[3][1], (3, 0))
    eriB = np.tensordot(eriB, c[2][1], (2, 0))
    eriAB = np.tensordot(c[0][0], eri_[2], (0, 0))
    eriAB = np.tensordot(c[1][0], eriAB, (0, 1))
    eriAB = np.tensordot(eriAB, c[3][1], (3, 0))
    eriAB = np.tensordot(eriAB, c[2][1], (2, 0))
    return np.asarray([eriA, eriB, eriAB])

# unrestricted (integral) hartree-fock routine
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
            if settings.save_mem:
                eri = self._eri[0]
                norb = dm[0].shape[0]
                nImp = eri.shape[0]
                rhoAI = dm[0][:nImp, :nImp]
                rhoBI = dm[1][:nImp, :nImp]
                vj00 = np.tensordot(rhoAI, eri, ((0,1), (0,1)))
                vj11 = np.tensordot(rhoBI, eri, ((0,1), (0,1)))
                vj10 = vj00
                vj01 = vj11
                vk00 = np.tensordot(rhoAI, eri, ((0,1), (0,3)))
                vk11 = np.tensordot(rhoBI, eri, ((0,1), (0,3)))
                va, vb = np.zeros((norb, norb)), np.zeros((norb, norb))
                va[:nImp, :nImp] = vj00 + vj01 - vk00
                vb[:nImp, :nImp] = vj11 + vj10 - vk11
            else:
                vj00 = np.tensordot(dm[0], self._eri[0], ((0,1), (0,1)))
                vj11 = np.tensordot(dm[1], self._eri[1], ((0,1), (0,1)))
                vj10 = np.tensordot(dm[0], self._eri[2], ((0,1), (0,1)))
                vj01 = np.tensordot(self._eri[2], dm[1], ((2,3), (0,1)))
                vk00 = np.tensordot(dm[0], self._eri[0], ((0,1), (0,3)))
                vk11 = np.tensordot(dm[1], self._eri[1], ((0,1), (0,3)))
                va = vj00 + vj01 - vk00
                vb = vj11 + vj10 - vk11
            vhf = np.asarray([va, vb])
        else:
            log.error("Direct SCF not implemented")
        return vhf

    def energy_elec(self, dm, h1e, vhf):
        e1 = np.sum(h1e * dm)
        e_coul = 0.5 * np.sum(vhf * dm)
        log.debug(1, "E_coul = %.15f", e_coul)
        return e1 + e_coul, e_coul

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

# unrestricted Hartree-Fock Bogoliubov

def _UHFB_get_grad(mo_coeff, mo_occ, fock_ao):
    '''RHF Gradients'''
    occidx = np.where(mo_occ> 0)[0]
    viridx = np.where(mo_occ==0)[0]

    fock = reduce(np.dot, (mo_coeff.T.conj(), fock_ao, mo_coeff))
    g = fock[viridx[:,None],occidx]
    return g.reshape(-1)

def _get_veff_bcs(rhoA, rhoB, kappaBA, eri):
    eriA, eriB, eriAB = eri
    vj00 = np.tensordot(rhoA, eriA, ((0,1), (0,1)))
    vj11 = np.tensordot(rhoB, eriB, ((0,1), (0,1)))
    vj10 = np.tensordot(rhoA, eriAB, ((0,1), (0,1)))
    vj01 = np.tensordot(eriAB, rhoB, ((2,3), (0,1)))
    vk00 = np.tensordot(rhoA, eriA, ((0,1), (0,3)))
    vk11 = np.tensordot(rhoB, eriB, ((0,1), (0,3)))
    vl10 = np.tensordot(kappaBA, eriAB, ((1,0), (0,2))) # wrt kappa_ba
    va = vj00 + vj01 - vk00
    vb = vj11 + vj10 - vk11
    vd = vl10
    return va, vb, vd

def _get_veff_bcs_full(rhoA, rhoB, kappaBA, eri, eri2, eri4):
    eriA, eriB, eriAB = eri
    eri2A, eri2B = eri2
    eri4AB = eri4[0]
    vj00 = np.tensordot(rhoA, eriA, ((0,1), (0,1)))
    vj11 = np.tensordot(rhoB, eriB, ((0,1), (0,1)))
    vj10 = np.tensordot(rhoA, eriAB, ((0,1), (0,1)))
    vj01 = np.tensordot(eriAB, rhoB, ((2,3), (0,1)))
    vk00 = np.tensordot(rhoA, eriA, ((0,1), (0,3)))
    vk11 = np.tensordot(rhoB, eriB, ((0,1), (0,3)))
    vl10 = np.tensordot(kappaBA, eriAB, ((1,0), (0,2)))
    vy00 = -np.tensordot(kappaBA, eri2A, ((1,0), (0,2)))
    vy11 = np.tensordot(kappaBA, eri2B, ((0,1), (0,2)))
    vy10 = np.tensordot(rhoA, eri2A, ((0,1), (0,3))) - np.tensordot(rhoB, eri2B, ((0,1), (0,3))).T
    vx10 = np.tensordot(kappaBA, eri4AB, ((1,0), (0,2)))
    va = vj00 + vj01 - vk00 + vy00 + vy00.T
    vb = vj11 + vj10 - vk11 + vy11 + vy11.T
    vd = vl10 + vy10 - vx10
    return va, vb, vd

def _get_veff_bcs_save_mem(rhoA, rhoB, kappaBA, _eri):
    eri = _eri[0]
    nImp = eri.shape[0]
    rhoAI = rhoA[:nImp, :nImp]
    rhoBI = rhoB[:nImp, :nImp]
    kappaBAI = kappaBA[:nImp, :nImp]
    vj00 = np.tensordot(rhoAI, eri, ((0,1), (0,1)))
    vj11 = np.tensordot(rhoBI, eri, ((0,1), (0,1)))
    vj10 = vj00
    vj01 = vj11
    vk00 = np.tensordot(rhoAI, eri, ((0,1), (0,3)))
    vk11 = np.tensordot(rhoBI, eri, ((0,1), (0,3)))
    vl10 = np.tensordot(kappaBAI, eri, ((1,0), (0,2)))# wrt kappa_ba
    va = vj00 + vj01 - vk00
    vb = vj11 + vj10 - vk11
    vd = vl10
    return va, vb, vd

class UHFB(hf.RHF):
    def __init__(self, mol, DiisDim = 12, MaxIter = 30):
        hf.RHF.__init__(self, mol)
        self._keys = self._keys.union(["h1e", "ovlp", "norb", "Mu"])
        self.direct_scf = False
        self.diis_space = DiisDim
        self.max_cycle = MaxIter
        self.h1e = None
        self.ovlp = None
        hf.get_grad = _UHFB_get_grad

    def get_veff(self, mol, dm, dm_last = 0, vhf_last = 0, hermi = 1):
        assert(self._eri is not None)
        assert(self._eri["cccd"] is None or la.norm(self._eri["cccd"]) == 0)
        assert(self._eri["cccc"] is None or la.norm(self._eri["cccc"]) == 0)

        rhoA, rhoB, kappaBA = extractRdm(dm)

        if settings.save_mem:
            va, vb, vd = _get_veff_bcs_save_mem(rhoA, rhoB, kappaBA, \
                    self._eri["ccdd"])
        else:
            va, vb, vd = _get_veff_bcs(rhoA, rhoB, kappaBA, self._eri["ccdd"])

        norb = self.norb
        nv = va.shape[0]
        vhf = np.zeros((norb*2, norb*2))
        vhf[:nv, :nv] = va
        vhf[norb:norb+nv, norb:norb+nv] = -vb
        vhf[:nv, norb:norb+nv] = vd
        vhf[norb:norb+nv, :nv] = vd.T
        return vhf

    def energy_elec(self, dm, h1e, vhf):
        rhoA, rhoB, kappaBA = extractRdm(dm)
        HA, HB, DT = extractH1(h1e)
        HA += np.eye(self.norb) * self.Mu
        HB += np.eye(self.norb) * self.Mu
        VA, VB, VDT = extractH1(vhf)
        e1 = np.sum(rhoA * HA + rhoB * HB + 2 * DT * kappaBA)
        e_coul = 0.5 * np.sum(rhoA * VA + rhoB * VB + 2 * VDT * kappaBA)
        return e1 + e_coul, e_coul

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

    def get_occ(self, mo_energy = None, mo_coeff = None):
        if mo_energy is None: mo_energy = self.mo_energy
        mo_occ = np.zeros_like(mo_energy)
        nocc = self.mol.nelectron // 2
        mo_occ[:nocc] = 1
        pyscflogger.info(self, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            pyscflogger.warn(self, '!! HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
        if self.verbose >= pyscflogger.DEBUG:
            np.set_printoptions(threshold=len(mo_energy))
            pyscflogger.debug(self, '  mo_energy = %s', mo_energy)
            np.set_printoptions()
        return mo_occ

    def get_fock(self, h1e, s1e, vhf, dm, cycle = -1, adiis = None, \
            diis_start_cycle = None, level_shift_factor = None, \
            damp_factor = None):
        return self.get_fock_(h1e, s1e, vhf, dm*2., cycle, adiis, \
                diis_start_cycle, level_shift_factor, damp_factor)


# Newton Raphson method for unrestricted Hartre-Fock Bogoliubov

def gen_g_hop_uhfb(mf, mo_coeff, mo_occ, fock_ao = None):
    mol = mf.mol
    occidx = np.where(mo_occ == 1)[0]
    viridx = np.where(mo_occ == 0)[0]
    nocc, nvir = len(occidx), len(viridx)

    if fock_ao is None:
        # GRho
        dm1 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_hcore() + mf.get_veff(mol, dm1)
    fock = mdot(mo_coeff.T, fock_ao, mo_coeff)

    g = fock[viridx[:, None], occidx] * 2

    foo = fock[occidx[:, None], occidx]
    fvv = fock[viridx[:, None], viridx]

    # approximated by the non-interacting part
    h_diag = (fvv.diagonal().reshape(-1,1) - foo.diagonal()) * 2

    def h_op(x):
        x = x.reshape(nvir, nocc)
        x2 = (np.dot(fvv, x) - np.dot(x, foo)) * 2

        d1 = mdot(mo_coeff[:, viridx], x, mo_coeff[:, occidx].T)
        grho = d1 + d1.T
        nmo = mo_occ.shape[0] / 2
        grho[nmo:, nmo:] += np.eye(nmo)
        dvhf = mf.get_veff(mol, grho)

        x2 += 0.5 * mdot(mo_coeff[:, viridx].T, dvhf, mo_coeff[:, occidx]) * 4

        return x2.reshape(-1)

    return g.reshape(-1), h_op, h_diag.reshape(-1)

def newton(mf):
    assert(isinstance(mf, UHFB))
    class newtonUHFB(UHFB):
        def __init__(self):
            self._scf = mf
            self.max_stepsize = 0.05

            self.ah_start_tol = 5.
            # the following three are the most useful options
            # seems 6 / 0. / 15 gives good results
            self.ah_start_cycle = 6
            self.ah_level_shift = 0.
            self.max_cycle_inner = 15
            self.canonicalization = False

            self.ah_conv_tol = 1e-12
            self.ah_lindep = 1e-14
            self.ah_max_cycle = 30
            self.ah_grad_trust_region = 3.
            self.ah_decay_rate = .8
            self.keyframe_interval = 5
            self.keyframe_interval_rate = 1.
            self_keys = set(self.__dict__.keys())

            self.__dict__.update(mf.__dict__)
            self._keys = self_keys.union(mf._keys)

        def dump_flags(self):
            pyscflogger.info(self, '\n')
            pyscflogger.info(self, '******** SCF Newton Raphson flags ********')
            pyscflogger.info(self, 'SCF tol = %g', self.conv_tol)
            pyscflogger.info(self, 'conv_tol_grad = %s',    self.conv_tol_grad)
            pyscflogger.info(self, 'max. SCF cycles = %d', self.max_cycle)
            pyscflogger.info(self, 'direct_scf = %s', self._scf.direct_scf)
            if self._scf.direct_scf:
                pyscflogger.info(self, 'direct_scf_tol = %g', self._scf.direct_scf_tol)
            if self.chkfile:
                pyscflogger.info(self, 'chkfile to save SCF result = %s', self.chkfile)
            pyscflogger.info(self, 'max_cycle_inner = %d',  self.max_cycle_inner)
            pyscflogger.info(self, 'max_stepsize = %g', self.max_stepsize)
            pyscflogger.info(self, 'ah_start_tol = %g',     self.ah_start_tol)
            pyscflogger.info(self, 'ah_level_shift = %g',   self.ah_level_shift)
            pyscflogger.info(self, 'ah_conv_tol = %g',      self.ah_conv_tol)
            pyscflogger.info(self, 'ah_lindep = %g',        self.ah_lindep)
            pyscflogger.info(self, 'ah_start_cycle = %d',   self.ah_start_cycle)
            pyscflogger.info(self, 'ah_max_cycle = %d',     self.ah_max_cycle)
            pyscflogger.info(self, 'ah_grad_trust_region = %g', self.ah_grad_trust_region)
            pyscflogger.info(self, 'keyframe_interval = %d', self.keyframe_interval)
            pyscflogger.info(self, 'keyframe_interval_rate = %g', self.keyframe_interval_rate)
            pyscflogger.info(self, 'augmented hessian decay rate = %g', self.ah_decay_rate)
            pyscflogger.info(self, 'max_memory %d MB (current use %d MB)',
                     self.max_memory, pyscf.lib.current_memory()[0])

        def get_fock_(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                      diis_start_cycle=None, level_shift_factor=None,
                      damp_factor=None):
            return h1e + vhf

        def from_dm(self, dm):
            '''Transform density matrix to the initial guess'''
            mol = self.mol
            h1e = self.get_hcore(mol)
            s1e = self.get_ovlp(mol)
            vhf = self._scf.get_veff(mol, dm)
            fock = self.get_fock(h1e, s1e, vhf, dm, 0, None)
            mo_energy, mo_coeff = self.get_mo_energy(fock, s1e, dm)
            mo_occ = self.get_occ(mo_energy, mo_coeff)
            return mo_coeff, mo_occ

        def gen_g_hop(self, mo_coeff, mo_occ, fock_ao = None, h1e = None):
            return gen_g_hop_uhfb(self, mo_coeff, mo_occ, fock_ao)

        def update_rotate_matrix(self, dx, mo_occ, u0 = 1):
            import scipy
            nmo = len(mo_occ)
            occidx = mo_occ == 1
            viridx = np.logical_not(occidx)
            dr = np.zeros((nmo, nmo))
            dr[viridx.reshape(-1,1) & occidx] = dx
            dr = dr - dr.T
            return np.dot(u0, scipy.linalg.expm(dr))

        def rotate_mo(self, mo_coeff, u, log = None):
            return np.dot(mo_coeff, u)

        def get_mo_energy(self, fock, s1e, dm):
            return self.eig(fock, s1e)

    return newtonUHFB()

# unrestricted MP2

def UMP2_kernel(mp, mo_coeff, mo_energy, nocc):
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
            nocc = self._scf.nelec

        self.E, self.t2 = UMP2_kernel(self, mo, mo_energy, nocc)

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

# main class for uihf, uhfb and ump2

class SCF(object):
    def __init__(self, tmp = "/tmp", newton_ah = True):
        self.sys_initialized = False
        self.integral_initialized = False
        self.doneHF = False
        self.newton_ah = newton_ah
        log.debug(0, "Using pyscf version %s", pyscf.__version__)
        if self.newton_ah:
            if log.Level[log.verbose] <= log.Level["RESULT"]:
                pyscflogger.flush.addkey("macro X")
            elif log.Level[log.verbose] <= log.Level["INFO"]:
                pyscflogger.flush.addkey("macro")
            else:
                pyscflogger.flush = flush([""])
        else:
            if log.Level[log.verbose] <= log.Level["INFO"]:
                pyscflogger.flush.addkey("cycle=")
            else:
                pyscflogger.flush = flush([""])

    def set_system(self, nelec, spin, bogoliubov, spinRestricted):
        log.eassert(not spinRestricted, "Only spin-unrestricted version is implemented")
        if bogoliubov:
            log.eassert(nelec is None, "nelec cannot be specified when doing BCS calculations")
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

        self.mol.nelectron = self.nelec
        self.mol.spin = self.spin
        self.sys_initialized = True

    def set_integral(self, *args):
        log.eassert(self.sys_initialized, "set_integral() should be used after initializing set_system()")
        if len(args) == 1:
            log.eassert(self.bogoliubov == args[0].bogoliubov, \
                    "Integral is not consistent with system type")
            self.integral = args[0]
        elif len(args) == 4:
            self.integral = integral.Integral(args[0], self.spinRestricted, self.bogoliubov, *args[1:])
        else:
            log.error("input either an integral object, or (norb, H0, H1, H2)")
        self.integral_initialized = True
        self.mol.energy_nuc = lambda *args: self.integral.H0
        if self.bogoliubov:
            self.mol.nelectron = self.integral.norb*2

    def HF(self, DiisDim = 12, MaxIter = 30, InitGuess = None, tol = 1e-6, Mu = None):
        log.eassert(self.sys_initialized and self.integral_initialized, \
                "components for Hartree-Fock (Bogoliubov) calculation are not ready"
                "\nsys_init = %s\nint_init = %s", \
                self.sys_initialized, self.integral_initialized)
        if self.bogoliubov:
            return self.HFB(0., DiisDim, MaxIter, InitGuess, tol)

        # otherwise do UHF
        if not self.spinRestricted:
            log.result("Unrestricted Hartree-Fock with pyscf")
            self.mf = UIHF(self.mol, DiisDim = DiisDim, MaxIter = MaxIter)
            self.mf.h1e = self.integral.H1["cd"]
            self.mf.ovlp = np.eye(self.integral.norb)
            self.mf._eri = self.integral.H2["ccdd"] #vaa, vbb, vab
            self.mf.conv_tol = tol

            if self.newton_ah:
                from pyscf.scf.newton_ah import kernel, newton

                if InitGuess is not None:

                    mo_occ = [None, None]
                    mo_coeff = [None, None]
                    mo_occ[0], mo_coeff[0] = la.eigh(InitGuess[0])
                    mo_occ[1], mo_coeff[1] = la.eigh(InitGuess[1])

                    newtonUIHF = newton(self.mf)
                    newtonUIHF.max_cycle_inner = 15
                    newtonUIHF.ah_start_cycle = 6
                    newtonUIHF.dump_flags()
                    conv, E, mo_energy, mo_coeff, mo_occ = kernel(newtonUIHF, \
                            tuple(mo_coeff), tuple(mo_occ), \
                            max_cycle=50, conv_tol = self.mf.conv_tol, verbose=5)
                else:
                    self.mf.max_cycle = 0
                    self.mf.scf(np.zeros((2, self.integral.norb, self.integral.norb)))

                    newtonUIHF = newton(self.mf)
                    newtonUIHF.max_cycle_inner = 15
                    newtonUIHF.ah_start_cycle = 6
                    newtonUIHF.dump_flags()
                    conv, E, mo_energy, mo_coeff, mo_occ = kernel(newtonUIHF, \
                            self.mf.mo_coeff, self.mf.mo_occ, \
                            max_cycle=50, conv_tol = self.mf.conv_tol, verbose=5)

                self.mf.mo_energy = mo_energy
                self.mf.mo_coeff = mo_coeff
                self.mf.mo_occ = mo_occ
                self.mf.converged = conv

            else:
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

    def HFB(self, Mu, DiisDim = 12, MaxIter = 30, InitGuess = None, tol = 1e-6):
        log.eassert(self.sys_initialized and self.integral_initialized, \
                "components for Hartree-Fock Bogoliubov calculation are not ready"
                "\nsys_init = %s\nint_init = %s", \
                self.sys_initialized, self.integral_initialized)

        norb = self.integral.norb
        if not self.spinRestricted:
            log.result("Unrestricted Hartree-Fock-Bogoliubov with pyscf")
            self.mf = UHFB(self.mol, DiisDim = DiisDim, MaxIter = MaxIter)
            h1e = np.empty((norb*2, norb*2))
            self.mf.Mu = Mu
            self.mf.norb = norb
            h1e[:norb, :norb] = self.integral.H1["cd"][0] - np.eye(norb) * Mu
            h1e[norb:, norb:] = -(self.integral.H1["cd"][1] - np.eye(norb) * Mu)
            h1e[:norb, norb:] = self.integral.H1["cc"][0]
            h1e[norb:, :norb] = self.integral.H1["cc"][0].T
            self.mf.h1e = h1e
            self.mf.ovlp = np.eye(norb*2)
            self.mf._eri = self.integral.H2 # we can have cccd and cccc terms
            self.mf.conv_tol = tol

            if self.newton_ah:
                from pyscf.scf.newton_ah import kernel

                if InitGuess is not None:
                    # use traditional HF: uncomment this
                    # E = self.mf.scf(InitGuess)

                    # get orbitals for initial guess, and round out fractions in occupation number
                    mo_occ, mo_coeff = la.eigh(InitGuess)
                    nmo = mo_occ.shape[0]/2
                    mo_occ[:nmo], mo_occ[nmo:] = 0, 1

                    newtonUHFB = newton(self.mf)
                    newtonUHFB.dump_flags()

                    conv, E, mo_energy, mo_coeff, mo_occ = kernel(newtonUHFB, mo_coeff, mo_occ, \
                            max_cycle=50, conv_tol = self.mf.conv_tol, verbose=5)
                else:
                    # do an initial traditional HF to generate orbitals
                    self.mf.max_cycle = 1
                    self.mf.scf(np.zeros((norb*2, norb*2)))

                    newtonUHFB = newton(self.mf)
                    newtonUHFB.dump_flags()

                    conv, E, mo_energy, mo_coeff, mo_occ = kernel(newtonUHFB, \
                            self.mf.mo_coeff, self.mf.mo_occ, \
                            max_cycle=50, conv_tol = self.mf.conv_tol, verbose=5)

                self.mf.mo_energy = mo_energy
                self.mf.mo_coeff = mo_coeff
                self.mf.mo_occ = mo_occ
                self.mf.converged = conv

            else:
                if InitGuess is not None:
                    E = self.mf.scf(InitGuess)
                else:
                    E = self.mf.scf(np.zeros((norb*2, norb*2)))

            coefs = self.mf.mo_coeff
            occs = self.mf.mo_occ
            GRho = mdot(coefs, np.diag(occs), coefs.T)
        else:
            log.error("Restricted Hartree-Fock-Bogoliubov not implemented yet")

        log.result("Hartree-Fock-Bogoliubov convergence: %s", self.mf.converged)
        log.result("Hartree-Fock-Bogoliubov energy = %20.12f", E)
        self.doneHF = True
        return E, GRho

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

    def get_mo_energy(self):
        log.eassert(self.doneHF, "Hartree-Fock calculation is not done")
        return np.asarray(self.mf.mo_energy)

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

    # UHF
    scf.set_system(8, 0, False, False)
    scf.set_integral(8, 0, {"cd": Int1e}, \
            {"ccdd": Int2e})
    _, rhoHF = scf.HF(MaxIter = 100, tol = 1e-3, \
        InitGuess = (
            np.diag([1,0,1,0,1,0,1,0]),
            np.diag([0,1,0,1,0,1,0,1])
        ))
    log.result("HF density matrix:\n%s\n%s", rhoHF[0], rhoHF[1])

    # UHFB
    np.random.seed(8)
    scf.set_system(None, 0, True, False)
    scf.set_integral(8, 0, {"cd": Int1e, "cc": np.random.rand(1,8,8) * 0.1}, \
            {"ccdd": Int2e, "cccd": None, "cccc": None})
    _, GRhoHFB = scf.HF(MaxIter = 100, tol = 1e-3, Mu = 2.02, \
        InitGuess = np.diag([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    rhoA, rhoB, kappaBA = extractRdm(GRhoHFB)
    log.result("HFB density matrix:\n%s\n%s\n%s", rhoA, rhoB, -kappaBA.T)
