#!/usr/bin/env python
import numpy as np
import numpy.linalg as la
from pyscf import gto
from pyscf.scf.hf import UHF, damping, level_shift
from pyscf.mp.mp2 import MP2
from pyscf import ao2mo
from pyscf.lib import logger as log
import time
import os

class UIHF(UHF):
  def __init__(self, mol):
    UHF.__init__(self, mol)
    self.direct_scf = False
    self.diis_space = 12
    
  def get_veff(self, mol, dm, dm_last = 0, vhf_last = 0, hermi = 1):
    '''UHF Columb repulsion with different spin orbitals'''
    t0 = (time.clock(), time.time())
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
      dm = np.array((dm*.5,dm*.5))
    if self._eri is not None:
      vj00 = np.tensordot(dm[0], self._eri[0], ((0,1), (0,1)))
      vj11 = np.tensordot(dm[1], self._eri[1], ((0,1), (0,1)))
      vj10 = np.tensordot(dm[0], self._eri[2], ((0,1), (0,1)))
      vj01 = np.tensordot(dm[1], self._eri[2], ((1,0), (3,2)))
      vk00 = np.tensordot(dm[0], self._eri[0], ((0,1), (0,3)))
      vk11 = np.tensordot(dm[1], self._eri[1], ((0,1), (0,3)))
      v_a = vj00 + vj01 - vk00
      v_b = vj11 + vj10 - vk11
      vhf = np.array((v_a, v_b))
    else:
      raise Exception("Direct SCF not implemented")
    log.timer(self, 'vj and vk', *t0)
    return vhf

  def make_fock(self, h1e, s1e, vhf, dm, cycle = -1, adiis = None):
    f = (h1e[0]+vhf[0], h1e[1]+vhf[1])
    if 0 <= cycle < self.diis_start_cycle-1:
      f = (damping(s1e, dm[0], f[0], self.damp_factor), \
           damping(s1e, dm[1], f[1], self.damp_factor))
      f = (level_shift(s1e, dm[0], f[0], self.level_shift_factor), \
           level_shift(s1e, dm[1], f[1], self.level_shift_factor))
    elif 0 <= cycle:
      fac = self.level_shift_factor \
          * np.exp(self.diis_start_cycle-cycle-1)
      f = (level_shift(s1e, dm[0], f[0], fac), \
           level_shift(s1e, dm[1], f[1], fac))
    if adiis is not None and cycle >= self.diis_start_cycle:
      f = adiis.update(s1e, dm, np.array(f))
    return f

  def calc_tot_elec_energy(self, h1e, vhf, dm):
    e1 = np.einsum('ij,ij', h1e[0], dm[0]) + np.einsum('ij,ij', h1e[1], dm[1])
    e_coul = np.einsum('ij,ji', dm[0], vhf[0]) \
           + np.einsum('ij,ji', dm[1], vhf[1])
    e_coul *= .5
    log.debug1(self, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul

def incore_transform_u(eri_, c):
  eri = [None,] * 3
  eri[0] = np.tensordot(c[0][0], eri_[0], (0, 0))
  eri[0] = np.swapaxes(np.tensordot(c[1][0], eri[0], (0, 1)), 0, 1)
  eri[0] = np.swapaxes(np.tensordot(eri[0], c[2][0], (2, 0)), 2, 3)
  eri[0] = np.tensordot(eri[0], c[3][0], (3, 0))
  eri[1] = np.tensordot(c[0][1], eri_[1], (0, 0))
  eri[1] = np.swapaxes(np.tensordot(c[1][1], eri[1], (0, 1)), 0, 1)
  eri[1] = np.swapaxes(np.tensordot(eri[1], c[2][1], (2, 0)), 2, 3)
  eri[1] = np.tensordot(eri[1], c[3][1], (3, 0))
  eri[2] = np.tensordot(c[0][0], eri_[2], (0, 0))
  eri[2] = np.swapaxes(np.tensordot(c[1][0], eri[2], (0, 1)), 0, 1)
  eri[2] = np.swapaxes(np.tensordot(eri[2], c[2][1], (2, 0)), 2, 3)
  eri[2] = np.tensordot(eri[2], c[3][1], (3, 0))
  return tuple(eri)

def kernel_u(mp, mo_coeff, mo_energy, nocc, verbose = None):
  ovov = mp.ao2mo(mo_coeff, nocc)
  nvir = (len(mo_energy[0]) - nocc[0], len(mo_energy[1]) - nocc[1])
  eia = (mo_energy[0][:nocc[0], None] - mo_energy[0][None, nocc[0]:],
         mo_energy[1][:nocc[1], None] - mo_energy[1][None, nocc[1]:])
  t2 = [np.empty((nocc[0], nocc[0], nvir[0], nvir[0])),
        np.empty((nocc[1], nocc[1], nvir[1], nvir[1])),
        np.empty((nocc[0], nocc[1], nvir[1], nvir[0]))]
  emp2 = 0
  for s in range(2): # spin
    for i in range(nocc[s]):
      djba = (eia[s].reshape(-1,1) + eia[s][i].reshape(1,-1)).ravel()
      gi = ovov[s][i].transpose(1,2,0)
      t2[s][i] = (gi.ravel()/djba).reshape(nocc[s], nvir[s], nvir[s])
      theta = gi - np.swapaxes(gi, 1, 2)
      emp2 += 0.25 * np.tensordot(t2[s][i], theta, ((0,1,2), (0,1,2)))

  for i in range(nocc[0]):
    djba = (eia[1].reshape(-1,1) + eia[0][i].reshape(1,-1)).ravel()
    gi = ovov[2][i].transpose(1,2,0)
    t2[2][i] = (gi.ravel()/djba).reshape(nocc[1], nvir[0], nvir[1])
    emp2 += 0.5 * np.tensordot(t2[2][i], gi, ((0,1,2), (0,1,2)))

  return emp2, tuple(t2)

class UMP2(MP2):
  def __init__(self, mf):
    MP2.__init__(self, mf)

  def run(self, mo = None, mo_energy = None, nocc = None):
    if mo is None:
      mo = self._scf.mo_coeff
    if mo_energy is None:
      mo_energy = self._scf.mo_energy
    if nocc is None:
      nocc = (self._scf.nelectron_alpha, 
              self._scf.mol.nelectron-self._scf.nelectron_alpha)

    self.emp2, self.t2 = \
        kernel_u(self, mo, mo_energy, nocc, verbose=self.verbose)
    
    log.log(self, "UMP2 energy = %.15g", self.emp2)
    return self.emp2, self.t2

  def make_rdm(self):
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
    return (rdm_a, rdm_b)
    
  def ao2mo(self, mo, nocc):
    print "transform integrals..."
    nmo = mo[0].shape[1]
    nvir = (nmo - nocc[0], nocc[1])
    co = (mo[0][:, :nocc[0]], mo[1][:, :nocc[1]])
    cv = (mo[0][:, nocc[0]:], mo[1][:, nocc[1]:])
    if self._scf._eri is not None:
      eri = incore_transform_u(self._scf._eri, (co, cv, co, cv))
    else:
      raise Exception("On-disk calculation not implemented")
    print "integral transformed"
    return eri

if __name__ == "__main__":

  mol = gto.Mole()
  mol.build(verbose = 4)
  mol.nelectron = 176
  norbs = 176  
  mol.intor_symmetric = lambda *args: np.eye(norbs)
  mol.nuclear_repulsion = lambda *args: 0.

  mf = UIHF(mol)
  
  print "loading integrals..."
  h1 = np.load("Int1e.npy")

  eri = (np.load("Int2eAA.npy"), np.load("Int2eBB.npy"), np.load("Int2eAB.npy"))
  print "integrals loaded"
  
  mf.get_hcore = lambda *args: (h1[::2, ::2], h1[1::2, 1::2])
  mf.get_ovlp = lambda *args: np.eye(norbs)
  mf.max_cycle = 20
  mf._eri = eri
  
  #ew, ev = la.eigh(h1)
  #dm = np.dot(ev[:, :mol.nelectron], ev[:, :mol.nelectron].T)
  
  if os.path.isfile("rdmA.npy") and os.path.isfile("rdmB.npy"):
    print "loading initial guess..."
    dm = (np.load("rdmA.npy"), np.load("rdmB.npy"))
  else:
    print "computing initial guess..."
    occs = np.zeros(norbs)
    occs[:mol.nelectron/2] = 1
    dm = (np.diag(occs), np.diag(occs))

  mf.make_init_guess = lambda *args: (0, dm)
  
  mf.scf()

  dm = (np.dot(mf.mo_coeff[0], np.dot(np.diag(mf.mo_occ[0]), mf.mo_coeff[0].T)),
        np.dot(mf.mo_coeff[1], np.dot(np.diag(mf.mo_occ[1]), mf.mo_coeff[1].T)))

  np.save("coefA.npy", mf.mo_coeff[0])
  np.save("coefB.npy", mf.mo_coeff[1])
  
  mp = UMP2(mf)
  mp.run()
  dm_mp2 = mp.make_rdm()
  np.save("mp2_rdmA.npy", dm_mp2[0])
  np.save("mp2_rdmB.npy", dm_mp2[1])
  #print la.eigh(dm_mp2[0])[0]
  #print la.eigh(dm_mp2[1])[0]
