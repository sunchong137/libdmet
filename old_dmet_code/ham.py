#
# File: geometry.py
# Author: Bo-Xiao Zheng <boxiao@princeton.edu>
#

import numpy as np
import numpy.linalg as la
import itertools as it
import re

def get_current_path():
  import inspect, os
  return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def read_clusterdump(filename, nscsites, ncells):
  with open(filename, "r") as f:
    lines = f.readlines()
  assert(nscsites == int(re.split("[=,]", lines[0])[1]))
  idx = [int(x) for x in re.split(",", lines[2])[:nscsites]]
  
  for i, j in enumerate(np.sort(idx)):
    assert(i+1==j)

  Int2e = np.zeros((nscsites, nscsites, nscsites, nscsites))
  nsites = nscsites*ncells
  overlap = np.zeros((nsites, nsites))

  for line in lines[4:]:
    tokens = line.split()
    real, imag = float(tokens[0]), float(tokens[1])
    i,j,k,l = [int(x)-1 for x in tokens[2:]]
    assert(abs(imag) < 1e-9)
    if l >= 0:
      Int2e[i,j,k,l] = real
    else:
      overlap[i,j] = real
  assert(np.allclose(Int2e - np.swapaxes(Int2e, 0, 1), 0))
  assert(np.allclose(Int2e - np.swapaxes(Int2e, 2, 3), 0))
  assert(np.allclose(Int2e - np.swapaxes(np.swapaxes(Int2e, 0, 2), 1, 3), 0))
  
  return Int2e, overlap, idx

def read_JKdump(filename, nscsites, ncells):
  with open(filename, "r") as f:
    lines = f.readlines()
  nsites = nscsites*ncells
  assert(nsites == int(re.split("[=,]", lines[0])[1]))
  nelec = int(re.split("[=,]", lines[0])[3])
  J = np.zeros((nsites, nsites))

  for line in lines[2:]:
    tokens = line.split()
    real, imag = float(tokens[0]), float(tokens[1])
    i,j = [int(x) for x in tokens[2:4]]
    assert(abs(imag) < 1e-10)
    J[i-1, j-1] = real

  assert(np.allclose(J-J.T, 0))
  
  return J, nelec

def read_FOCKdump(filename, nscsites, ncells):
  with open(filename, "r") as f:
    lines = f.readlines()
  nsites = nscsites*ncells
  assert(nsites == int(re.split("[=,]", lines[0])[1]))
  nelec = int(re.split("[=,]", lines[0])[3])

  ews = np.zeros(nsites)

  for line in lines[2:]:
    tokens = line.split()
    real, imag = float(tokens[0]), float(tokens[1])
    i,j = [int(x) for x in tokens[2:4]]
    assert(abs(imag) < 1e-10)
    if j == 0:
      ews[i-1] = real

  return ews, nelec

def read_FCIdump(filename, nscsites, ncells, Int2eShape = "SC"):
  with open(filename, "r") as f:
    lines = f.readlines()
  nsites = nscsites*ncells
  # 1e index 1 to nsites, 2e index 1 to nscsites
  assert(nscsites == int(re.split("[=,]", lines[0])[1]))
  
  if Int2eShape == "SC":
    Int2e = np.zeros((nscsites, nscsites, nscsites, nscsites))
  else:
    Int2e = np.zeros((nscsites, nsites, nsites, nsites))
  H1e = np.zeros((nsites, nsites))
  for line in lines[4:]:
    tokens = line.split()
    val = float(tokens[0])
    i,j,k,l = [int(x)-1 for x in tokens[1:]]
    if k < 0 and l < 0:
      H1e[k,l] = H1e[l,k] = val
    elif Int2eShape == "SC":
      Int2e[i,j,k,l] = Int2e[j,i,k,l] = Int2e[i,j,l,k] = Int2e[j,i,l,k] = \
          Int2e[k,l,i,j] = Int2e[k,l,j,i] = Int2e[l,k,i,j] = Int2e[l,k,j,i] = val
    else:
      assert(i < nscsites)
      Int2e[i,j,k,l] = Int2e[i,j,l,k] = val # FIXME we did not check the symmetry
  Int2e = Int2e
  return H1e, Int2e


class FHam(object):
  def __init__(self):
    raise Exception("FHam::__init__ must be implemented in derived class")
  
  def get_h0(self):
    return self.H1e
  
  def get_Int2e(self):
    return self.Int2e

  def get_fock(self):
    raise Exception("FHam::get_fock must be implemented in derived class")

  def get_imp_corr(self):
    raise Exception("FHam::get_eff_imp must be implemented in derived class")

  def get_U(self):
    raise Exception("Not Hubbard model")

class FHamHubbard(FHam):
  def __init__(self, Params, inp_ctrl, lattice):
    assert(inp_ctrl.Fock != "Store")
    
    # assign parameters
    self.t = Params["t"]
    self.t1 = Params["t1"]
    self.U = Params["U"]
    
    if inp_ctrl.Int2eShape == "SC":
      assert(abs(Params["V"]) < 1e-7 and abs(Params["V1"]) < 1e-7 or \
        abs(Params["J"]) < 1e-7 or abs(Params["J1"]) < 1e-7)
    else:
      self.V = Params["V"]
      self.V1 = Params["V1"]
      self.J = Params["J"]
      self.J1 = Params["J1"]
    
    self.Int2eShape = inp_ctrl.Int2eShape
    if self.Int2eShape == "SC":
      self.type = "Hubbard"
    else:
      self.type = "ExtendHubbard"

    # build h0
    if lattice.bc in [-1, 0, 1]:
      nsc = lattice.nscells
      nscsites = lattice.supercell.nsites
      self.H1e = np.zeros((nsc, nscsites, nscsites))
      if abs(self.t) > 1e-7:
        pairs = lattice.get_NearNeighbor(sites = range(nscsites))
        for nn in pairs[0]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t
        for nn in pairs[1]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t * lattice.bc
      if abs(self.t1) > 1e-7:
        pairs = lattice.get_2ndNearNeighbor(sites = range(nscsites))
        for nn in pairs[0]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t1
        for nn in pairs[1]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t1 * lattice.bc
    else:
      raise Exception("Unsupported boundary condition")
    
    # build int2e matrix
    if self.Int2eShape == "SC":
      # which means there's only U
      nsites = lattice.supercell.nsites
      self.Int2e = np.zeros((nsites, nsites, nsites, nsites))
      for i in range(nsites):
        self.Int2e[i,i,i,i] = self.U
    else:
      assert(lattice.bc == 1)
      self.Int2e = np.zeros((nscsites, nscsites*nsc, nscsites*nsc, nscsites*nsc))
      if abs(self.U) > 1e-7:
        for n in range(nscsites):
          self.Int2e[n, n, n, n] = self.U
      if abs(self.V) > 1e-7 or abs(self.J) > 1e-7:
        pairs = lattice.get_NearNeighbor(sites = range(nscsites))
        for nn in pairs[0]+pairs[1]:
          self.Int2e[nn[0], nn[0], nn[1], nn[1]] = self.V
          self.Int2e[nn[0], nn[1], nn[0], nn[1]] = self.J
      if abs(self.V1) > 1e-7 or abs(self.J1) > 1e-7:
        pairs = lattice.get_2ndNearNeighbor(sites = range(nscsites))
        for nn in pairs[0]+pairs[1]:
          self.Int2e[nn[0], nn[0], nn[1], nn[1]] = self.V1
          self.Int2e[nn[0], nn[1], nn[0], nn[1]] = self.J1

  def get_U(self):
    if self.Int2eShape == "SC":
      return self.U
    else:
      raise Exception("Not Hubbard model")


def reduce_to_trans_inv(Mat, lattice):
  nscsites = lattice.supercell.nsites
  ncells = np.product(lattice.scsize)
  assert(Mat.shape[0] == Mat.shape[1])
  assert(Mat.shape[0] == nscsites * ncells)
  assert(lattice.bc == 1)
  
  assert(np.allclose(Mat - Mat.T, 0., atol = 1e-7))
  Mat = 0.5 * (Mat + Mat.T)

  Mat1 = np.zeros((ncells, nscsites, nscsites))
  for i in range(ncells):
    ref = Mat[:nscsites, i*nscsites:(i+1)*nscsites]
    for j in range(ncells):
      k = lattice.add(i, j)
      assert(la.norm(Mat[j*nscsites:(j+1)*nscsites, k*nscsites:(k+1)*nscsites] - ref, np.inf) < 1e-3)
      Mat1[i] += Mat[j*nscsites:(j+1)*nscsites, k*nscsites:(k+1)*nscsites]

  Mat1 /= ncells
  return Mat1

class FHamNonInt(FHam):
  def __init__(self, inp_ham, inp_ctrl, lattice):

    int_path = inp_ham.Params["Integral"]
    if not int_path.startswith("/"): # relative path
      int_path = get_current_path() + "/" + int_path

    # process parameters
    assert(inp_ctrl.Int2eShape == "SC")
    self.Int2eShape = "SC"

    assert(inp_ctrl.Fock == "Store" or inp_ctrl.Fock == "None")
    if inp_ctrl.BathInt2e and not inp_ham.Local:
      raise Exception("Cannot deal with non-local Hamiltonian with non-interacting scheme and interacting bath")

    # get sizes
    self.nscsites = lattice.supercell.nsites
    self.cells = lattice.scsize
    self.type = "Non Interacting"
    
    if inp_ctrl.Fock == "Store":
      # read data      
      self.Int2e, overlap, cluster_idx = read_clusterdump(int_path + \
          "/FCIDUMP.CLUST.GTO", self.nscsites, np.product(self.cells))
      J, nelec = read_JKdump(int_path + "/JDUMP", self.nscsites, np.product(self.cells))
      K, _ = read_JKdump(int_path + "/KDUMP", self.nscsites, np.product(self.cells))
      ew, _ = read_FOCKdump(int_path + "/FOCKDUMP", self.nscsites, np.product(self.cells))
      
      J = np.dot(overlap, np.dot(J, overlap.T))
      K = np.dot(overlap, np.dot(K, overlap.T))
      Fock = np.dot(overlap, np.dot(np.diag(ew), overlap.T))

      # build translational invariant operators
      self.Fock, self.J, self.K = map(lambda M: reduce_to_trans_inv(M, lattice), [Fock, J, K])

      nocc = nelec/2

      homo, lumo = ew[nocc-1], ew[nocc]
      mo_occ = np.zeros(self.nscsites * np.product(self.cells))
      if lumo-homo < 1e-6:
        occ = ew < lumo+1e-6
        virt = ew > homo-1e-6
        core_orbs = np.sum(occ * (1-virt))
        frac_orbs = np.sum(occ * virt)
        mo_occ[:core_orbs] = 1.
        mo_occ[core_orbs:core_orbs+frac_orbs] = 1.*(nocc-core_orbs) / frac_orbs
      else:
        mo_occ[:nocc] = 1.

      rho_imp = np.dot(overlap[:self.nscsites], np.dot(np.diag(mo_occ), overlap[:self.nscsites].T))
      self.Jimp = np.einsum("ijkl,kl->ij", self.Int2e, rho_imp)
      self.Kimp = np.einsum("ikjl,kl->ij", self.Int2e, rho_imp)
      self.H1e = self.Fock - self.J*2 + self.K
    
    else:
      H1e, self.Int2e = read_fcidump(int_path + "/FCIDUMP", \
          nscsites, np.product(self.cells))
      self.H1e = reduce_to_trans_inv(H1e, lattice)
      self.Fock = self.J = self.K = self.Jimp = self.Kimp = None

  def get_fock(self):
    return self.Fock

  def get_imp_corr(self):
    return 2*self.Jimp-self.Kimp # should dress with spin


class FHamInt(FHam):
  def __init__(self, inp_ham, inp_ctrl, lattice):
    assert(inp_ctrl.Fock == "Comp")
    self.Int2eShape = inp_ctrl.Int2eShape

    # get sizes
    self.nscsites = lattice.supercell.nsites
    self.cells = lattice.scsize

    int_path = inp_ham.Params["Integral"]
    if not int_path.startswith("/"): # relative path
      int_path = get_current_path() + "/" + int_path

    # read data
    H1e, self.Int2e = read_fcidump(int_path + "/FCIDUMP.CLUST.GTO", self.nscsites, \
        np.product(self.cells), self.Int2eShape)
    
    self.H1e = reduce_to_trans_inv(H1e, lattice)
    self.type = "Interacting"


def Hamiltonian(inp_ham, inp_ctrl, lattice):
  if inp_ham.Type == "Hubbard":
    return FHamHubbard(inp_ham.Params, inp_ctrl, lattice)
  elif inp_ham.Type == "NonInt":
    return FHamNonInt(inp_ham, inp_ctrl, lattice)
  elif inp_ham.Type == "Int":
    return FHamInt(inp_ham, inp_ctrl, lattice)
  else:
    raise KeyError('HamiltonianType %s not exists' % inp_ham.Type)
