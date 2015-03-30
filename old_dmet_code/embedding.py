import numpy as np
import numpy.linalg as la

#import mps_iface, block_iface
import block_iface
from settings import TmpDir, StoreDir
from utils import mdot, ToClass, ToSpatOrb
from BCS_transform import transform_Integral_DumpStyle

import os
from math import sqrt
from commands import getoutput
from shutil import rmtree
from tempfile import mkdtemp
from copy import deepcopy
import sys

def GetFragRdm(rho_emb, kappa_emb, basis, size, UHFB):
  u, v = deepcopy(basis.u), deepcopy(basis.v)
  rdm0, kappa0 = deepcopy(rho_emb), kappa_emb
  if UHFB:
    u = [u0[:size] for u0 in u]
    v = [v0[:size] for v0 in v]
    rdm0 = ToSpatOrb(rdm0)
    rdm = np.zeros((size*2, size*2))
    rdm[::2, ::2] = mdot(u[0].conj(), rdm0[0], u[0].T) - mdot(v[0], rdm0[1].T, v[0].T.conj()) + \
        mdot(v[0], v[0].T.conj()) - mdot(v[0], kappa0.T, u[0].T) - mdot(u[0].conj(), kappa0.conj(), v[0].T.conj())
    rdm[1::2, 1::2] = mdot(u[1].conj(), rdm0[1], u[1].T) - mdot(v[1], rdm0[0].T, v[1].T.conj()) + \
        mdot(v[1], v[1].T.conj()) + mdot(v[1], kappa0, u[1].T) + mdot(u[1].conj(), kappa0.T.conj(), v[1].T.conj())
    kappa = mdot(u[0], kappa0, u[1].T) + mdot(v[0].conj(), kappa0.T.conj(), v[1].T.conj()) + \
        mdot(u[0], v[1].T.conj()) - mdot(u[0], rdm0[0].T, v[1].T.conj()) + mdot(v[0].conj(), rdm0[1], u[1].T)
  else:
    u = u[:size]
    v = v[:size]
    rdm = mdot(u.conj(), rdm0, u.T) - mdot(v, rdm0.T, v.T.conj()) + mdot(v, v.T.conj()) + mdot(v, kappa0.T, u.T) + \
        mdot(u.conj(), kappa0.conj(), v.T.conj())
    kappa = mdot(u, kappa0, u.T) - mdot(v.conj(), kappa0.T.conj(), v.T.conj()) + mdot(u, v.T.conj()) - \
        mdot(u, rdm0.T, v.T.conj()) - mdot(v.conj(), rdm0, u.T)
  
  return rdm, kappa

def EmbResults_BLOCK(basis, emb_result, E0, E_raw, docc, nsites, UHFB, verbose):
  rho_frag, kappa_frag = GetFragRdm(emb_result.rho, emb_result.kappa , basis, nsites, UHFB)
  E = E0 + E_raw
  
  if UHFB:
    n = np.trace(rho_frag) / 2.
    norm = la.norm(emb_result.rho) / sqrt(2).real
  else:
    n = np.trace(rho_frag)
    norm = la.norm(emb_result.rho)
  
  if verbose > 1:
    print "n(interacting) = %20.12f" % n
    print "E(interacting) = %20.12f" % E
    print "dRho (norm)    = %20.12f" % norm
    print "dKappa (norm)  = %20.12f" % la.norm(emb_result.kappa)
    if docc is not None:
      print "double occupancies: %s" % docc
    print
    sys.stdout.flush()

  return ToClass({"n": n, "E": E, "rho_emb": emb_result.rho, 
      "kappa_emb": emb_result.kappa, "rho_frag": rho_frag, "kappa_frag": kappa_frag, "docc": docc})

class EmbSystemBLOCK(object):
  def __init__(self, options):
    self.TmpDir = options.TmpDir
    self.StoreDir = options.StoreDir
    self.RestartDir = options.RestartDir
    getoutput("mpirun -npernode 1 mkdir -p %s" % TmpDir)

    self.npar = options.N_QP
    self.M = options.M
    self.restart = options.Restart
    self.UHFB = options.UHFB
    self.CoreHam = options.CoreHam
    self.IntHam = options.IntHam

    self.compute_docc = options.DoubleOcc 
    
    self.CWD = None
    self.nproc = options.nproc
    self.node = options.node
  
  def run(self, basis, Vcor, mu, mu0, verbose = 0, fitting = False):
    if fitting:
      verbose -= 1
    if verbose > 2:
      print "\n******** Embedding System High Level Calculation *********\n"
    
    Delta = Vcor[1]
    nscsites = Delta.shape[0]
    nsorbs = nscsites*2 if self.UHFB else nscsites
    Vloc = Vcor[0] + np.eye(nsorbs)*(mu-mu0)        

    HamCoreEmb, HamCoreFrag = self.CoreHam(basis, (Vloc, Delta), mu)
    Int, Int_frag = self.IntHam(basis)
    Int1e, Int2e = map(deepcopy, Int)
    V0_frag, Int1e_frag, Int2e_frag = map(deepcopy, Int_frag)
    Int1e.cd += HamCoreEmb.H
    Int1e.cc += HamCoreEmb.D
    
    V0_frag += HamCoreFrag.E0
    Int1e_frag.cd += HamCoreFrag.H
    Int1e_frag.cc += HamCoreFrag.D

    restart = self.handle_restart()
    
    if verbose > 1:
      print "Run on Machine:"
      print getoutput("mpirun -npernode 1 hostname | sort")
      print "Temporary File Location:"
      print self.CWD
      if restart:
        print "Restart from previous calculation"
      print
    sys.stdout.flush()
    
    npar = self.npar
    if npar is None:
      npar = nscsites * 2
    
    M = self.M
    if isinstance(M, int):
      if restart:
        M = (M, M)
      else:
        M = (250, M)

    options = {"restart": restart, "M": M, "npar": npar, "UHFB": self.UHFB, "nproc": self.nproc, "node": self.node}
    
    reload(block_iface)
    EmbRawResult = block_iface.run_emb(Int1e, Int2e, self.CWD, options, verbose)
    Energy = block_iface.run_energy(Int1e_frag, Int2e_frag, self.CWD, options)
    
    docc = None
    if self.compute_docc and not fitting:
      maskInt2e = [np.array([0, 0, 0], dtype = int)]
      sInt2e = [np.zeros((nscsites, nscsites, nscsites, nscsites))]      
      irange = [0]
      docc = np.zeros((nscsites))
      for i in range(nscsites):
        sInt2e[0] *= 0
        sInt2e[0][i,i,i,i] = 1.
        _, dInt = transform_Integral_DumpStyle(nscsites, basis, sInt2e, maskInt2e, irange, False, self.UHFB, \
            lambda x, y: x+y)
        docc[i] = block_iface.run_energy(dInt[1], dInt[2], self.CWD, options) + dInt[0]
    
    return EmbResults_BLOCK(basis, EmbRawResult, V0_frag, Energy, docc, nscsites, self.UHFB, verbose)

  def setM(self, M):
    self.M = M

  def CleanUp(self):
    if self.CWD is not None:
      try:
        rmtree(self.CWD)
      except OSError:
        pass
    self.CWD = None
  
  def prepare_restart_info(self):
    if self.StoreDir is None:
      return None
    else:
      store_dir = mkdtemp(prefix = "BLOCK_RESTART", dir = self.StoreDir)
      getoutput("cp %s/RestartReorder.dat %s" % (self.CWD, store_dir))
      getoutput("cp %s/Rotation* %s" % (self.CWD, store_dir))
      getoutput("cp %s/StateInfo* %s" % (self.CWD, store_dir))
      getoutput("cp %s/statefile* %s" % (self.CWD, store_dir))
      getoutput("cp %s/wave* %s" % (self.CWD, store_dir))
      getoutput("mpirun -npernode 1 rm -rf %s" % self.CWD)
      self.CWD = None
      return store_dir

  def handle_restart(self):
    restart = self.restart
    if restart:
      if self.RestartDir is not None:
        self.CWD = mkdtemp(prefix = "BLOCK", dir = self.TmpDir)
        getoutput("cp %s/RestartReorder.dat %s" % (self.RestartDir, self.CWD))
        getoutput("cp %s/Rotation* %s" % (self.RestartDir, self.CWD))
        getoutput("cp %s/StateInfo* %s" % (self.RestartDir, self.CWD))
        getoutput("cp %s/statefile* %s" % (self.RestartDir, self.CWD))
        getoutput("cp %s/wave* %s" % (self.RestartDir, self.CWD))
        #rmtree(self.RestartDir)
        self.RestartDir = None
      elif self.CWD is None:
        restart = False
        self.CWD = mkdtemp(prefix = "BLOCK", dir = self.TmpDir)
    else:
      if self.CWD is not None:
        getoutput("mpirun -npernode 1 rm -rf %s" % self.CWD)
      self.CWD = mkdtemp(prefix = "BLOCK", dir = self.TmpDir)
    getoutput("mpirun -npernode 1 mkdir -p %s" % self.CWD) # create tmp dir on each node
    return restart

#def EmbResults_MPS(basis, emb_result, E0, ham_emb, ham_frag, nsites, UHFB, verbose):
#  rho_frag, kappa_frag = GetFragRdm(emb_result.rho, emb_result.kappa , basis, nsites, UHFB)
#  E0 += emb_result.E
#  
#  if UHFB:
#    n = np.trace(rho_frag) / 2.
#    E = E0 - np.sum((ham_emb.H - ham_frag.H) * emb_result.rho) + 2 * np.sum((ham_emb.D - ham_frag.D) * emb_result.kappa) # FIXME
#    norm = la.norm(emb_result.rho) / sqrt(2).real
#  else:
#    n = np.trace(rho_frag)
#    E = E0 - 2*np.sum((ham_emb.H-ham_frag.H) * emb_result.rho) + 2*np.sum((ham_emb.D-ham_frag.D) * emb_result.kappa)
#    norm = la.norm(emb_result.rho)
#
#  if verbose > 1:
#    print "n(interacting) = %20.12f" % n
#    print "E(interacting) = %20.12f" % E
#    print "dRho (norm)    = %20.12f" % norm
#    print "dKappa (norm)  = %20.12f" % la.norm(emb_result.kappa)
#    print
#    sys.stdout.flush()
#
#  return ToClass({"n": n, "E": E, "rho_emb": emb_result.rho, 
#      "kappa_emb": emb_result.kappa, "rho_frag": rho_frag, "kappa_frag": kappa_frag, "docc": None})
#
#
#class EmbSystemMPS(object):
#  def __init__(self, options):
#    if options.DoubleOcc:
#      raise Exception("Cannot compute double occupancy with MPS DMRG solver")
#    self.TmpDir = options.TmpDir
#    getoutput("mkdir -p %s" % TmpDir)
#
#    self.M = options.M # M is a tuple (StartM, maxM)
#    self.UHFB = options.UHFB
#    self.CoreHam = options.CoreHam
#    self.IntHam = options.IntHam
#    self.CWD = None
#    self.nproc = options.nproc
#  
#  def run(self, basis, Vcor, mu, mu0, verbose = 0):
#    if verbose > 2:
#      print "\n******** Embedding System High Level Calculation *********\n"
#    
#    Delta = Vcor[1]
#    norbs = Delta.shape[0]
#    nsorbs = norbs*2 if self.UHFB else norbs
#    Vloc = Vcor[0] + np.eye(nsorbs)*(mu-mu0)
#    
#    HamCoreEmb, HamCoreFrag = self.CoreHam(basis, (Vloc, Delta), mu)
#    V_Int = self.IntHam(basis)
#    HembInt = deepcopy(HamCoreEmb.H)
#    DembInt = deepcopy(HamCoreEmb.D)
#    E0 = HamCoreEmb.E0
#    if self.UHFB:
#      for i in range(norbs):
#        E0 += V_Int.U * V_Int.eps[0][i] * V_Int.eps[1][i]
#        HembInt[::2,::2] += V_Int.U * (V_Int.eps[0][i] * V_Int.g[1][i] + V_Int.eps[1][i] * V_Int.f[0][i])
#        HembInt[1::2,1::2] += V_Int.U * (V_Int.eps[0][i] * V_Int.f[1][i] + V_Int.eps[1][i] * V_Int.g[0][i])
#        DembInt += V_Int.U * (-V_Int.eps[0][i] * V_Int.d[1][i].T + V_Int.eps[1][i] * V_Int.d[0][i])
#    else:
#      for i in range(norbs):
#        E0 += V_Int.U * V_Int.eps[i] ** 2
#        HembInt += V_Int.U * V_Int.eps[i] * (V_Int.f[i]+V_Int.g[i])
#        DembInt += V_Int.U * V_Int.eps[i] * (V_Int.d[i]+V_Int.d[i].T)
#
#    self.CWD = mkdtemp(prefix = "MPS", dir = self.TmpDir)
#    if verbose > 1:
#      print "Temporary File Location"
#      print "%s:%s" % (getoutput("hostname"), self.CWD)
#    sys.stdout.flush()
#
#    options = {"M": self.M, "UHFB": self.UHFB, "nproc": self.nproc}
#
#    EmbRawResult = mps_iface.run_emb(HembInt, DembInt, V_Int.U, V_Int.f, V_Int.g, V_Int.d, self.CWD, options, verbose)
#    rmtree(self.CWD)
#    self.CWD = None
#    return EmbResults_MPS(basis, EmbRawResult, E0, HamCoreEmb, HamCoreFrag, norbs, self.UHFB, verbose)
#
#  def setM(self, M):
#    self.M = M
#
#  def CleanUp(self):
#    if self.CWD is not None:
#      try:
#        rmtree(self.CWD)
#      except OSError:
#        pass
#    self.CWD = None

def EmbSolver(inp_solver, OrbType, FCore, FInt):
  options = deepcopy(inp_solver)
  options.UHFB = (OrbType == "UHFB")
  options.CoreHam = FCore
  #if inp_solver.ImpSolver == "MPS_DMRG":
  #  options.IntHam = lambda basis: FInt(basis, "MPS")
  #  return EmbSystemMPS(options)
  if inp_solver.ImpSolver == "BLOCK_DMRG":
    options.IntHam = lambda basis: FInt(basis, "DUMP")
    return EmbSystemBLOCK(options)
  else:
    raise Exception("ImpSolver not available")

def IntType(method):
  if method == "MPS_DMRG":
    return "MPS"
  elif method == "BLOCK_DMRG":
    return "DUMP"
