import numpy as np
import numpy.linalg as la
import itertools as it
from utils import mdot, ToSpinOrb, ToSpatOrb, ToClass
from embedding import GetFragRdm
from BCS_transform import transform_trans_inv_rdm

__all__ = ["FitVcorEmb", "FitVcorLatSCF"]

# functions related to fitting vloc

def Make_Vloc_Matrix_UHFB(v, nImp):
  # this function transforms the reduced representation
  # of vloc (as a row vector) into full representation
  # (Vloc(SpinOrb), Delta)
  Vloc = [np.zeros((nImp, nImp)), np.zeros((nImp, nImp))]
  Delta = np.zeros((nImp, nImp))
  nVloc = nImp * (nImp+1) / 2

  for idx, pos in enumerate(it.combinations_with_replacement(range(nImp), 2)):
    Vloc[0][pos] += v[idx]
    Vloc[0][pos[::-1]] += v[idx]
    Vloc[1][pos] += v[idx + nVloc]
    Vloc[1][pos[::-1]] += v[idx + nVloc]
  
  for idx, pos in enumerate(it.product(range(nImp), repeat = 2)):
    Delta[pos] += v[idx + 2*nVloc]

  return [ToSpinOrb(Vloc), Delta]

def Make_Vloc_Matrix_RHFB(v, nImp):
  # this function transforms the reduced representation
  # of vloc (as a row vector) into full representation
  # (Vloc, Delta)  used for restricted case
  Vloc = np.zeros((nImp, nImp))
  Delta = np.zeros((nImp, nImp))
  nVloc = nImp * (nImp+1) / 2

  for idx, pos in enumerate(it.combinations_with_replacement(range(nImp), 2)):
    Vloc[pos] += v[idx]
    Vloc[pos[::-1]] += v[idx]
    Delta[pos] += v[idx + nVloc]
    Delta[pos[::-1]] += v[idx + nVloc]

  return [Vloc, Delta]

def MakeTensorA(n, nImp, basis, UHFB):
  # reduced Vloc has a linear relation with embedding system BdG "Fock" Matrix
  # Tensor A transforms reduced Vloc into emb BdG
  A = np.zeros((n, nImp * 4, nImp * 4))
  u, v = basis.u, basis.v
  
  def TransformVloc(h, u, v):
    # h is a block diagonal matrix within duplicated
    # blocks along diagonal
    # we compute u.T.conj() * h * v
    assert(h.shape[0] == h.shape[1] == nImp)
    assert(u.shape[1] == v.shape[1] == 2*nImp)
    assert(u.shape[0] == v.shape[0] and u.shape[0] % nImp == 0)
    ncells = u.shape[0] / nImp

    h1 = np.zeros((nImp*2, nImp*2))
    for i in range(ncells):
      h1 += mdot(u[i*nImp:(i+1)*nImp].T.conj(), h, v[i*nImp:(i+1)*nImp])
    return h1
  
  if UHFB:
    func_MakeV = Make_Vloc_Matrix_UHFB
  else:
    func_MakeV = Make_Vloc_Matrix_RHFB

  for i in range(n):
    vloc = np.zeros(n)
    vloc[i] = 1.
    V, D = func_MakeV(vloc, nImp)

    if UHFB:
      V = ToSpatOrb(V)
      V_emb = [None] * 2
      V_emb[0] = TransformVloc(V[0], u[0], u[0]) - TransformVloc(V[1], v[1], v[1]) \
          + TransformVloc(D, u[0], v[1]) + TransformVloc(D.T.conj(), v[1], u[0])
      V_emb[1] = TransformVloc(V[1], u[1], u[1]) - TransformVloc(V[0], v[0], v[0]) \
          - TransformVloc(D.T, u[1], v[0]) - TransformVloc(D.conj(), v[0], u[1])
      D_emb = TransformVloc(D, u[0], u[1].conj()) + TransformVloc(D.T.conj(), v[1], v[0].conj()) \
          - TransformVloc(V[1], v[1], u[1].conj()) + TransformVloc(V[0], u[0], v[0].conj())
      A[i, :nImp*2, :nImp*2] = V_emb[0]
      A[i, nImp*2:, nImp*2:] = -V_emb[1]
    else:
      V_emb = TransformVloc(V, u, u) - TransformVloc(V, v, v) + TransformVloc(D, u, v) \
          + TransformVloc(D.T.conj(), v, u)
      D_emb = TransformVloc(D, u, u) - TransformVloc(D.T.conj(), v, v) - TransformVloc(V, u, v) \
          - TransformVloc(V, v, u)
      A[i, :nImp*2, :nImp*2] = V_emb
      A[i, :nImp*2, :nImp*2] = -V_emb
      
    A[i, :nImp*2, nImp*2:] = D_emb
    A[i, nImp*2:, :nImp*2] = D_emb.T.conj()
    
  return A

def rdm_err(vloc, EmbBdG, G0, TensorA):
  # given delta_vloc, build new emb bdg equation and solve for mean-field rdm
  # return the difference between mean-field rdm and FCI rdm
  nsites = EmbBdG.shape[0]/2
  H = EmbBdG + np.einsum('i,ijk->jk', vloc, TensorA)
  ew, ev = la.eigh(H)
  cocc = ev[:, nsites:]
  G = np.dot(cocc, cocc.T.conj())
  return G-G0

def Minimize(fn, x0, op, verbose):
  # Minimization function
  from scipy.optimize import fmin
  from scipy.linalg import lstsq

  nx = x0.shape[0]

  def Gradient(x):
    g = np.zeros_like(x)
    step = 1e-7
    for ix in range(nx):
      dx = np.zeros_like(x)
      dx[ix] = step
      g[ix] = (0.5/step) * (fn(x+dx) - fn(x-dx))
    return g

  x = x0
  if verbose > 4:
    print
    print "  Iter           Value               Grad                 Step"
    print "---------------------------------------------------------------------"

  r = fn(x)
  for iter in range(op.MaxIter):
    if (r < 1e-7 and iter != 0): # fn(x) supposed to be larger than 0
      break
    
    g = Gradient(x)  # use real gradient every time

    def GetDir():
      g2 = np.zeros((1+nx, nx))
      g2[0] = g
      g2[1:] = 0.1 * r * np.eye(nx)
      r2 = np.zeros(1+nx)
      r2[0] = r
      dx2, fitresid, rank, sigma = lstsq(g2, r2)
      dx = dx2[:nx]
      return dx

    dx = GetDir()

    def LineSearchFn(step):
      r1 = fn(x - step*dx)
      return r1
    
    def FindStep():
      grid = list(np.arange(0., 2.001, 0.2))
      val = [LineSearchFn(step) for step in grid]
      s = grid[np.argmin(val)]
      if (abs(s) > 1e-4):
        return s
      else:
        return fmin(LineSearchFn, np.array([0.001]), disp=0, xtol=1e-10)
    
    step = FindStep()
    dx *= step
    r_new = fn(x-dx)
    if r_new > r * 1.5:
      break
    x -= dx
    r = r_new
    if verbose > 4:
      print "%4d %20.12f %20.12f %20.12f" % (iter, r, la.norm(g), la.norm(dx))

  return x, r

def FitVcorEmb(nImp, HlResult, basis, EmbBdG, inp_fit, UHFB, verbose):
  if UHFB:
    nV = nImp * (nImp+1)
    nD = nImp * nImp
    func_MakeV = Make_Vloc_Matrix_UHFB
  else:
    nV = nImp * (nImp+1)/2
    nD = nImp * (nImp+1)/2
    func_MakeV = Make_Vloc_Matrix_RHFB

  TA = MakeTensorA(nV + nD, nImp, basis, UHFB)

  # Target Generalized Density Matrix
  G = np.zeros((nImp*4, nImp*4))
  if UHFB:
    G[:nImp*2, :nImp*2] = np.eye(nImp*2) - HlResult.rho_emb[::2,::2]
    G[nImp*2:, nImp*2:] = HlResult.rho_emb[1::2,1::2]
  else:
    G[:nImp*2, :nImp*2] = np.eye(nImp*2) - HlResult.rho_emb
    G[nImp*2:, nImp*2:] = HlResult.rho_emb
  G[:nImp*2, nImp*2:] = HlResult.kappa_emb
  G[nImp*2:, :nImp*2] = HlResult.kappa_emb.T.conj()
  
  options = ToClass({"MaxIter": inp_fit.MaxIter})

  if verbose > 4:
    print "Minimize difference in in embedding one-body density matrix"

  def comp(v):
    v1 = np.zeros(nV+nD)
    v1[:nV] = v
    return v1

  def get_frag_occ(dG):
    rho_emb = ToSpinOrb(dG[:nImp*2,:nImp*2], dG[nImp*2:, nImp*2:])
    kapp_emb = dG[:nImp*2, nImp*2:]
    return np.diag(GetFragRdm(rho_emb, kapp_emb, basis, nImp, True)[0])
      
  def err_func(x, BdG, G0, TensorA):
    dG = rdm_err(x, BdG, G0, TensorA)
    return la.norm(dG)# + 0*la.norm(get_frag_occ(dG))

  if not inp_fit.UpdateDelta:
    # then only fit Vloc
    dvloc = np.zeros(nV)
    dvloc, err_min = Minimize(lambda x: err_func(comp(x), EmbBdG, G, TA), dvloc, options, verbose)
    return func_MakeV(comp(dvloc), nImp), err_min
  else:
    dvloc = np.zeros(nV + nD)
    # minimize error
    dvloc, err_min = Minimize(lambda x: err_func(x, EmbBdG, G, TA), dvloc, options, verbose)
    return func_MakeV(dvloc, nImp), err_min

def FitVcorLatSCF(lattice, HlResult, basis, MfdSolver, Vcor, mu, inp_fit, UHFB, verbose):
  nImp = lattice.supercell.nsites
  if UHFB:
    nV = nImp * (nImp+1)
    nD = nImp * nImp
    func_MakeV = Make_Vloc_Matrix_UHFB
  else:
    nV = nImp * (nImp+1)/2
    nD = nImp * (nImp+1)/2
    func_MakeV = Make_Vloc_Matrix_RHFB
  
  # Target Generalized Density Matrix
  G0 = np.zeros((nImp*4, nImp*4))
  if UHFB:
    G0[:nImp*2, :nImp*2] = np.eye(nImp*2) - HlResult.rho_emb[::2,::2]
    G0[nImp*2:, nImp*2:] = HlResult.rho_emb[1::2,1::2]
  else:
    G0[:nImp*2, :nImp*2] = np.eye(nImp*2) - HlResult.rho_emb
    G0[nImp*2:, nImp*2:] = HlResult.rho_emb
  G0[:nImp*2, nImp*2:] = HlResult.kappa_emb
  G0[nImp*2:, :nImp*2] = HlResult.kappa_emb.T.conj()

  options = ToClass({"MaxIter": inp_fit.MaxIter})

  if verbose > 4:
    print "Minimize difference in in embedding one-body density matrix"

  def comp(v):
    v1 = np.zeros(nV+nD)
    v1[:nV] = v
    return v1

  def err_func(dx, G0):
    dVcor = func_MakeV(dx, nImp)
    MfdSolver.run([Vcor[i]+dVcor[i] for i in range(2)], mu, 0)
    rho, kappa = MfdSolver.rho, MfdSolver.kappa
    G = np.zeros((nImp*4, nImp*4))
    if UHFB:
      rho_emb, kappa_emb = transform_trans_inv_rdm(basis, lattice, rho[0], kappa, rho[1])
      G[:nImp*2, :nImp*2] = np.eye(nImp*2) - rho_emb[::2,::2]
      G[nImp*2:, nImp*2:] = rho_emb[1::2,1::2]
    else:
      rho_emb, kappa_emb = transform_trans_inv_rdm(basis, lattice, rho, kappa)
      G[:nImp*2, :nImp*2] = np.eye(nImp*2) - rho_emb
      G[nImp*2:, nImp*2:] = rho_emb
    G[:nImp*2, nImp*2:] = kappa_emb
    G[nImp*2:, :nImp*2] = kappa_emb.T.conj()
    return la.norm(G-G0)

  if not inp_fit.UpdateDelta:
    # then only fit Vloc
    dvloc = np.zeros(nV)
    dvloc, err_min = Minimize(lambda x: err_func(comp(x), G0), dvloc, options, verbose)
    return func_MakeV(comp(dvloc), nImp), err_min
  else:
    dvloc = np.zeros(nV + nD)
    # minimize error
    dvloc, err_min = Minimize(lambda x: err_func(x, G0), dvloc, options, verbose)
    return func_MakeV(dvloc, nImp), err_min

