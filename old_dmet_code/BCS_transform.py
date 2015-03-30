import numpy as np
import numpy.linalg as la
from utils import mdot, ToClass
import itertools as it

__all__ = ["transform_trans_inv", "transform_local", "transform_impurity", \
    "transform_scalar", "transform_full_lattice", "transform_imp_env", \
    "LatticeIntegral", "transform_EmbIntHam_MPS", "transform_EmbIntHam_Dump", \
    "transform_Integral_DumpStyle", "transform_trans_inv_rdm"]

def transform1(U, H, V, lattice):
  """
  compute matrix product UHV, where H is in reduced form,
  U is mxn, H is nxn, V is nxm
  """
  assert(H.shape[1] == H.shape[2])
  assert(U.shape[1] == V.shape[0] == H.shape[0] * H.shape[1])
  nscsites = H.shape[1]
  ncells = H.shape[0]

  nonzero = []
  for i in range(ncells):
    if not np.allclose(H[i], 0):
      nonzero.append(i)

  M = np.zeros((U.shape[0], V.shape[1]))
  for i in range(ncells):
    r1 = lattice.sc_idx2pos(i)
    for j in nonzero:
      dr = lattice.sc_idx2pos(j)
      k = lattice.sc_pos2idx(r1+dr)
      M += mdot(U[:, i*nscsites:(i+1)*nscsites], H[j], V[k*nscsites:(k+1)*nscsites])
  return M

def transform2(U, H, V):
  """
  in this case, U is mxnscsites, which corresponds to only 
  the first block of core Hamiltonian (all other elements are zeros)
  """
  assert(H.shape[1] == H.shape[2])
  assert(V.shape[0] == H.shape[0] * H.shape[1])
  assert(U.shape[0] == V.shape[1])
  
  nscsites = H.shape[1]
  ncells = H.shape[0]
  temp = np.zeros((nscsites, V.shape[1]))
  
  for i in range(ncells):
    temp += np.dot(H[i], V[i*nscsites:(i+1)*nscsites])
  return np.dot(U[:, :nscsites], temp)

def transform3(U, H, V):
  """
  compute UHV, where H is translational invariant and block diagonal
  H = diag(H0,H0,H0,...)
  """
  assert(H.shape[0] == H.shape[1])
  assert(U.shape[1] == V.shape[0])
  assert(U.shape[1] % H.shape[0] == 0)
  nscsites = H.shape[0]
  ncells = U.shape[1] / H.shape[0]
  
  M = np.zeros((U.shape[0], V.shape[1]))
  for i in range(ncells):
    M += mdot(U[:, i*nscsites:(i+1)*nscsites], H, V[i*nscsites:(i+1)*nscsites])
  return M

def transform_trans_inv(basis, lattice, FA, D, FB = None):
  nscsites = lattice.supercell.nsites
  trans = lambda x,H,y: transform1(x.T.conj(), H, y, lattice)
  u, v = basis.u, basis.v

  if FB is not None: # unrestricted
    D_T = lattice.transpose_reduced(D)
    Hemb = np.zeros((nscsites*4, nscsites*4))
    Hemb[::2,::2] = trans(u[0], FA, u[0]) - trans(v[1], FB, v[1]) + \
        trans(u[0], D, v[1]) + trans(v[1], D_T.conj(), u[0])
    Hemb[1::2,1::2] = trans(u[1], FB, u[1]) - trans(v[0], FA, v[0]) - \
        trans(u[1], D_T, v[0]) - trans(v[0], D.conj(), u[1])
    Demb = trans(u[0], D, u[1].conj()) + trans(v[1], D_T.conj(), v[0].conj()) \
        - trans(v[1], FB, u[1].conj()) + trans(u[0], FA, v[0].conj())
  else:
    Hemb = trans(u, FA, u) - trans(v, FA, v) + trans(u, D, v) + trans(v, D.conj(), u)
    Demb = -trans(v, FA, u.conj()) - trans(u, FA, v.conj()) + \
        trans(u, D, u.conj()) - trans(v, D.conj(), v.conj())
  return Hemb, Demb

def transform_local(basis, lattice, VA, D, VB = None):
  # local and replicate over the lattice, a special case of translation invariant
  nscsites = lattice.supercell.nsites
  trans = lambda x,H,y: transform3(x.T.conj(), H, y)
  u, v = basis.u, basis.v
  
  if VB is not None: # unrestricted
    Hemb = np.zeros((nscsites*4, nscsites*4))
    Hemb[::2,::2] = trans(u[0], VA, u[0]) - trans(v[1], VB, v[1]) + \
        trans(u[0], D, v[1]) + trans(v[1], D.T.conj(), u[0])
    Hemb[1::2,1::2] = trans(u[1], VB, u[1]) - trans(v[0], VA, v[0]) - \
        trans(u[1], D.T, v[0]) - trans(v[0], D.conj(), u[1])
    Demb = trans(u[0], D, u[1].conj()) + trans(v[1], D.T.conj(), v[0].conj()) \
        - trans(v[1], VB, u[1].conj()) + trans(u[0], VA, v[0].conj())
  else:
    Hemb = trans(u, VA, u) - trans(v, VA, v) + trans(u, D, v) + trans(v, D.conj(), u)
    Demb = -trans(v, VA, u.conj()) - trans(u, VA, v.conj()) + \
        trans(u, D, u.conj()) - trans(v, D.conj(), v.conj())
  return Hemb, Demb

def transform_impurity(basis, lattice, VA, D, VB = None):
  # the matrix only lives on the impurity
  nscsites = lattice.supercell.nsites
  trans = lambda x,H,y: mdot(x[:nscsites].T.conj(), H, y[:nscsites])
  u, v = basis.u, basis.v

  if VB is not None: # unrestricted
    Hemb = np.zeros((nscsites*4, nscsites*4))
    Hemb[::2,::2] = trans(u[0], VA, u[0]) - trans(v[1], VB, v[1]) + \
        trans(u[0], D, v[1]) + trans(v[1], D.T.conj(), u[0])
    Hemb[1::2,1::2] = trans(u[1], VB, u[1]) - trans(v[0], VA, v[0]) - \
        trans(u[1], D.T, v[0]) - trans(v[0], D.conj(), u[1])
    Demb = trans(u[0], D, u[1].conj()) + trans(v[1], D.T.conj(), v[0].conj()) \
        - trans(v[1], VB, u[1].conj()) + trans(u[0], VA, v[0].conj())
  else:
    Hemb = trans(u, VA, u) - trans(v, VA, v) + trans(u, D, v) + trans(v, D.conj(), u)
    Demb = -trans(v, VA, u.conj()) - trans(u, VA, v.conj()) + \
        trans(u, D, u.conj()) - trans(v, D.conj(), v.conj())
  return Hemb, Demb

def transform_scalar(basis, lattice, s = 1., UHFB = False):
  nscsites = lattice.supercell.nsites  
  u, v = basis.u, basis.v
  if UHFB:
    Hemb = np.zeros((nscsites*4, nscsites*4))
    Hemb[::2, ::2] = np.dot(u[0].T.conj(), u[0]) - np.dot(v[1].T.conj(), v[1])
    Hemb[1::2,1::2] = np.dot(u[1].T.conj(), u[1]) - np.dot(v[0].T.conj(), v[0])
    Demb = np.dot(u[0].T.conj(), v[0].conj()) - np.dot(v[1].T.conj(), u[1].conj())
  else:
    Hemb = np.dot(u.T.conj(), u) - np.dot(v.T.conj(), v)
    Demb = - np.dot(v.T.conj(), u.conj()) - np.dot(u.T.conj(), v.conj())
  return Hemb*s, Demb*s

def transform_full_lattice(basis, lattice, VA, D, VB = None):
  nscsites = lattice.supercell.nsites  
  trans = lambda x,H,y: mdot(x.T.conj(), H, y)  
  u, v = basis.u, basis.v
  
  if VB is not None: # unrestricted
    Hemb = np.zeros((nscsites*4, nscsites*4))
    Hemb[::2,::2] = trans(u[0], VA, u[0]) - trans(v[1], VB, v[1]) + \
        trans(u[0], D, v[1]) + trans(v[1], D.T.conj(), u[0])
    Hemb[1::2,1::2] = trans(u[1], VB, u[1]) - trans(v[0], VA, v[0]) - \
        trans(u[1], D.T, v[0]) - trans(v[0], D.conj(), u[1])
    Demb = trans(u[0], D, u[1].conj()) + trans(v[1], D.T.conj(), v[0].conj()) \
        - trans(v[1], VB, u[1].conj()) + trans(u[0], VA, v[0].conj())
  else:
    Hemb = trans(u, VA, u) - trans(v, VA, v) + trans(u, D, v) + trans(v, D.conj(), u)
    Demb = -trans(v, VA, u.conj()) - trans(u, VA, v.conj()) + \
        trans(u, D, u.conj()) - trans(v, D.conj(), v.conj())
  return Hemb, Demb

def transform_imp_env(basis, lattice, H, UHFB):
  nscsites = lattice.supercell.nsites  
  trans = lambda x,H,y: transform2(x[:nscsites].T.conj(), H, y)
  u, v = basis.u, basis.v

  if UHFB:
    Hfrag = np.zeros((nscsites*4, nscsites*4))
    Hfrag[::2, ::2] = trans(u[0], H, u[0]) - trans(v[1], H, v[1])
    Hfrag[1::2, 1::2] = trans(u[1], H, u[1]) - trans(v[0], H, v[0])
    Dfrag = trans(u[0], H, v[0].conj()) - trans(v[1], H, u[1].conj())
    E0 = np.trace(trans(v[0].conj(), H, v[0].conj()) + trans(v[1].conj(), H, v[1].conj()))
  else: 
    Hfrag = trans(u, H, u) - trans(v, H, v)
    Dfrag = - (trans(v, H, u.conj()) + trans(u, H, v.conj()))
    E0 = np.trace(trans(v.conj(), H, v)) * 2
  return Hfrag, Dfrag, E0

##################################################################

def transform_trans_inv_rdm(basis, lattice, rhoA, kappa, rhoB = None):
  nscsites = lattice.supercell.nsites
  trans = lambda x,H,y: transform1(x.T, H, y.conj(), lattice)
  u, v = basis.u, basis.v

  if rhoB is not None: # unrestricted
    kappa_T = lattice.transpose_reduced(kappa)
    rho_emb = np.zeros((nscsites*4, nscsites*4))
    rho_emb[::2,::2] = trans(u[0], rhoA, u[0]) - trans(v[1], rhoB, v[1]) \
        - trans(u[0], kappa.conj(), v[1]) - trans(v[1], kappa_T, u[0]) + np.dot(v[1].T, v[1].conj())
    rho_emb[1::2,1::2] = trans(u[1], rhoB, u[1]) - trans(v[0], rhoA, v[0]) \
        + trans(u[1], kappa_T.conj(), v[0]) + trans(v[0], kappa, u[1]) + np.dot(v[0].T, v[0].conj())
    kappa_emb = trans(u[0].conj(), kappa, u[1]) + trans(v[1].conj(), kappa_T.conj(), v[0]) \
        - trans(u[0].conj(), rhoA, v[0]) + trans(v[1].conj(), rhoB, u[1]) + np.dot(u[0].T.conj(), v[0].conj())
  else:
    rho_emb = trans(u, rhoA, u) - trans(v, rhoA, v) - trans(u, kappa.conj(), v) \
        - trans(v, kappa, u) + np.dot(v.T, v.conj())
    kappa_emb = trans(u.conj(), kappa, u) - trans(v.conj(), kappa.conj(), v) \
        + trans(u.conj(), rhoA, v) + trans(v.conj(), rhoA, u) - np.dot(u.T.conj(), v.conj())
  return rho_emb, kappa_emb

###################################################################

class LatticeIntegral(object):
  def __init__(self, lattice, Int2e, Int2eShape, UHFB, thr_rdm = 0., thr_int = 0.):
    self.Int2e = Int2e
    self.Int2eShape = Int2eShape
    self.maskInt2e = None
    self.sInt2e = None
    self.UHFB = UHFB
    self.thr_rdm = thr_rdm
    self.thr_int = thr_int
    self.lattice = lattice
    
  def __Int2e_to_cell(self, Int2e):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells
    Int2eCell = np.zeros((ncells, ncells, ncells, nscsites, \
        nscsites, nscsites, nscsites))
    for j,k,l in it.product(range(ncells), repeat = 3):
      Int2eCell[j,k,l] = Int2e[:, j*nscsites: (j+1)*nscsites, k*nscsites: (k+1)*nscsites, \
          l*nscsites: (l+1)*nscsites]
    return Int2eCell

  def __to_sparse_Int2e(self):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells
    
    if self.Int2eShape == "Lat":
      Int2eCell = self.__Int2e_to_cell(self.Int2e)
      self.maskInt2e = np.array(np.nonzero(la.norm(Int2eCell.reshape(ncells, \
          ncells, ncells, nscsites**4), axis = 3) > self.thr_int*nscsites)).T
      sInt2e = [Int2eCell[tuple(idx)] for idx in self.maskInt2e]
    else:
      self.maskInt2e = np.zeros((1,3), dtype = int)
      sInt2e = [self.Int2e]
    self.sInt2e = np.array(sInt2e)

  def get_sparse_Int2e(self):
    if self.sInt2e is None:
      self.__to_sparse_Int2e()
    return self.sInt2e, self.maskInt2e

  def __to_sparse_rdm(self, rdm):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells

    rdm1 = np.zeros((ncells, ncells, nscsites, nscsites))
    for i, j in it.product(range(ncells), repeat = 2):
      rdm1[i, j] = rdm[i*nscsites: (i+1)*nscsites, j*nscsites: (j+1)*nscsites]
    return rdm1

  def __call__(self, basis):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells

    u, v = basis.u, basis.v

    if self.sInt2e is None:
      self.__to_sparse_Int2e()

    if self.UHFB:
      rdms = map(self.__to_sparse_rdm, [np.dot(v[0], v[0].T), np.dot(v[1], v[1].T), np.dot(u[0], v[1].T)])
      mask_rdm = reduce(np.ndarray.__add__, \
          map(lambda x: la.norm(x, axis = (2,3)) > self.thr_rdm*nscsites, rdms))
      J = np.zeros((nscsites*ncells, nscsites*ncells))
      Ka = np.zeros((nscsites*ncells, nscsites*ncells))
      Kb = np.zeros((nscsites*ncells, nscsites*ncells))
      L = np.zeros((nscsites*ncells, nscsites*ncells))
      
      for idx, jkl in enumerate(self.maskInt2e):
        for i in range(ncells):
          j, k, l = map(lambda x: self.lattice.add(x, i), jkl)
          if mask_rdm[k, l]:
            J[i*nscsites:(i+1)*nscsites, j*nscsites:(j+1)*nscsites] += \
                0.5 * np.einsum("ijkl,kl->ij", self.sInt2e[idx], rdms[0][k,l]+rdms[1][k,l])
          if mask_rdm[k,j]:
            Ka[i*nscsites:(i+1)*nscsites, l*nscsites:(l+1)*nscsites] += \
                np.einsum("ijkl,kj->il", self.sInt2e[idx], rdms[0][k,j])
            Kb[i*nscsites:(i+1)*nscsites, l*nscsites:(l+1)*nscsites] += \
                np.einsum("ijkl,kj->il", self.sInt2e[idx], rdms[1][k,j])
          if mask_rdm[j,l]:
            L[i*nscsites:(i+1)*nscsites, k*nscsites:(k+1)*nscsites] += \
                np.einsum("ijkl,jl->ik", self.sInt2e[idx], rdms[2][j,l])
      return J*2-Ka, J*2-Kb, -L
    else:
      rdms = map(to_sparse_rdm, [np.dot(v, v.T), np.dot(u, v.T)])
      mask_rdm = np.nonzero(reduce(np.ndarray.__add__, \
          map(lambda x: la.norm(x, axis = (2,3)) > self.thr_rdm*nscsites, rdms)))
      J = np.zeros((nscsites*ncells, nscsites*ncells))
      K = np.zeros((nscsites*ncells, nscsites*ncells))
      L = np.zeros((nscsites*ncells, nscsites*ncells))

      for idx, jkl in enumerate(self.maskInt2e):
        for i in range(ncells):
          j, k, l = map(lambda x: self.lattice.add(x, i), jkl)
          if mask_rdm[k, l]:
            J[i*nscsites:(i+1)*nscsites, j*nscsites:(j+1)*nscsites] += \
                np.einsum("ijkl,kl->ij", self.sInt2e[idx], rdms[0][k,l])
          if mask_rdm[k,j]:
            K[i*nscsites:(i+1)*nscsites, l*nscsites:(l+1)*nscsites] += \
                np.einsum("ijkl,kj->il", self.sInt2e[idx], rdms[0][k,j])
          if mask_rdm[j,l]:
            L[i*nscsites:(i+1)*nscsites, k*nscsites:(k+1)*nscsites] += \
                np.einsum("ijkl,jl->ik", self.sInt2e[idx], rdms[1][j,l])
      return J*2-K, None, L

###############################################################

def transform_EmbIntHam_MPS(basis, lattice, UHFB):
  U = lattice.ham.get_U()
  u, v = basis.u, basis.v
  sites = range(lattice.supercell.nsites)
  if UHFB:
    eps = [[], []]
    f = [[], []]
    g = [[], []]
    d = [[], []]
    for i in range(2):
      eps[i] = [la.norm(v[i][site]) ** 2 for site in sites]
      f[i] = [np.dot(u[i][site:site+1].T.conj(), u[i][site:site+1]) for site in sites]
      g[i] = [-np.dot(v[i][site:site+1].T.conj(), v[i][site:site+1]) for site in sites]
      d[i] = [np.dot(u[i][site:site+1].T.conj(), v[i][site:site+1].conj()) for site in sites]
  else:
    eps = [la.norm(v[site]) ** 2 for site in sites]
    f = [np.dot(u[site:site+1].T.conj(), u[site:site+1]) for site in sites]
    g = [-np.dot(v[site:site+1].T.conj(), v[site:site+1]) for site in sites]
    d = [-np.dot(u[site:site+1].T.conj(), v[site:site+1].conj()) for site in sites]
  
  return ToClass({"U": U, "eps": eps, "f": f, "g": g, "d": d})

def transform_EmbIntHam_Dump(basis, lattice, trans_lat_int2e, BathInt, UHFB):
  nscsites = lattice.supercell.nsites
  ncells = lattice.nscells
  if not BathInt:
    maskInt2e = [np.array([0, 0, 0], dtype = int)]
    sInt2e = [lattice.Ham.Int2e[:, :nscsites, :nscsites, :nscsites]]
    irange = [0]
  else:
    sInt2e, maskInt2e = trans_lat_int2e.get_sparse_Int2e()
    irange = range(ncells)
  
  return transform_Integral_DumpStyle(nscsites, basis, sInt2e, maskInt2e, irange, BathInt, UHFB, lattice.add)

def transform_Integral_DumpStyle(nscsites, basis, sInt2e, maskInt2e, irange, BathInt, UHFB, add):
  u, v = basis.u, basis.v
  
  # see doc for the formula
  # attention, the formula only works for real integrals
  if UHFB:
    x = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    v_a = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    v_b = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    w_a = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    w_b = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    w_ab = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    vcc = np.zeros((nscsites*2, nscsites*2))
    vcd = np.zeros((nscsites*4, nscsites*4))
    v0 = 0.
    if BathInt:
      # FIXME does the frag-lattice part of Int2e have the correct symmetry?
      x_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      v_a_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      v_b_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      w_a_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      w_b_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      w_ab_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      vcc_frag = np.zeros((nscsites*2, nscsites*2))
      vcd_frag = np.zeros((nscsites*4, nscsites*4))
      v0_frag = 0.
    for idx, kjl in enumerate(maskInt2e):
      for i in irange:
        k, j, l = map(lambda x: add(x, i), kjl)
        ui, uk, uj, ul = map(lambda p: [u0[p*nscsites:(p+1)*nscsites] for u0 in u], [i,k,j,l])
        vi, vk, vj, vl = map(lambda p: [v0[p*nscsites:(p+1)*nscsites] for v0 in v], [i,k,j,l])
        
        # vcccc
        x1a = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui[0].conj())
        x1b = np.einsum("ikjl,ip->pkjl", sInt2e[idx], vi[1].conj())
        x2 = np.einsum("pkjl,kr->prjl", x1a, vk[0].conj()) - np.einsum("pkjl,kr->prjl", x1b, uk[1].conj())
        x2a = np.einsum("prjl,jq->prql", x2, uj[0].conj())
        x2b = np.einsum("prjl,jq->prql", x2, vj[1].conj())
        x3 = np.einsum("prql,ls->prqs", x2a, vl[0].conj()) - np.einsum("prql,ls->prqs", x2b, ul[1].conj())
        temp_x = x3 - np.swapaxes(x3, 1, 3)
        # vcccd
        v1_a1 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui[0].conj())
        v1_a2 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], vi[1].conj())        
        v2_a = np.einsum("pkjl,ks->psjl", v1_a1, uk[0]) - np.einsum("pkjl,ks->psjl", v1_a2, vk[1])
        v3_a1 = np.einsum("psjl,jq->psql", v2_a, uj[0].conj())
        v3_a2 = np.einsum("psjl,jq->psql", v2_a, vj[1].conj())
        v4_a = np.einsum("psql,lr->psqr", v3_a1, vl[0].conj()) - \
            np.einsum("psql,lr->psqr", v3_a2, ul[1].conj())
        temp_v_a = v4_a - np.swapaxes(v4_a, 0, 2)

        v1_b1 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui[1].conj())
        v1_b2 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], vi[0].conj())
        v2_b = np.einsum("pkjl,ks->psjl", v1_b1, uk[1]) - np.einsum("pkjl,ks->psjl", v1_b2, vk[0])
        v3_b1 = np.einsum("psjl,jq->psql", v2_b, uj[1].conj())
        v3_b2 = np.einsum("psjl,jq->psql", v2_b, vj[0].conj())
        v4_b = np.einsum("psql,lr->psqr", v3_b1, vl[1].conj()) - \
            np.einsum("psql,lr->psqr", v3_b2, ul[0].conj())
        temp_v_b = v4_b - np.swapaxes(v4_b, 0, 2)

        # vccdd_aa and vccdd_bb
        w1_a1 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui[0].conj())
        w1_a2 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], vi[1].conj())
        w2_a = np.einsum("pkjl,kr->prjl", w1_a1, uk[0]) - np.einsum("pkjl,kr->prjl", w1_a2, vk[1])
        w3_a1 = np.einsum("prjl,jq->prql", w2_a, uj[0].conj())
        w3_a2 = np.einsum("prjl,jq->prql", w2_a, vj[1].conj())
        temp_w_a = np.einsum("prql,ls->prqs", w3_a1, ul[0]) - np.einsum("prql,ls->prqs", w3_a2, vl[1])

        w1_b1 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui[1].conj())
        w1_b2 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], vi[0].conj())
        w2_b = np.einsum("pkjl,kr->prjl", w1_b1, uk[1]) - np.einsum("pkjl,kr->prjl", w1_b2, vk[0])
        w3_b1 = np.einsum("prjl,jq->prql", w2_b, uj[1].conj())
        w3_b2 = np.einsum("prjl,jq->prql", w2_b, vj[0].conj())
        temp_w_b = np.einsum("prql,ls->prqs", w3_b1, ul[1]) - np.einsum("prql,ls->prqs", w3_b2, vl[0])

        # vccdd_ab
        w13_ab1 = np.einsum("prjl,jq->prql", w2_a, uj[1].conj())
        w13_ab2 = np.einsum("prjl,jq->prql", w2_a, vj[0].conj())
        w14_ab = np.einsum("prql,ls->prqs", w13_ab1, ul[1]) - np.einsum("prql,ls->prqs", w13_ab2, vl[0])
        
        w22_ab = np.einsum("pkjl,kq->pqjl", w1_a1, vk[0].conj()) - \
            np.einsum("pkjl,kq->pqjl", w1_a2, uk[1].conj())
        w23_ab1 = np.einsum("pqjl,js->pqls", w22_ab, vj[0])
        w23_ab2 = np.einsum("pqjl,js->pqls", w22_ab, uj[1])
        w24_ab = np.einsum("pqls,lr->pqrs", w23_ab1, ul[0]) - np.einsum("pqls,lr->pqrs", w23_ab2, vl[1])
        temp_w_ab = w14_ab + np.swapaxes(w24_ab, 1, 2)
    
        # vcc
        d1_1 = np.einsum("ikjl,jr,lr->ik", sInt2e[idx], vj[0].conj(), vl[0]) + \
            np.einsum("ikjl,jr,lr->ik", sInt2e[idx], vj[1].conj(), vl[1])
        d1_2_1 = np.einsum("ik,ip->pk", d1_1, ui[0].conj())
        d1_2_2 = np.einsum("ik,ip->pk", d1_1, vi[1].conj())
        d1 = np.einsum("pk,kq->pq", d1_2_1, vk[0].conj()) - \
            np.einsum("pk,kq->pq", d1_2_2, uk[1].conj()) # first term
        
        d2_1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[0].conj(), uj[1])
        d2_1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[0].conj(), vj[0])
        d2_2 = np.einsum("il,lq->iq", d2_1_1, ul[1].conj()) - np.einsum("il,lq->iq", d2_1_2, vl[0].conj())
        d2 = np.einsum("iq,ip->pq", d2_2, ui[0].conj())
        
        d3_1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[1].conj(), vj[1])
        d3_1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[1].conj(), uj[0])
        d3_2 = np.einsum("il,lp->pi", d3_1_1, vl[1].conj()) - np.einsum("il,lp->pi", d3_1_2, ul[0].conj())
        d3 = np.einsum("pi,iq->pq", d3_2, ui[1].conj())

        d4_2 = np.einsum("il,lp->pi", d2_1_1.conj(), vl[1].conj()) - \
            np.einsum("il,lp->pi", d2_1_2.conj(), ul[0].conj())
        d4 = np.einsum("pi,iq->pq", d4_2, vi[0].conj())

        d5_2 = np.einsum("il,lq->iq", d3_1_1.conj(), ul[1].conj()) - \
            np.einsum("il,lq->iq", d3_1_2.conj(), vl[0].conj())
        d5 = np.einsum("iq,ip->pq", d5_2, vi[1].conj())
        temp_d = d1 + 0.5 * (d2 + d3 + d4 + d5)

        # vcd
        h1_a2_1 = np.einsum("ik,ip->pk", d1_1, ui[0].conj())
        h1_a2_2 = np.einsum("ik,ip->pk", d1_1, vi[1].conj())
        h1_a = np.einsum("pk,kq->pq", h1_a2_1, uk[0]) - np.einsum("pk,kq->pq", h1_a2_2, vk[1])
        
        h1_b2_1 = np.einsum("ik,ip->pk", d1_1, ui[1].conj())
        h1_b2_2 = np.einsum("ik,ip->pk", d1_1, vi[0].conj())
        h1_b = np.einsum("pk,kq->pq", h1_b2_1, uk[1]) - np.einsum("pk,kq->pq", h1_b2_2, vk[0])

        h2_a1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[0].conj(), uj[1])
        h2_a1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[0].conj(), vj[0])
        h2_a2 = np.einsum("il,lq->iq", h2_a1_1, vl[1]) - np.einsum("il,lq->iq", h2_a1_2, ul[0])
        h2_a = np.einsum("iq,ip->pq", h2_a2, ui[0].conj())
        h3_a2 = np.einsum("il,lp->pi", h2_a1_1.conj(), vl[1].conj()) - \
            np.einsum("il,lp->pi", h2_a1_2.conj(), ul[0].conj())
        h3_a = np.einsum("pi,iq->pq", h3_a2, ui[0])
        
        h2_b1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[1].conj(), uj[0])
        h2_b1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[1].conj(), vj[1])
        h2_b2 = np.einsum("il,lq->iq", h2_b1_1, vl[0]) - np.einsum("il,lq->iq", h2_b1_2, ul[1])
        h2_b = np.einsum("iq,ip->pq", h2_b2, ui[1].conj())
        h3_b2 = np.einsum("il,lp->pi", h2_b1_1.conj(), vl[0].conj()) - \
            np.einsum("il,lp->pi", h2_b1_2.conj(), ul[1].conj())
        h3_b = np.einsum("pi,iq->pq", h3_b2, ui[1])

        h4_a1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[1], vj[1].conj())
        h4_a1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[1], uj[0].conj())
        h4_a2 = np.einsum("il,lq->iq", h4_a1_1, vl[1]) - np.einsum("il,lq->iq", h4_a1_2, ul[0])
        h4_a = np.einsum("iq,ip->pq", h4_a2, vi[1].conj())
        h5_a2 = np.einsum("il,lp->pi", h4_a1_1.conj(), vl[1].conj()) - \
            np.einsum("il,lp->pi", h4_a1_2.conj(), ul[0].conj())
        h5_a = np.einsum("pi,iq->pq", h5_a2, vi[1])

        h4_b1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[0], vj[0].conj())
        h4_b1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk[0], uj[1].conj())
        h4_b2 = np.einsum("il,lq->iq", h4_b1_1, vl[0]) - np.einsum("il,lq->iq", h4_b1_2, ul[1])
        h4_b = np.einsum("iq,ip->pq", h4_b2, vi[0].conj())
        h5_b2 = np.einsum("il,lp->pi", h4_b1_1.conj(), vl[0].conj()) - \
            np.einsum("il,lp->pi", h4_b1_2.conj(), ul[1].conj())
        h5_b = np.einsum("pi,iq->pq", h5_b2, vi[0])
        
        temp_h = np.zeros((nscsites*4, nscsites*4))
        temp_h[::2, ::2] = h1_a + 0.5 * (h2_a + h3_a + h4_a + h5_a)
        temp_h[1::2, 1::2] = h1_b + 0.5 * (h2_b + h3_b + h4_b + h5_b)

        # v0
        e1_1 = np.einsum("ikjl,ip,kp->jl", sInt2e[idx], vi[0], vk[0].conj()) + \
            np.einsum("ikjl,ip,kp->jl", sInt2e[idx], vi[1], vk[1].conj())
        e1 = np.einsum("jl,jq,lq->", e1_1, vj[0], vl[0].conj()) + \
            np.einsum("jl,jq,lq->", e1_1, vj[1], vl[1].conj())
        
        e2_1_1 = np.einsum("ikjl,ip,jp->kl", sInt2e[idx], vi[0], uj[1].conj())
        e2_1_2 = np.einsum("ikjl,ip,jp->kl", sInt2e[idx], vi[0], vj[0].conj())
        e2 = np.einsum("kl,kq,lq->", e2_1_1, vk[0].conj(), ul[1]) - \
            np.einsum("kl,kq,lq->", e2_1_2, vk[0].conj(), vl[0])
    
        e3_1_1 = np.einsum("ikjl,ip,jp->kl", sInt2e[idx], vi[1], uj[0].conj())
        e3_1_2 = np.einsum("ikjl,ip,jp->kl", sInt2e[idx], vi[1], vj[1].conj())
        e3 = np.einsum("kl,kq,lq->", e3_1_1, vk[1].conj(), ul[0]) - \
            np.einsum("kl,kq,lq->", e3_1_2, vk[1].conj(), vl[1])

        temp_e = 0.5 * (e1+e2+e3)

        x += temp_x # x_{prqs}
        v_a += temp_v_a # exchange p, q indices to antisymmetrize, v_{psqr,a}
        v_b += temp_v_b
        w_a += temp_w_a
        w_b += temp_w_b
        w_ab += temp_w_ab # w_{prqs,ab}
        vcc += temp_d
        vcd += temp_h
        v0 += temp_e

        if BathInt and 0 in [i,j,k,l]:
          # about the factor: in principle, we should restrict i = 0, and j,k,l be any number. However, we used 
          # the 8-fold symmetry of (ij||kl) in deriving the formula, so we have to preserve the symmetry, while making
          # the result equivalent to restricting i = 0
          # 
          # there are different situations:
          # a) 4 of i,j,k,l is 0: the only case is (00||00), factor = 1.
          # b) 3 of i,j,l,l is 0: (00||01), (00|10), (01||00), (10||00), factor = 3/4 = 0.75
          # c) 2 of i,j,k,l is 0:
          #    i) (00||11), (11||00) 0.5
          #   ii) (00||12), (00||21), (12||00), (21||00) 0.5
          #  iii) (10||10), (10||01), (01||10), (01||01) 0.5
          #   iv) (01||02), (01||20), (10||02), (10||20) <-> 4/8 = 0.5
          # d) 1 of i,j,k,l is 0:
          #    i) (01||11), (10||11), (11||01), (11||10) 0.25
          #   ii) (01||12), (01||21), (10||12), (10||21) <-> 0.25
          #  iii) (01||23), (01||32), (10||23), (10||32) <-> 0.25
          #   iv) (01||22), (10||22) <-> 0.25
          factor = np.sum(np.array([i,j,k,l]) == 0) * 0.25
          x_frag += factor * temp_x
          v_a_frag += factor * temp_v_a
          v_b_frag += factor * temp_v_b
          w_a_frag += factor * temp_w_a
          w_b_frag += factor * temp_w_b
          w_ab_frag += factor * temp_w_ab
          vcc_frag += factor * temp_d
          vcd_frag += factor * temp_h
          v0_frag += factor * temp_e

    # now transform to dict
    vcccc = {}
    for q, p in it.combinations(range(nscsites*2), 2):
      for s, r in it.combinations(range(nscsites*2), 2):
        vcccc[p,q,s,r] = x[p,r,q,s]
    vcccd = [{}, {}]
    for q, p in it.combinations(range(nscsites*2), 2):
      for r, s in it.product(range(nscsites*2), repeat = 2):
        vcccd[0][p,q,r,s] = v_a[p,s,q,r]
        vcccd[1][p,q,r,s] = v_b[p,s,q,r]
    vccdd = [{}, {}, {}]
    for (p, q, s, r) in it.product(range(nscsites*2), repeat = 4):
      if p*nscsites*2+r >= q*nscsites*2+s:
        vccdd[0][p,q,s,r] = w_a[p,r,q,s]
        vccdd[1][p,q,s,r] = w_b[p,r,q,s]
      vccdd[2][p,q,s,r] = w_ab[p,r,q,s]

    if BathInt:
      vcccc_frag = {}
      for q, p in it.combinations(range(nscsites*2), 2):
        for s, r in it.combinations(range(nscsites*2), 2):
          vcccc_frag[p,q,s,r] = x_frag[p,r,q,s]
      vcccd_frag = [{}, {}]
      for q, p in it.combinations(range(nscsites*2), 2):
        for r, s in it.product(range(nscsites*2), repeat = 2):
          vcccd_frag[0][p,q,r,s] = v_a_frag[p,s,q,r]
          vcccd_frag[1][p,q,r,s] = v_b_frag[p,s,q,r]
      vccdd_frag = [{}, {}, {}]
      for (p, q, s, r) in it.product(range(nscsites*2), repeat = 4):
        if p*nscsites*2+r >= q*nscsites*2+s:
          vccdd_frag[0][p,q,s,r] = w_a_frag[p,r,q,s]
          vccdd_frag[1][p,q,s,r] = w_b_frag[p,r,q,s]
        vccdd_frag[2][p,q,s,r] = w_ab_frag[p,r,q,s]

  else: # restricted
    x = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    coef_v = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    w = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
    vcc = np.zeros((nscsites*2, nscsites*2))
    vcd = np.zeros((nscsites*2, nscsites*2))
    v0 = 0.
    if BathInt:
      x_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      v_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      w_frag = np.zeros((nscsites*2, nscsites*2, nscsites*2, nscsites*2))
      vcc_frag = np.zeros((nscsites*2, nscsites*2))
      vcd_frag = np.zeros((nscsites*2, nscsites*2))
      v0_frag = 0.
    
    for idx, kjl in enumerate(maskInt2e):
      for i in irange:
        k, j, l = map(lambda x: add(x, i), kjl)
        ui, uk, uj, ul = map(lambda p: u[p*nscsites:(p+1)*nscsites], [i,k,j,l])
        vi, vk, vj, vl = map(lambda p: v[p*nscsites:(p+1)*nscsites], [i,k,j,l])
        # vcccc
        x1_0 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui.conj())
        x2_0 = np.einsum("pkjl,kr->prjl", x1_0, vk.conj())
        x2 = x2_0 + np.swapaxes(x2_0, 0, 1)
        x3_0 = np.einsum("prjl,jq,ls->prqs", x2, uj.conj(), vl.conj())
        x3 = x3_0 + np.swapaxes(x3_0, 2, 3)
        temp_x = x3 - np.swapaxes(x3, 1, 3)
        # vcccd
        v3_1 = np.einsum("prjl,jq->prql", x2, uj.conj())
        v3_2 = np.einsum("prjl,jq->prql", x2, vj.conj())
        v4 = np.einsum("prql,ls->prqs", v3_1, ul) - np.einsum("prql,ls->prqs", v3_2, vl)
        temp_v = v4 - np.swapaxes(v4, 0, 2) # swap p, q

        # vccdd
        w1_1_1 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui.conj())
        w1_1_2 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], vi.conj())
        w1_2 = np.einsum("pkjl,kr->prjl", w1_1_1, uk) - np.einsum("pkjl,kr->prjl", w1_1_2, vk)
        w1_3_1 = np.einsum("prjl,jq->prql", w1_2, uj.conj())
        w1_3_2 = np.einsum("prjl,jq->prql", w1_2, vj.conj())
        w1_4 = np.einsum("prql,ls->prqs", w1_3_1, ul) - np.einsum("prql,ls->prqs", w1_3_1, vl)
        
        w2_1 = np.einsum("ikjl,ip->pkjl", sInt2e[idx], ui.conj())
        w2_2_0 = np.einsum("pkjl,kq->pjql", w2_1, vk.conj())
        w2_2 = w2_2_0 + np.swapaxes(w2_2_0, 0, 2)
        w2_3 = np.einsum("pjql,jr->prql", w2_2, uj)
        w2_4_0 = np.einsum("prql,ls->prqs", w2_3, vl)
        w2_4 = w2_4_0 + np.swapaxes(w2_4_0, 1, 3)
        temp_w = w1_4 + w2_4

        # vcc
        d1_1 = np.einsum("ikjl,jr,lr->ik", sInt2e[idx], vj.conj(), vl)
        d1_2 = np.einsum("ik,ip->pk", d1_1, ui.conj())
        d1_3 = np.einsum("pk,kq->pq", d1_2, vk.conj())

        d2_1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk.conj(), vj)
        d2_1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk.conj(), uj)
        d2_2 = np.einsum("il,lq->iq", d2_1_1, vl.conj()) - np.einsum("il,lq->iq", d2_1_2, ul.conj())
        d2_3 = np.einsum("iq,ip->pq", d2_2, ui.conj())

        d3_1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk, vj.conj())
        d3_1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk, uj.conj())
        d3_2 = np.einsum("il,lq->iq", d3_1_1, ul.conj()) + np.einsum("il,lq->iq", d3_1_2, vl.conj())
        d3_3 = np.einsum("iq,ip->pq", d3_2, vi.conj())

        d4 = -2. * d1_3 + 0.5 * (d2_3 + d3_3)
        temp_d = d4 + d4.T # swap p,q

        # vcd
        h1_1 = np.einsum("ikjl,jr,lr->ik", sInt2e[idx], vj, vl.conj())
        h1_2_1 = np.einsum("ik,ip->pk", h1_1, ui.conj())
        h1_2_2 = np.einsum("ik,iq->kq", h1_1, vi)
        h1_3 = np.einsum("pk,kq->pq", h1_2_1, uk) - np.einsum("kq,kp->pq", h1_2_2, vk.conj())
        
        h2_1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk.conj(), uj)
        h2_1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk.conj(), vj)
        h2_2 = np.einsum("il,lq->iq", h2_1_1, vl) + np.einsum("il,lq->iq", h2_1_2, ul)
        h2_3 = np.einsum("iq,ip->pq", h2_2, ui.conj())

        h3_1_1 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk, vj.conj())
        h3_1_2 = np.einsum("ikjl,kr,jr->il", sInt2e[idx], vk, uj.conj())
        h3_2 = np.einsum("il,lq->iq", h3_1_1, vl) - np.einsum("il,lq->iq", h3_1_2, ul)
        h3_3 = np.einsum("iq,ip->pq", h3_2, vi.conj())

        h4 = h1_3 - 0.5*(h2_3-h3_3)
        temp_h = h4 + h4.T.conj()

        # v0
        e1_1_1 = np.einsum("ikjl,ip,jp->kl", sInt2e[idx], vi, uj.conj())
        e1_1_2 = np.einsum("ikjl,ip,jp->kl", sInt2e[idx], vi, vj.conj())
        e1_2 = np.einsum("kl,kq,lq->", e1_1_1, vk.conj(), ul) - \
            np.einsum("kl,kq,lq->", e1_1_2, vk.conj(), vl)

        e2_1 = np.einsum("ikjl,ip,kp->jl", sInt2e[idx], vi, vk.conj())
        e2_2 = np.einsum("jl,jq,lq->", e2_1, vj, vl.conj())

        temp_e = e1_2 + 2.*e2_2

        x += temp_x
        coef_v += temp_v
        w += temp_w
        vcc += temp_d
        vcd += temp_h
        v0 += temp_e

        if BathInt and 0 in [i,j,k,l]:
          factor = np.sum(np.array([i,j,k,l]) == 0) * 0.25
          x_frag += factor * temp_x # x_{prqs}
          v_frag += factor * temp_v
          w_frag += factor * temp_w
          vcc_frag += factor * temp_d
          vcd_frag += factor * temp_h
          v0_frag += factor * temp_e

    vcccc = {}
    for q, p in it.combinations(range(nscsites*2), 2):
      for s, r in it.combinations(range(nscsites*2), 2):
        if p > r or (p == r and q >= s):
          vcccc[p,q,s,r] = x[p,r,q,s]
    vcccd = {}
    for q, p in it.combinations(range(nscsites*2), 2):
      for r, s in it.product(range(nscsites*2), repeat = 2):
        vcccd[p,q,r,s] = coef_v[p,s,q,r]
    vccdd = {}
    for (p, q, s, r) in it.product(range(nscsites*2), repeat = 4):
      if p*nscsites*2+r >= q*nscsites*2+s:
        vccdd[p,q,s,r] = w[p,r,q,s]

    if BathInt:
      vcccc_frag = {}
      for q, p in it.combinations(range(nscsites*2), 2):
        for s, r in it.combinations(range(nscsites*2), 2):
          if p > r or (p == r and q >= s):
            vcccc_frag[p,q,s,r] = x_frag[p,r,q,s]
      vcccd_frag = {}
      for q, p in it.combinations(range(nscsites*2), 2):
        for r, s in it.product(range(nscsites*2), repeat = 2):
          vcccd_frag[p,q,r,s] = v_frag[p,s,q,r]
      vccdd_frag = {}
      for (p, q, s, r) in it.product(range(nscsites*2), repeat = 4):
        if p*nscsites*2+r >= q*nscsites*2+s:
          vccdd_frag[p,q,s,r] = w_frag[p,r,q,s]
  if BathInt:
    return (ToClass({"cd":vcd, "cc":vcc}), ToClass({"ccdd":vccdd, "cccd":vcccd, "cccc":vcccc})), \
        (v0_frag, ToClass({"cd":vcd_frag, "cc":vcc_frag}), ToClass({"ccdd":vccdd_frag, \
        "cccd":vcccd_frag, "cccc":vcccc_frag}))
  else:
    return (ToClass({"cd":vcd, "cc":vcc}), ToClass({"ccdd":vccdd, "cccd":vcccd, "cccc":vcccc})), \
        (v0, ToClass({"cd":vcd, "cc":vcc}), ToClass({"ccdd":vccdd, "cccd":vcccd, "cccc":vcccc}))
