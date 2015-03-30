import numpy as np
import numpy.linalg as la
import itertools as it
import random as r
from math import sqrt

def TriLocalize(u_emb, v_emb, nImp, verbose = 0):
  if verbose > 1:
    print
    print "********* Trigonal Localization of Quasiparticles *********"
    print
  
  # a very naive localization method
  # make the embedding basis trigonal in impurity range
  nsites = u_emb.shape[0]
  emb = np.zeros((nsites*2, nImp*2))
  emb[::2] = u_emb
  emb[1::2] = v_emb

  for k in range(nImp):
    S = np.dot(emb[2*k:2*(k+1), 0:(nImp-k)*2].T, emb[2*k:2*(k+1), 0:(nImp-k)*2])
    ew, ev = la.eigh(S)
    emb[:, :(nImp-k)*2] = np.dot(emb[:, :(nImp-k)*2], ev)

  u_emb = emb[::2, ::-1]
  v_emb = emb[1::2, ::-1]

  def reorder(k):
    # first fix phase
    if emb[k*2, k*2] < 0.:
      emb[:, k*2] *= -1
    if emb[k*2, k*2+1] < 0.:
      emb[:, k*2+1] *= -1
    # then reorder
    if emb[k*2, k*2] < emb[k*2, k*2+1]:
      emb[:, [k*2+1, k*2]] = emb[:, [k*2, k*2+1]]

  for k in range(nImp):
    # now reorder and correct phase of localized emb orbitals
    reorder(k)
  
  return u_emb, v_emb

def unitary_opt(fn, u0, method = "CGPR", verbose = 0, gradient = None):
  # Based on "Unitary optimization of localized molecular orbitals"
  #  by Susi Lehtola and Hannes Jonsson
  #     Dept. of Applied Phys., Aalto University, Finland
  u = u0
  assert(u.shape[0] == u.shape[1])
  n = u.shape[0]
  assert(np.allclose(np.dot(u.T, u), np.eye(n)))

  from scipy.linalg import expm2
  from scipy.optimize import fmin

  def Num_Grad(u):
    g = np.zeros_like(u)
    step = 1e-8
    for i in range(n):
      for j in range(n):
        du = np.zeros_like(u)
        du[i, j] = step
        g[i, j] = (0.5/step) * (fn(u+du) - fn(u-du))
    # transform to Riemannian derivative
    g = np.dot(g, u.T) - np.dot(u, g.T)
    return g
  
  if verbose > 2:
    print " Iter.        Value             Grad               Step"

  last_r = 100000
  for iter in range(2000):
    r = fn(u)
    if ((abs(r-last_r) < 1e-4 or r < 1e-2) and iter != 0):
      if verbose > 1:
        print "Localization converged."
      break
    
    # compute Riemannian gradient
    if gradient is not None:
      g = gradient(u)
    else:
      g = Num_Grad(u)
    norm_g = la.norm(g)
    
    if norm_g < 1e-2:
      if verbose > 1:
        print "Localization converged."
      break
    
    cg_dim = max(n/2, 4)
    def GetDir():
      if method == "SD" or iter % cg_dim == 0:
        return g
      elif method == "CGFR":
        gamma = np.trace(np.dot(g.T, g)) / np.trace(np.dot(last_g.T, last_g))
        h = g + last_g * gamma
      elif method == "CGPR":
        gamma = np.trace(np.dot(g.T, g-last_g)) / np.trace(np.dot(last_g.T, last_g))
        h = g + last_g * gamma
      else:
        raise Exception("method not available")
      
      if np.trace(np.dot(h.T, g)) < 0:
        return g
      else:
        return h

    # get search direction
    h = GetDir()

    # now linear search
    def LineSearchFn(step):
      r1 = fn(np.dot(expm2(-step*h), u))
      return r1

    def FindStep():
      #grid = list(np.arange(0., 0.5, 0.025))
      #val = [LineSearchFn(step) for step in grid]
      #s = grid[np.argmin(val)]
      #if (abs(s) > 1e-4):
      #  return s
      #else:
      return fmin(LineSearchFn, np.array([0.001]), disp=0, xtol=1e-10)[0]
    
    step = FindStep()
    
    if verbose > 2:
      print "%3d  %16.6f  %16.6f  %16.6f" % (iter, r, norm_g, step)

    u = np.dot(expm2(-step*h), u)
    last_g = g
    last_r = r

  return u, fn(u)

def PinLocalize(u_emb, v_emb, topology, verbose = 0):
  if topology is None:
    raise Exception("Must have a valid topology to use pin localization")
  
  nsites = topology.distance.shape[1]
  nImp = topology.distance.shape[0]
  dis = np.zeros((nImp*2, nsites))
  dis[:nImp] = dis[nImp:] = topology.distance ** 2

  u = np.zeros((nImp*2, nsites))
  u[:nImp], u[nImp:] = u_emb[:, ::2].T, u_emb[:, 1::2].T
  v = np.zeros((nImp*2, nsites))
  v[:nImp], v[nImp:] = v_emb[:, ::2].T, v_emb[:, 1::2].T

  def metric(u, v, verbose = 0):
    return np.sum((u**2+v**2) * dis)
  #def get_grad_tensor():


  #nsites = np.product(LatShape)
  #nImp = np.product(ImpShape)
  
  #topo = VirtualTopology(ImpShape, LatShape, idx2pos)
  
  #pin = [topo.r[i/2] for i in range(nImp*2)]
  
  #nsites1 = len(topo.s)
  #u_emb1 = np.zeros((nsites1, nImp*2))
  #v_emb1 = np.zeros((nsites1, nImp*2))
  #for i in range(nsites1):
  #  u_emb1[i] = u_emb[topo.s[i]]
  #  v_emb1[i] = v_emb[topo.s[i]]

  #def metric(u_emb, v_emb, pinpoints, verbose = 0):
  #  m = 0.
  #  for i in range(nImp * 2):
  #    m_i = 0.
  #    for j in range(nsites1):
  #      m_i += (u_emb[j, i]**2+v_emb[j, i]**2) * la.norm(topo.r[j]-pin[i])**2
  #    if verbose > 4:
  #      print m_i
  #    m += m_i
  #  if verbose > 4:
  #    print
  #  return m
  
  def get_grad_tensor():
    AA = np.einsum("qk,jk->jkq", u, u) + np.einsum("qk,jk->jkq", v, v)
    return 2*np.einsum("pk,jkq->jpq", dis, AA)
  F = get_grad_tensor()
  
  def grad(U):
    g = np.einsum("pj,jpq->pq", U, F)
    return np.dot(g, U.T) - np.dot(U, g.T)
  
  #F = get_grad_tensor()
  #def grad(u):
  #  g = np.zeros_like(u)
  #  for l in range(nImp*2):
  #    g[:, l] = np.dot(F[l], u[:, l])
  #  g = np.dot(g, u.T) - np.dot(u, g.T)    
  #  return g
  
  U = np.eye(nImp*2, nImp*2)

  if verbose > 1:
    print
    print "********** Pinned Localization of Quasiparticles **********"
    print

  U, r = unitary_opt(lambda unitary: metric(np.dot(unitary, u), np.dot(unitary, v)), \
      U, verbose = verbose, gradient = grad)
 
  assert(np.allclose(np.dot(U.T, U), np.eye(nImp*2)))
  u, v = np.dot(U, u), np.dot(U, v)
  u_emb[:, ::2], u_emb[:, 1::2] = u[:nImp].T, u[nImp:].T
  v_emb[:, ::2], v_emb[:, 1::2] = v[:nImp].T, v[nImp:].T
  
  def reorder(k):
    if max(abs(u_emb[:, k*2])) < max(abs(v_emb[:, k*2])) and max(abs(u_emb[:, k*2+1])) > max(abs(v_emb[:, k*2+1])):
      u_emb[:, [k*2+1, k*2]] = u_emb[:, [k*2, k*2+1]]
      v_emb[:, [k*2+1, k*2]] = v_emb[:, [k*2, k*2+1]]
    if u_emb[k, k*2] < 0:
      u_emb[:, k*2] *= -1
      v_emb[:, k*2] *= -1
    if v_emb[k, k*2+1] < 0:
      u_emb[:, k*2+1] *= -1
      v_emb[:, k*2+1] *= -1
  
  for k in range(nImp):
    reorder(k)
  return u_emb, v_emb

