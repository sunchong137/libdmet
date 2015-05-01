import numpy as np
import numpy.linalg as la

mf_coef = (np.load("coefA.npy"), np.load("coefB.npy"))
mp_rdm = (np.load("mp2_rdmA.npy"), np.load("mp2_rdmB.npy"))
basis = (np.load("basisA.npy"), np.load("basisB.npy"))

#print la.eigh(mp_rdm[0])[0]

rdm = (np.dot(mf_coef[0], np.dot(mp_rdm[0], mf_coef[0].T)), # in the embedding basis
       np.dot(mf_coef[1], np.dot(mp_rdm[1], mf_coef[1].T)))

#print np.diag(rdm[0])
#print np.diag(rdm[1])

ew1, ev1 = la.eigh(rdm[0])
ew2, ev2 = la.eigh(rdm[1])

for ncore in range(80, 120, 4):
  print ncore
  core = (np.dot(ev1[:,-ncore:], ev1[:,-ncore:].T),
          np.dot(ev2[:,-ncore:], ev2[:,-ncore:].T))
  
  rdm_real = (np.dot(basis[0], np.dot(core[0], basis[0].T)), # in the real basis
         np.dot(basis[1], np.dot(core[1], basis[1].T)))
  
  print (np.diag(rdm_real[0]) + np.diag(rdm_real[1]))[[43,49,55,61,67,73,79,85]]
  #print la.eigh(rdm[0])[0]
  #print la.eigh(rdm[1])[0]

