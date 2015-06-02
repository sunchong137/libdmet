# localizer.py
# by Bo-Xiao Zheng
# Edmiston-Ruedenberg localization through Jacobi rotations
# following the algorithm by Raffenetti et al.
# Theor Chim Acta 86, 149 (1992)
import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
from math import cos, sin
from copy import deepcopy

class Localizer(object):
    
    def __init__(self, Int2e, thr = 1e-6): # Int2e[i,j,k,l] = (ij||kl)
        self.norbs = Int2e.shape[0]
        self.Int2e = deepcopy(Int2e)
        self.coefs = np.eye(self.norbs)
        self.thr = thr

    def transform(self, i, j, theta):
        # transform 2e integrals wrt Jacobi rotation J_ii = J_jj = cos\theta, J_ij = sin\theta, J_ij = -sin\theta
        # restrict to i < j
        # The scaling of this transformation is O(n^3)
        log.eassert(i != j, "rotation meaningless with i=j")
        # this works even for general cases where Int2e does not have symmetry
        delta = np.asarray([[cos(theta)-1, sin(theta)],[-sin(theta), cos(theta)-1]])
        # four index part
        g4 = self.Int2e[np.ix_([i,j],[i,j],[i,j],[i,j])]
        g4 = np.einsum("pi,qj,rk,sl,ijkl->pqrs", delta, delta, delta, delta, g4)
        # three index part
        g3_1 = self.Int2e[np.ix_(range(self.norbs), [i,j], [i,j], [i,j])]
        g3_1 = np.einsum("qj,rk,sl,pjkl->pqrs", delta, delta, delta, g3_1)
        g3_2 = self.Int2e[np.ix_([i,j], range(self.norbs), [i,j], [i,j])]
        g3_2 = np.einsum("pi,rk,sl,iqkl->pqrs", delta, delta, delta, g3_2)
        g3_3 = self.Int2e[np.ix_([i,j], [i,j], range(self.norbs), [i,j])]
        g3_3 = np.einsum("pi,qj,sl,ijrl->pqrs", delta, delta, delta, g3_3)
        g3_4 = self.Int2e[np.ix_([i,j], [i,j], [i,j], range(self.norbs))]
        g3_4 = np.einsum("pi,qj,rk,ijks->pqrs", delta, delta, delta, g3_4)
        # two index part
        g2_12 = self.Int2e[np.ix_(range(self.norbs), range(self.norbs), [i,j], [i,j])]
        g2_12 = np.einsum("rk,sl,pqkl->pqrs", delta, delta, g2_12)
        g2_13 = self.Int2e[np.ix_(range(self.norbs), [i,j], range(self.norbs), [i,j])]
        g2_13 = np.einsum("qj,sl,pjrl->pqrs", delta, delta, g2_13)
        g2_14 = self.Int2e[np.ix_(range(self.norbs), [i,j], [i,j], range(self.norbs))]
        g2_14 = np.einsum("qj,rk,pjks->pqrs", delta, delta, g2_14)
        g2_23 = self.Int2e[np.ix_([i,j], range(self.norbs), range(self.norbs), [i,j])]
        g2_23 = np.einsum("pi,sl,iqrl->pqrs", delta, delta, g2_23)
        g2_24 = self.Int2e[np.ix_([i,j], range(self.norbs), [i,j], range(self.norbs))]
        g2_24 = np.einsum("pi,rk,iqks->pqrs", delta, delta, g2_24)
        g2_34 = self.Int2e[np.ix_([i,j], [i,j], range(self.norbs), range(self.norbs))]
        g2_34 = np.einsum("pi,qj,ijrs->pqrs", delta, delta, g2_34)
        # one index part
        g1_1 = self.Int2e[[i,j], :, :, :]
        g1_1 = np.einsum("pi,iqrs->pqrs", delta, g1_1)
        g1_2 = self.Int2e[:, [i,j], :, :]
        g1_2 = np.einsum("qj,pjrs->pqrs", delta, g1_2)
        g1_3 = self.Int2e[:, :, [i,j], :]
        g1_3 = np.einsum("rk,pqks->pqrs", delta, g1_3)
        g1_4 = self.Int2e[:, :, :, [i,j]]
        g1_4 = np.einsum("sl,pqrl->pqrs", delta, g1_4)
        # sum over all rotations
        self.Int2e[np.ix_([i,j],[i,j],[i,j],[i,j])] += g4
        self.Int2e[np.ix_(range(self.norbs), [i,j], [i,j], [i,j])] += g3_1
        self.Int2e[np.ix_([i,j], range(self.norbs), [i,j], [i,j])] += g3_2
        self.Int2e[np.ix_([i,j], [i,j], range(self.norbs), [i,j])] += g3_3
        self.Int2e[np.ix_([i,j], [i,j], [i,j], range(self.norbs))] += g3_4
        self.Int2e[np.ix_(range(self.norbs), range(self.norbs), [i,j], [i,j])] += g2_12
        self.Int2e[np.ix_(range(self.norbs), [i,j], range(self.norbs), [i,j])] += g2_13
        self.Int2e[np.ix_(range(self.norbs), [i,j], [i,j], range(self.norbs))] += g2_14
        self.Int2e[np.ix_([i,j], range(self.norbs), range(self.norbs), [i,j])] += g2_23
        self.Int2e[np.ix_([i,j], range(self.norbs), [i,j], range(self.norbs))] += g2_24
        self.Int2e[np.ix_([i,j], [i,j], range(self.norbs), range(self.norbs))] += g2_34
        self.Int2e[[i,j], :, :, :] += g1_1
        self.Int2e[:, [i,j], :, :] += g1_2
        self.Int2e[:, :, [i,j], :] += g1_3
        self.Int2e[:, :, :, [i,j]] += g1_4

    def optimize(self):
        # Edmiston-Ruedenberg: maximizing self-energy
        # L = \sum_p (pp||pp)
        # each Jacobian step \theta between -pi/4 to pi/4
        


if __name__ == "__main__":
    s = np.random.rand(8,8,8,8)
    loc = Localizer(s)
    i = 2
    j = 7
    theta = -0.3
    loc.transform(i,j,theta)
    R = np.eye(8)
    R[i,i] = cos(theta)
    R[j,j] = cos(theta)
    R[i,j] = sin(theta)
    R[j,i] = -sin(theta)
    print la.norm(loc.Int2e - np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, s))
