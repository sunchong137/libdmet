# localizer.py
# by Bo-Xiao Zheng
# Edmiston-Ruedenberg localization through Jacobi rotations
# following the algorithm by Raffenetti et al.
# Theor Chim Acta 86, 149 (1992)

# Pipek-Mezey localization of Hubbard model.
# modified from pyscf.

import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
from math import cos, sin, atan, pi
from copy import deepcopy
import itertools as it
from pyscf import lo
from pyscf import gto

class Localizer(object):
    
    def __init__(self, Int2e): # Int2e[i,j,k,l] = (ij||kl)
        self.norbs = Int2e.shape[0]
        self.Int2e = deepcopy(Int2e)
        self.coefs = np.eye(self.norbs)

    def transformInt(self, i, j, theta):
        # transform 2e integrals wrt Jacobi rotation J_ii = J_jj = cos\theta, J_ij = sin\theta, J_ij = -sin\theta
        # restrict to i < j
        # The scaling of this transformation is O(n^3)
        log.eassert(i != j, "rotation meaningless with i=j")
        # this works even for general cases where Int2e does not have symmetry
        delta = np.asarray([[cos(theta)-1, sin(theta)],[-sin(theta), cos(theta)-1]])
        # four index part O(1)
        g4 = self.Int2e[np.ix_([i,j],[i,j],[i,j],[i,j])]
        g4 = np.einsum("pi,qj,rk,sl,ijkl->pqrs", delta, delta, delta, delta, g4)
        # three index part O(n)
        g3_1 = self.Int2e[np.ix_(range(self.norbs), [i,j], [i,j], [i,j])]
        g3_1 = np.einsum("qj,rk,sl,pjkl->pqrs", delta, delta, delta, g3_1)
        g3_2 = self.Int2e[np.ix_([i,j], range(self.norbs), [i,j], [i,j])]
        g3_2 = np.einsum("pi,rk,sl,iqkl->pqrs", delta, delta, delta, g3_2)
        g3_3 = self.Int2e[np.ix_([i,j], [i,j], range(self.norbs), [i,j])]
        g3_3 = np.einsum("pi,qj,sl,ijrl->pqrs", delta, delta, delta, g3_3)
        g3_4 = self.Int2e[np.ix_([i,j], [i,j], [i,j], range(self.norbs))]
        g3_4 = np.einsum("pi,qj,rk,ijks->pqrs", delta, delta, delta, g3_4)
        # two index part O(n^2)
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
        # one index part O(n^3)
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

    def transformCoef(self, i, j, theta):
        U = np.eye(self.norbs)
        U[i,i] = U[j,j] = cos(theta)
        U[i,j] = sin(theta)
        U[j,i] = -sin(theta)
        self.coefs = np.dot(U, self.coefs)

    def predictor(self, i, j):
        # for the rotation between orbitals i,j
        # we restrict theta in -pi/4, +pi/4
        # compute (i'i'||i'i')+(j'j'||j'j')-(ii||ii)-(jj||jj)
        # i'=i*cos\theta+j*sin\theta
        # j'=j*cos\theta-i*sin\theta
        # (i'i'||i'i')+(j'j'||j'j') = [(ii||ii)+(jj||jj)][3/4+1/4*cos4\theta]
        # + [(ii||ij)+...-(ij||jj)-...]*1/4*sin4\theta
        # + [(ii||jj)+...][1/4-1/4*cos4\theta]
        A = self.Int2e[i,i,i,i] + self.Int2e[j,j,j,j]
        B = self.Int2e[i,i,i,j] + self.Int2e[i,i,j,i] + self.Int2e[i,j,i,i] + self.Int2e[j,i,i,i] \
                - self.Int2e[i,j,j,j] - self.Int2e[j,i,j,j] - self.Int2e[j,j,i,j] - self.Int2e[j,j,j,i]
        C = self.Int2e[i,i,j,j] + self.Int2e[i,j,i,j] + self.Int2e[i,j,j,i] + self.Int2e[j,i,i,j] \
                + self.Int2e[j,i,j,i] + self.Int2e[j,j,i,i]

        def get_dL(theta):
            return 0.25 * ((cos(4*theta)-1) * (A-C) + sin(4*theta) * B)
        
        def get_theta():
            # solve dL/dtheta = 0, take theta that corresponds to maximum
            if abs(A-C) > 1e-8:
                alpha = atan(B/(A-C))
            else:
                alpha = pi/2
            if alpha > 0:
                theta = [alpha*0.25, (alpha-pi)*0.25]
            else:
                theta = [alpha*0.25, (alpha+pi)*0.25]
            vals = map(get_dL, theta)
            if vals[0] > vals[1]:
                return theta[0], vals[0]
            else:
                return theta[1], vals[1]

        return get_theta()

    def getL(self):
        return np.sum(map(lambda i: self.Int2e[i,i,i,i], range(self.norbs)))

    def optimize(self, thr = 1e-3, MaxIter = 2000):
        # Edmiston-Ruedenberg: maximizing self-energy
        # L = \sum_p (pp||pp)
        # each Jacobian step \theta between -pi/4 to pi/4
        if self.norbs < 2:
            log.info("Norb = %d, too few to localize", self.norbs)
            return
        Iter = 0
        log.info("Edmiston-Ruedenberg localization")
        initL = self.getL()
        log.debug(0, "Iter        L            dL     (i , j)   theta/pi")
        sweep = []
        for i,j in it.combinations(range(self.norbs), 2):
            sweep.append((i, j) + self.predictor(i, j))
        sweep.sort(key = lambda x: x[3])
        i, j, theta, dL = sweep[-1]
        log.debug(0, "%4d %12.6f %12.6f %3d %3d  %10.6f", \
                Iter, self.getL(), dL, i, j, theta/pi)
        while dL > thr and Iter < MaxIter:
            self.transformInt(i,j,theta)
            self.transformCoef(i,j,theta)
            Iter += 1
            sweep = []
            for i,j in it.combinations(range(self.norbs), 2):
                sweep.append((i, j) + self.predictor(i, j))
            sweep.sort(key = lambda x: x[3])
            i, j, theta, dL = sweep[-1]
            log.debug(0, "%4d %12.6f %12.6f %3d %3d  %10.6f", \
                    Iter, self.getL(), dL, i, j, theta/pi)
        log.info("Localization converged after %4d iterations", Iter)
        log.info("Cost function: init %12.6f   final %12.6f", initL, self.getL())

# add PM localization class. <ZHC

class HubbardPM(lo.pipek.PM):
    def __init__(self, *args, **kwargs):
        lo.pipek.PM.__init__(self, *args, **kwargs)
        self.init_guess = 'rand'
    def atomic_pops(self, mol, mo_coeff, method=None):
        return np.einsum('pi,pj->pij', mo_coeff, mo_coeff)
    def get_init_guess(self, key='atomic'):
        '''Generate initial guess for localization.

        Kwargs:
            key : str or bool
                If key is 'atomic', initial guess is based on the projected
                atomic orbitals. False
        '''
        nmo = self.mo_coeff.shape[1]
        if isinstance(key, str) and key.lower() == 'atomic':
            u0 = atomic_init_guess(self.mol, self.mo_coeff)
        else:
            u0 = np.eye(nmo)
        if (isinstance(key, str) and key.lower().startswith('rand')
            or np.linalg.norm(self.get_grad(u0)) < 1e-5):
            # Add noise to kick initial guess out of saddle point
            dr = np.cos(np.arange((nmo-1)*nmo//2)) * np.random.rand()
            u0 = self.extract_rotation(dr)
        return u0
# >ZHC

def localize_bath(B):
    B_shape = B.shape
    bath_orbs = B.reshape((-1, B_shape[-1])) # full
    log.info("PM localization of bath orbitals")
    loc_orb_collect = []

    for i in xrange(5):
        mol = gto.M()
        mol.verbose = 3
        loc_orb_input = bath_orbs
        loc_obj = HubbardPM(mol, loc_orb_input)
        cost_before = loc_obj.cost_function()
        if i == 0:
            log.debug(0, 'cost function of PM localization of bath orb (before): %12.5f', cost_before)
        log.debug(0, "%d th trial: ", i)

        loc_orb = loc_obj.kernel()
        loc_obj = HubbardPM(mol, loc_orb)
        loc_orb = loc_obj.kernel()
        loc_obj = HubbardPM(mol, loc_orb)
        loc_orb = loc_obj.kernel()
        loc_obj = HubbardPM(mol, loc_orb)
        loc_orb = loc_obj.kernel()
        
        cost_after = loc_obj.cost_function()
        loc_orb_collect.append((cost_after, loc_orb))
        log.debug(0, 'cost function of PM localization of bath orb (after): %12.5f', cost_after)
        
    loc_orb_collect.sort(key = lambda tup: tup[0])
    bath_orbs = loc_orb_collect[-1][1]
    return bath_orbs.reshape(B_shape)

def visualize_bath(lattice, LatSize, GRho, localize_bath = False, spin = 0, bath_index = None):
    import matplotlib.pyplot as plt
    from bcs import embBasis
    nscsites = lattice.supercell.nsites
    lat_coords = np.array(lattice.sites).T
    # make impurity at the center of lattice
    lat_coords[0][lat_coords[0] > LatSize[0] // 2] -= LatSize[0]
    lat_coords[1][lat_coords[1] > LatSize[1] // 2] -= LatSize[1]
    
    B = embBasis(lattice, GRho, localize_bath = localize_bath, return_bath = True)
    z = np.zeros((nscsites, nscsites*2))     
    if spin == 0:  
        bath_orbs = B[:, :nscsites,:].reshape((-1, 2*nscsites)) # alpha
    else:
        bath_orbs = B[:, nscsites:,:].reshape((-1, 2*nscsites)) # beta

    bath_orbs = np.vstack((z, bath_orbs))
   
    if bath_index is None:
        ave_orb = np.abs(bath_orbs).sum(axis = 1)
    else:
        ave_orb = np.abs(bath_orbs)[:, bath_index]

    plt.xlim(-LatSize[0] / 2 + 1.0, LatSize[0] / 2 + 1.0)
    plt.ylim(-LatSize[1] / 2 + 1.0, LatSize[1] / 2 + 1.0)
    plt.xticks(np.arange(-LatSize[0] / 2 , LatSize[0] / 2 + 1.0, 1.0))
    plt.yticks(np.arange(-LatSize[1] / 2 , LatSize[1] / 2 + 1.0, 1.0))
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.grid()
    bath_plot = plt.scatter(lat_coords[0], lat_coords[1], s = 500 * ave_orb)
    plt.show()

def test_ER_localize():
    np.random.seed(9)
    norbs = 10
    s = np.random.rand(norbs,norbs,norbs,norbs)
    s = s + np.swapaxes(s, 0, 1)
    s = s + np.swapaxes(s, 2, 3)
    s = s + np.transpose(s, (2, 3, 0, 1))
    loc = Localizer(s)
    loc.optimize()
    R = loc.coefs
    err = loc.Int2e - np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, s)
    log.check(np.allclose(err, 0), "Inconsistent coefficients and integrals,"
            " difference is %.2e", la.norm(err))

def test_visualize():
    import libdmet.dmet.HubbardBCS as dmet
    LatSize = (18, 18)
    ImpSize = (2, 2)
    U = 6.0 
    Filling = 0.8 / 2.0 
    Mu = U * Filling # initial guess of global Mu
    
    Lat = dmet.SquareLattice(*(LatSize + ImpSize))
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham)
    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand = 0.00)
    GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu, thrnelec = 1e-7)
    visualize_bath(Lat, LatSize, GRho, localize_bath = True, spin = 0, bath_index = None) 


if __name__ == "__main__":
    log.verbose = "DEBUG0"
    test_ER_localize()
    test_visualize()
