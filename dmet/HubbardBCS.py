from Hubbard import *
import Hubbard
from libdmet.routine.mfd import HFB
from libdmet.routine.bcs_helper import mono_fit, extractRdm

def HartreeFockBogoliubov(Lat, v, filling, mu0, thrnelec = 1e-6):
    # fit chemical potential
    if filling is None:
        mu = mu0
    else:
        # fit mu to get correct filling
        log.info("chemical potential fitting, target = %20.12f", filling)
        log.info("before fitting, mu = %20.12f", mu0)
        fn = lambda mu: HFB(Lat, v, False, mu = mu, beta = np.inf, \
                ires = False)[1] / 2. / Lat.supercell.nsites
        mu = mono_fit(fn, filling, mu0, thrnelec, increase = True)
        log.info("after fitting, mu = %20.12f", mu)
    rho, n, E, res = HFB(Lat, v, False, mu = mu, beta = np.inf, \
            ires = True)
    rhoA, rhoB, kappa = extractRdm(rho[0])
    log.result("Local density matrix (mean-field): alpha, beta and pairing"
            "\n%s\n%s\n%s", rhoA, rhoB, kappa)

    # present results
    return rho, mu

def AFInitGuess(ImpSize, U, Filling, polar = None, rand = 0.01):
    return Hubbard.AFInitGuess(ImpSize, U, Filling, polar, True, rand)
