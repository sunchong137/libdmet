from Hubbard import *
import Hubbard
from libdmet.routine import bcs
from libdmet.routine.mfd import HFB
from libdmet.routine.bcs_helper import mono_fit, extractRdm, transform_imp

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
    rhoA, rhoB, kappaBA = extractRdm(rho[0])
    log.result("Local density matrix (mean-field): alpha, beta and pairing"
            "\n%s\n%s\n%s", rhoA, rhoB, kappaBA.T)

    # present results
    return rho, mu

def ConstructImpHam(Lat, GRho, v, matching = True, local = True, **kwargs):
    log.result("Making embedding basis")
    basis = bcs.embBasis(Lat, GRho, local = local)
    if matching:
        log.result("Rotate bath orbitals to match alpha and beta basis")
        nscsites = Lat.supercell.nsites
        if local:
            basis[:, :, :, nscsites:] = basisMatching(basis[:, :, :, nscsites:])
        else:
            basis = basisMatching(basis)

    log.result("Constructing impurity Hamiltonian")
    ImpHam, (H1e, H0e) = bcs.embHam(Lat, basis, v, local = local, **kwargs)

    return ImpHam, (H1e, H0e), basis

def apply_dmu(lattice, ImpHam, basis, dmu):
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[-1]
    dmu = 0.1
    tempCD, tempCC, tempH0 = transform_imp(basis, lattice, dmu * np.eye(nscsites))
    ImpHam.H1["cd"] -= tempCD
    ImpHam.H1["cc"] -= tempCC
    ImpHam.H0 -= tempH0
    ImpHam.H0 += dmu * nbasis
    return ImpHam

Hubbard.apply_dmu = apply_dmu

def AFInitGuess(ImpSize, U, Filling, polar = None, rand = 0.01):
    return Hubbard.AFInitGuess(ImpSize, U, Filling, polar, True, rand)

