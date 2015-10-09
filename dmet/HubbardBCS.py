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
    if filling is None:
        log.result("Local density matrix (mean-field): alpha, beta and pairing"
                "\n%s\n%s\n%s", rhoA, rhoB, kappaBA.T)
        nscsites = Lat.supercell.nsites
        log.result("nelec per site (mean-field) = %20.12f", n/nscsites)
        log.result("Energy per site (mean-field) = %20.12f", E/nscsites)
        log.result("Gap (mean-field) = %20.12f" % res["gap"])

    # present results
    return rho, mu

def transformResults(GRhoEmb, E, lattice, basis, ImpHam, H_energy, dmu):
    nscsites = basis.shape[-2] / 2
    GRhoImp, Efrag, nelec = bcs.transformResults(GRhoEmb, E, lattice, \
            basis, ImpHam, H_energy, dmu)
    log.debug(1, "impurity generalized density matrix:\n%s", GRhoImp)

    if Efrag is None:
        return nelec/nscsites
    else:
        log.result("Local density matrix (impurity): alpha, beta and pairing")
        rhoA, rhoB, kappaBA = extractRdm(GRhoImp)
        log.result("%s", rhoA)
        log.result("%s", rhoB)
        log.result("%s", -kappaBA.T)
        log.result("nelec per site (impurity) = %20.12f", nelec/nscsites)
        log.result("Energy per site (impurity) = %20.12f", Efrag/nscsites)

        return GRhoImp, Efrag/nscsites, nelec/nscsites

Hubbard.transformResults = lambda GRhoEmb, E, basis, ImpHam, H_energy: \
      transformResults(GRhoEmb, E, None, basis, ImpHam, H_energy, 0.)

def ConstructImpHam(Lat, GRho, v, mu, matching = True, local = True, **kwargs):
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
    ImpHam, (H1e, H0e) = bcs.embHam(Lat, basis, v, mu, local = local, **kwargs)

    return ImpHam, (H1e, H0e), basis

def apply_dmu(lattice, ImpHam, basis, dmu):
    nscsites = lattice.supercell.nsites
    nbasis = basis.shape[-1]
    tempCD, tempCC, tempH0 = transform_imp(basis, lattice, dmu * np.eye(nscsites))
    ImpHam.H1["cd"] -= tempCD
    ImpHam.H1["cc"] -= tempCC
    ImpHam.H0 -= tempH0
    return ImpHam

Hubbard.apply_dmu = apply_dmu

def AFInitGuess(ImpSize, U, Filling, polar = None, rand = 0.01):
    return Hubbard.AFInitGuess(ImpSize, U, Filling, polar, True, rand)

addDiag = bcs.addDiag

FitVcor = bcs.FitVcorTwoStep

foldRho = bcs.foldRho
