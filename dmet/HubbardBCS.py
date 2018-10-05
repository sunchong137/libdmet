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

def HartreeFockBogoliubov_full(Lat, v, filling, mu0, thrnelec = 1e-6):
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
    return rho, mu, res

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

def transformResults_new(GRhoEmb, E, lattice, basis, ImpHam, H_energy, last_dmu, Mu):
    nscsites = basis.shape[-2] / 2
    GRhoImp, Efrag, nelec = bcs.transformResults_new(GRhoEmb, E, lattice, \
            basis, ImpHam, H_energy, last_dmu, Mu)
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
    basis = bcs.embBasis(Lat, GRho, local = local, **kwargs)
    if matching:
        log.result("Rotate bath orbitals to match alpha and beta basis")
        nbasis = basis.shape[-1]
        if local:
            basis[:, :, :, nbasis/2:] = basisMatching(basis[:, :, :, nbasis/2:])
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

def get_tiled_vcor(vcor_small, imp_size_small, imp_size_big, rand = 0.0):
    import itertools as it
    from libdmet.system.lattice import SquareLattice

    Lat = SquareLattice(*(imp_size_big + imp_size_small))
    cell_dict = Lat.celldict 
    nscsites_big = np.prod(imp_size_big) 
    num_cells = len(Lat.cells)
    sites = np.array(list(it.product(*map(range, imp_size_big))))
    
    # compute idx of each small cell basis in the big cell
    cell_idx = [[] for i in xrange(num_cells)]
    for i, site_i in enumerate(sites):
        cell_idx[cell_dict[tuple(np.floor(site_i / imp_size_small).astype(np.int))]].append(i)
    
    # assign the vcor_small to correct place of vcor_big
    vcor_mat_small = vcor_small.get()
    vcor_mat_big = np.zeros((3, nscsites_big, nscsites_big), dtype = vcor_mat_small.dtype)
    vcor_mat_big[2] = (np.random.rand(nscsites_big, nscsites_big) - 0.5) * rand

    for i, cell_i in enumerate(cell_idx):
        idx = np.ix_(cell_i, cell_i)
        vcor_mat_big[0][idx] = vcor_mat_small[0]
        vcor_mat_big[1][idx] = vcor_mat_small[1]
        vcor_mat_big[2][idx] = vcor_mat_small[2]
    
    vcor_big = AFInitGuess(imp_size_big, 0.0, 0.5, rand = 0.0)
    vcor_big.assign(vcor_mat_big)

    return vcor_big

def restart_from_dmet_iter(vcor0, f_name = './dmet_iter.npy'):
    Mu, last_dmu, vcor_param = np.load(f_name)
    vcor0.update(vcor_param)
    return vcor0, Mu, last_dmu

def restart_from_hdf5():
    pass

def restart_mu_record():
    pass


addDiag = bcs.addDiag

FitVcor = bcs.FitVcorTwoStep

foldRho = bcs.foldRho

if __name__ == '__main__':
    imp_size_big = (2, 4)
    imp_size_small = (2, 2)
    np.set_printoptions(4, linewidth = 1000, suppress = True)
    vcor = AFInitGuess(imp_size_small, 8.0, 0.5, rand = 0.0)
    print get_tiled_vcor(vcor, imp_size_small, imp_size_big, rand = 0.001).get()
    #vcor_big = AFInitGuess(imp_size_big, 8.0, 0.5, rand = 0.001)
    #print vcor_big.get()

