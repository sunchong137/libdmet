from Hubbard import *
import Hubbard
import numpy as np
import numpy.linalg as la
from libdmet.system.hamiltonian import HamNonInt
import libdmet.system.lattice as Lat
import libdmet.system.integral as integral
import os
from libdmet.solver import scf, casscf
from libdmet.routine.slater_helper import transform_trans_inv_sparse
from libdmet.utils.misc import mdot

scfsolver = scf.SCF()

def buildUnitCell(size, atoms, basis):
    sites = []
    count = {}
    for a in atoms:
        coord, name = a
        if not name in count.keys():
            count[name] = 1
        else:
            count[name] += 1
        for b in basis[name]:
            sites.append((coord, name + "%d_" % count[name] + b))
    return Lat.UnitCell(size, sites)

def buildLattice(latSize, impSize, cellSize, atoms, basis):
    log.eassert(np.allclose(latSize % impSize, 0), "incompatible lattice size and supercell size")
    uc = buildUnitCell(cellSize, atoms, basis)
    sc = Lat.SuperCell(uc, impSize)
    lat = Lat.Lattice(sc, latSize / impSize)
    return lat

def __read_bin(dirname, name, shape):
    if os.path.exists(os.path.join(dirname, name + ".npy")):
        temp = np.load(os.path.join(dirname, name + ".npy"))
        log.eassert(temp.shape == shape,
            "when reading integral, the required shape is %s, but get %s", shape, temp.shape)
    elif os.path.exists(os.path.join(dirname, name + ".mmap")):
        temp = np.memmap(os.path.join(dirname, name + ".mmap"),\
            dtype = "float", mode = "c", shape = shape)
    else:
        log.error("couldn't find the integral file %s", name)
    return temp

def read_integral(dirname, lattice, cutoff):
    log.info("reading integrals from %s", os.path.realpath(dirname))
    nscsites = lattice.supercell.nsites
    nonzero = map(np.asarray, list(it.product(range(cutoff), repeat = lattice.dim)))
    dirname = os.path.realpath(dirname)
    if cutoff is None:
        nnonzero = lattice.ncells
    else:
        nnonzero = len(nonzero)
    H1 = __read_bin(dirname, "H1", (nnonzero, nscsites, nscsites))
    H2 = __read_bin(dirname, "H2", (nscsites,)*4)
    Fock = __read_bin(dirname, "Fock", (nnonzero, nscsites, nscsites))
    ImpJK = __read_bin(dirname, "ImpJK", (nscsites, nscsites))

    if cutoff is not None and nnonzero < lattice.ncells:
        FockFull = np.zeros((lattice.ncells, nscsites, nscsites))
        H1Full = np.zeros((lattice.ncells, nscsites, nscsites))
        for i, n in enumerate(nonzero):
            FockFull[lattice.cell_pos2idx(n)] = Fock[i]
            FockFull[lattice.cell_pos2idx(-n)] = Fock[i].T
            H1Full[lattice.cell_pos2idx(n)] = H1[i]
            H1Full[lattice.cell_pos2idx(-n)] = H1[i].T
        Fock, H1 = FockFull, H1Full
    return [H1, H2, Fock, ImpJK]

def buildHamiltonian(dirname, lattice, cutoff):
    return HamNonInt(lattice, *(read_integral(dirname, lattice, cutoff)))

def AFInitGuessOrbs(v, lattice, AForbs, PMorbs, shift = 0., polar = 0.5):
    names = lattice.supercell.names
    nscsites = lattice.supercell.nsites
    subA = map(names.index, AForbs[0])
    subB = map(names.index, AForbs[1])
    subC = map(names.index, PMorbs)
    vguess = np.zeros((2, nscsites, nscsites))
    for site in subA:
        vguess[0, site, site] = shift + polar
        vguess[1, site, site] = shift - polar
    for site in subB:
        vguess[0, site, site] = shift - polar
        vguess[1, site, site] = shift + polar
    for site in subC:
        vguess[0, site, site] = vguess[1, site, site] = shift

    # FIXME a hack, directly update the parameters
    p = np.zeros(v.length())
    psite = lambda site: (2*nscsites+1-site)*site/2
    for site in subA:
        p[psite(site)] = shift + polar
        p[psite(site) + psite(nscsites)] = shift - polar
    for site in subB:
        p[psite(site)] = shift - polar
        p[psite(site) + psite(nscsites)] = shift + polar
    for site in subC:
        p[psite(site)] = p[psite(site) + psite(nscsites)] = shift

    v.update(p)
    log.eassert(la.norm(v.get() - vguess) < 1e-10, "initial guess cannot be assgned directly")
    return v

def reportOccupation(lattice, rho, names = None):
    rhoImp = map(np.diag, rho)
    charge = (rhoImp[0] + rhoImp[1]) / 2
    spin = (rhoImp[0] - rhoImp[1]) / 2
    nscsites = lattice.supercell.nsites
    if names is None:
        names = lattice.supercell.names
        indices = range(nscsites)
    else:
        indices = map(lattice.supercell.index, names)
    
    results = []
    lines = ["%-3s   ", "charge", "spin  "]
    atom = names[0].split("_")[0]
    lines[0] = lines[0] % atom
    totalc, totals = 0., 0.
    for i, (name, index) in enumerate(zip(names, indices)):
        if atom != name.split("_")[0]:
            lines[0] += "%10s" % "total"
            lines[1] += "%10.5f" % totalc
            lines[2] += "%10.5f" % totals
            totalc, totals = 0., 0.
            results.append("\n".join(lines))
            lines = ["%-3s   ", "charge", "spin  "]
            atom = name.split("_")[0]
            lines[0] = lines[0] % atom            

        lines[0] += "%10s" % name.split("_")[1]
        lines[1] += "%10.5f" % charge[index]
        lines[2] += "%10.5f" % spin[index]
        totalc += charge[index]
        totals += spin[index]

    lines[0] += "%10s" % "total"
    lines[1] += "%10.5f" % totalc
    lines[2] += "%10.5f" % totals
    results.append("\n".join(lines))
    log.result("\n".join(results))

def selectActiveSpace(rho, thrRdm, nact = None):
    if rho.shape[0] == 2:
        rho_mix = 0.5 * (rho[0] + rho[1])
    else:
        rho_mix = rho
    natocc, natorb = la.eigh(rho_mix)
    log.debug(0, "MP2 natural orbital occupation:\n%s", natocc)
    ncore = np.sum(natocc > 1 - thrRdm)
    nvirt = np.sum(natocc < thrRdm)
    nactive = rho.shape[1] - ncore - nvirt
    if nact is not None and nactive != nact:
        log.warning("Imposing number of active orbitals: Using %d rather than %d", nact, nactive)
        ncore += (nactive - nact) / 2
        nactive = nact
        nvirt = rho.shape[1] - ncore - nactive
    log.info("Ncore = %d  Nvirt = %d  Nactive = %d", ncore, nvirt, nactive)
    return natorb[:, -ncore:], natorb[:, nvirt: -ncore] # core and active orbitals

def buildActiveHam(Ham, c, a):
    spin = Ham.H1["cd"].shape[0]
    if spin == 1:
        log.error("Active space Hamiltonian with restricted Hamiltonian not implemented yet")

    cRdm = np.dot(c, c.T)
    # core-core one-body
    H0 = np.sum(cRdm * (Ham.H1["cd"][0] + Ham.H1["cd"][1]))
    # core-fock
    vj00 = np.tensordot(cRdm, Ham.H2["ccdd"][0], ((0,1), (0,1)))
    vj11 = np.tensordot(cRdm, Ham.H2["ccdd"][1], ((0,1), (0,1)))
    vj10 = np.tensordot(cRdm, Ham.H2["ccdd"][2], ((0,1), (0,1)))
    vj01 = np.tensordot(cRdm, Ham.H2["ccdd"][2], ((1,0), (3,2)))
    vk00 = np.tensordot(cRdm, Ham.H2["ccdd"][0], ((0,1), (0,3)))
    vk11 = np.tensordot(cRdm, Ham.H2["ccdd"][1], ((0,1), (0,3)))
    v = np.asarray([vj00 + vj01 - vk00, vj11 + vj10 - vk11])
    # core-core two-body
    H0 += 0.5 * np.sum(cRdm * (v[0] + v[1]))
    # active one-body: bare + effective from core Fock
    H1 = {
        "cd": np.asarray(map(lambda s: mdot(a.T, v[s] + Ham.H1["cd"][s], a), range(spin)))
    }
    # transform active two-body part
    aSO = np.asarray((a, a))
    H2 = {
        "ccdd": scf.incore_transform(Ham.H2["ccdd"], (aSO, aSO, aSO, aSO))
    }
    return integral.Integral(a.shape[1], False, False, H0, H1, H2)

def __SolveImpHam_with_dmu(lattice, ImpHam, basis, M, dmu, rhoNonInt = None, nelec = None, nact = None, thrRdm = 5e-3):
    # H = H1 + Vcor - Mu
    # to keep H for mean-field Mu->Mu+dMu, Vcor->Vcor+dMu
    # In impurity Ham, equivalent to substracting dMu from impurity, but not bath
    # The evaluation of energy is not affected if using (corrected) ImpHam-dMu
    # alternatively, we can change ImpHam.H0 to compensate
    ImpHam = __apply_dmu(lattice, ImpHam, basis, dmu)
    result = SolveImpHamCASCI(ImpHam, M, lattice, basis, rhoNonInt, nelec, nact, thrRdm)
    ImpHam = __apply_dmu(lattice, ImpHam, basis, -dmu)    
    return result
 
Hubbard.__SolveImpHam_with_dmu = __SolveImpHam_with_dmu

def SolveImpHamCASCI(ImpHam, M, Lat, basis, rhoNonInt, nelec = None, nact = None, thrRdm = 5e-3, MP2 = True):
    spin = ImpHam.H1["cd"].shape[0]
    if nelec is None:
        nelec = ImpHam.norb
    scfsolver.set_system(nelec, 0, False, spin == 1)
    scfsolver.set_integral(ImpHam)
    # using non-Interacting density matrix as initial guess
    rhoHF = np.asarray(map(lambda s: transform_trans_inv_sparse(basis[s], Lat, rhoNonInt[s]), range(spin)))
    # do Hartree-Fock
    E_HF, rhoHF = scfsolver.HF(tol = 1e-5, MaxIter = 20, InitGuess = rhoHF)
    nscsites = Lat.supercell.nsites
    reportOccupation(Lat, rhoHF[:, :nscsites, :nscsites])
    # then MP2
    if MP2:
        E_MP2, rhoMP2 = scfsolver.MP2()
        log.result("MP2 energy = %20.12f", E_HF + E_MP2)
        reportOccupation(Lat, rhoMP2[:, :nscsites, :nscsites])
        rho_guess = rhoMP2
    else:
        rho_guess = rhoHF
    # define active space
    log.info("Setting up active space")
    if solver.optimized:
        core, active = selectActiveSpace(rho_guess, thrRdm, nact = solver.integral.norb)
    elif nact is not None:
        core, active = selectActiveSpace(rho_guess, thrRdm, nact = nact)
    else:
        core, active = selectActiveSpace(rho_guess, thrRdm)
    nelec_active = nelec - core.shape[1] * 2
    log.info("Number of electrons in active space: %d", nelec_active)
    # FIXME additional localization for active?
    # two ways: 1. location-based localization 2. integral based localization
    # build active space Hamiltonian
    log.debug(0, "build active space Hamiltonian")
    actHam = buildActiveHam(ImpHam, core, active)
    # solve active space Hamiltonian
    log.debug(0, "solve active space Hamiltonian with BLOCK")
    actRho, E = SolveImpHam(actHam, M, nelec = nelec_active)
    coreRho = np.dot(core, core.T)
    rho = np.asarray(map(lambda s: mdot(active, actRho[s], active.T) + coreRho, range(spin)))
    reportOccupation(Lat, rho[:, :nscsites, :nscsites])
    return rho, E

def SolveImpHamCASSCF(ImpHam, M, Lat, basis, rhoNonInt, nelec = None, nact = None, thrRdm = 1e-2):
    spin = ImpHam.H1["cd"].shape[0]
    if nelec is None:
        nelec = ImpHam.norb
    scfsolver.set_system(nelec, 0, False, spin == 1)
    scfsolver.set_integral(ImpHam)
    # using non-Interacting density matrix as initial guess
    rhoHF = np.asarray(map(lambda s: transform_trans_inv_sparse(basis[s], Lat, rhoNonInt[s]), range(spin)))
    # do Hartree-Fock
    E_HF, rhoHF = scfsolver.HF(tol = 1e-5, MaxIter = 20, InitGuess = rhoHF)
    nscsites = Lat.supercell.nsites
    reportOccupation(Lat, rhoHF[:, :nscsites, :nscsites])
    # define active space
    log.info("Setting up active space")
    if nact is not None:
        core, active = selectActiveSpace(rho_guess, thrRdm, nact = nact)
    else:
        core, active = selectActiveSpace(rho_guess, thrRdm)
        nact = active.shape[1]
    nelec_active = nelec - core.shape[1] * 2
    CASsolver = casscf.CASSCF(scfsolver.mf, nact, (nelec_active, nelec_active))
    E_CAS = CASsolver.mc1step()[0]
    rho = np.asarray(CASsolver.make_rdm1s())
    log.result("Energy (CASSCF) = %20.12f", E_CAS)
    reportOccupation(Lat, rho[:, :nscsites, :nscsites])
    return rho, E_CAS
