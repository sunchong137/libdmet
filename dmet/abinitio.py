from Hubbard import *
import numpy as np
import numpy.linalg as la
from libdmet.system.hamiltonian import HamNonInt
import libdmet.system.lattice as Lat
import libdmet.system.integral as integral
import os
from libdmet.solver import scf
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

def ConstructImpHam(Lat, rho, v, matching = True, **kwargs):
    log.result("Making embedding basis")
    basis = slater.embBasis(Lat, rho, local = True)

    if matching and basis.shape[0] == 2: 
        log.result("Rotate bath orbitals to match alpha and beta basis")
        nscsites = Lat.supercell.nsites
        bathA = basis[0, :, :, nscsites:]
        bathB = basis[1, :, :, nscsites:]
        S = np.tensordot(bathA, bathB, axes = ((0,1), (0,1)))
        # S=A^T*B svd of S is UGV^T then we let A'=AU, B'=BV
        # yields A'^T*B'=G diagonal and optimally overlapped
        u, gamma, vt = la.svd(S)
        log.result("overlap statistics:\n larger than 0.9: %3d  smaller than 0.9: %3d\n"
            " average: %10.6f  min: %10.6f", \
            np.sum(gamma > 0.9), np.sum(gamma < 0.9), np.average(gamma), np.min(gamma))
        basis[0, :, :, nscsites:] = np.tensordot(bathA, u, axes = (2, 0))
        basis[1, :, :, nscsites:] = np.tensordot(bathB, vt, axes = (2, 1)) # because of V.T

    log.result("Constructing impurity Hamiltonian")
    ImpHam, H1e = slater.embHam(Lat, basis, v, local = True, **kwargs)

    return ImpHam, H1e, basis

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
    rhoImp = map(np.diag, rho[:,0])
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
    for i, (name, index) in enumerate(zip(names, indices)):
        if atom != name.split("_")[0]:
            results.append("\n".join(lines))
            lines = ["%-3s   ", "charge", "spin  "]
            atom = name.split("_")[0]
            lines[0] = lines[0] % atom            

        lines[0] += "%10s" % name.split("_")[1]
        lines[1] += "%10.5f" % charge[index]
        lines[2] += "%10.5f" % spin[index]

    results.append("\n".join(lines))
    log.result("\n".join(results))

def selectActiveSpace(rho, thrRdm):
    if rho.shape[0] == 2:
        rho_mix = 0.5 * (rho[0] + rho[1])
    else:
        rho_mix = rho
    natocc, natorb = la.eigh(rho_mix)
    ncore = np.sum(natocc > 1 - thrRdm)
    nvirt = np.sum(natocc < thrRdm)
    nactive = rho.shape[1] - ncore - nvirt
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

def SolveImpCAS(ImpHam, M, Lat, basis, rhoNonInt, nelec = None, thrRdm = 5e-3):
    spin = ImpHam.H1["cd"].shape[0]
    if nelec is None:
        nelec = ImpHam.norb
    scfsolver.set_system(nelec, 0, False, spin == 1)
    scfsolver.set_integral(ImpHam)
    # using non-Interacting density matrix as initial guess
    rhoHF = np.asarray(map(lambda s: transform_trans_inv_sparse(basis[s], Lat, rhoNonInt[s]), range(spin)))
    # do Hartree-Fock
    E_HF, rhoHF = scfsolver.HF(tol = 1e-5, MaxIter = 20, InitGuess = rhoHF)
    # then MP2
    E_MP2, rhoMP2 = scfsolver.MP2()
    log.info("Setting up active space")
    core, active = selectActiveSpace(rhoMP2, thrRdm)
    # FIXME additional localization for active?
    # two ways: 1. location-based localization 2. integral based localization
    # build active space Hamiltonian
    log.debug(0, "build active space Hamiltonian")
    actHam = buildActiveHam(ImpHam, core, active)
    # solve active space Hamiltonian
    log.debug(0, "solve active space Hamiltonian with BLOCK")
    actRdm, E = SolveImpHam(actHam, M, nelec = nelec - core.shape[1] * 2)
    coreRdm = np.dot(core, core.T)
    rdm = np.asarray(map(lambda s: mdot(active, actRdm[s], active.T) + coreRdm, range(spin)))
    return rdm, E

