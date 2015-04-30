from Hubbard import *
import numpy as np
import numpy.linalg as la
from libdmet.system.hamiltonian import HamNonInt
import libdmet.system.lattice as Lat
import os

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
