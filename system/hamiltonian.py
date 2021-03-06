import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log

class HamNonInt(object):
    def __init__(self, lattice, H1, H2, Fock = None, ImpJK = None, kspace_input = False):
        ncells = lattice.ncells
        nscsites = lattice.supercell.nsites
        log.eassert(H1.shape == (ncells, nscsites, nscsites), \
            "H1 shape not compatible with lattice")
        if kspace_input:
            self.H1 = lattice.FFTtoT(H1)
        else:
            self.H1 = H1
        if Fock is None:
            self.Fock = H1
        else:
            log.eassert(Fock.shape == H1.shape, "Fock shape not compatible with lattice")
            if kspace_input:
                self.Fock = lattice.FFTtoT(Fock)
            else:
                self.Fock = Fock
        if ImpJK is None:
            self.ImpJK = None
        else:
            log.eassert(ImpJK.shape == H1[0].shape, "ImpJK shape not compatible with supercell")
            self.ImpJK = ImpJK

        log.eassert(H2.shape == (nscsites,) * 4, "H2 shape not compatible with supercell")
        self.H2 = H2

    def getH1(self):
        return self.H1

    def getH2(self):
        return self.H2

    def getFock(self):
        return self.Fock

    def getImpJK(self):
        return self.ImpJK

def HubbardHamiltonian(lattice, U, tlist = [1.]):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    H1 = np.zeros((ncells, nscsites, nscsites))

    for order, t in enumerate(tlist):
        if abs(t) < 1e-7:
            continue
        log.eassert(order < len(lattice.neighborDist), \
                "%dth near neighbor distance unspecified in Lattice object", order+1)
        dis = lattice.neighborDist[order]
        log.warning("Searching neighbor within only one supercell")
        pairs = lattice.neighbor(dis = dis, sitesA = range(nscsites))
        for i, j in pairs:
            H1[j / nscsites, i, j % nscsites] = -t

    H2 = np.zeros((nscsites,) * 4)
    for s in range(nscsites):
        H2[s,s,s,s] = U

    return HamNonInt(lattice, H1, H2)

def HubbardDCA(lattice, U, tlist = [1.]):
    assert(len(tlist) < 3)
    from libdmet.utils import dca_transform

    cells = tuple(lattice.csize)
    scsites = tuple(lattice.supercell.csize)
    dim = lattice.dim
    H = []
    def vec1(d, v1, v2):
        idx = [0] * dim * 2
        idx[d] = v1
        idx[d+dim] = v2
        return tuple(idx)

    for d in range(dim):
        H.append((vec1(d, 0, 1), -tlist[0]))
        H.append((vec1(d, cells[d]-1, scsites[d]-1), -tlist[0]))

    if len(tlist) == 2:
        assert(dim == 2)
        H.append(((0, 0, 1, 1), tlist[1]))
        H.append(((0, cells[1]-1, 1, scsites[1]-1), tlist[1]))
        H.append(((cells[0]-1, 0, scsites[0]-1, 1), tlist[1]))
        H.append(((cells[0]-1, cells[1]-1, scsites[0]-1, scsites[1]-1), tlist[1]))

    H_DCA = dca_transform.transformHam(cells, scsites, H)

    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    H1 = np.zeros((ncells, nscsites, nscsites))

    for pos, val in H_DCA:
        cidx = lattice.cell_pos2idx(pos[:dim])
        spos = np.asarray(pos[dim:])
        for s in range(nscsites):
            s1 = lattice.supercell.sitedict[tuple((lattice.supercell.sites[s]+spos) % scsites)]
            H1[cidx, s, s1] = val

    H2 = np.zeros((nscsites,) * 4)
    for s in range(nscsites):
        H2[s,s,s,s] = U

    return HamNonInt(lattice, H1, H2)

def Hubbard3band(lattice, Ud, Up, ed, tpd, tpp, tpp1 = 0.):
    ncells = lattice.ncells
    nscsites = lattice.supercell.nsites
    H1 = np.zeros((ncells, nscsites, nscsites))
    H2 = np.zeros((nscsites,) * 4)
    d_pd = lattice.neighborDist[0]
    d_pp = lattice.neighborDist[1]
    d_pp1 = lattice.neighborDist[2]
    log.warning("Searching neighbor within only one supercell")

    def get_vec(s1, s2):
        vec = (lattice.sites[s1] - lattice.sites[s2]) % np.diag(lattice.size)
        for i in range(vec.shape[0]):
            if vec[i] > lattice.size[i,i] / 2:
                vec[i] -= lattice.size[i,i]
        return vec

    pd_pairs = lattice.neighbor(dis = d_pd, sitesA = range(nscsites))
    for i, j in pd_pairs:
        if lattice.names[i] == "Cu":
            vec = get_vec(j, i)
        else:
            vec = get_vec(i, j)

        if vec[0] == 1. or vec[1] == -1.:
            sign = -1.
        elif vec[1] == 1. or vec[0] == -1.:
            sign = 1.
        else:
            log.error("invalid p-d neighbor")

        H1[j / nscsites, i, j % nscsites] = sign * tpd

    pp_pairs = lattice.neighbor(dis = d_pp, sitesA = range(nscsites))
    for i, j in pp_pairs:
        vec = get_vec(j, i)
        if vec[0] * vec[1] > 0:
            sign = -1.
        else:
            sign = 1.
        H1[j / nscsites, i, j % nscsites] = sign * tpp

    Osites = [idx for (idx, name) in \
            zip(range(nscsites), lattice.names[:nscsites]) if name == "O"]
    pp1_pairs = lattice.neighbor(dis = d_pp1, sitesA = Osites)
    for i, j in pp1_pairs:
        H1[j / nscsites, i, j % nscsites] = -tpp1

    for i, orb in enumerate(lattice.supercell.names):
        if orb == "Cu":
            H1[0,i,i] = ed
            H2[i,i,i,i] = Ud
        elif orb == "O":
            H2[i,i,i,i] = Up
        else:
            log.error("wrong orbital name %s in 3-band Hubbard model", orb)
    return HamNonInt(lattice, H1, H2)
