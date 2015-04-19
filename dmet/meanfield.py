# only implement non-scf routines
# restricted/unrestricted
# Slater/BCS
# thermal occupation

import numpy as np
import numpy.linalg as la
import itertools as it
import libdmet.utils.logger as log

def SimpleDiag(H1, vcor, lattice):
    # assume H1 is in reduced form
    H1prime = np.empty_like(H1)
    ncells = H1.shape[0]
    for i in range(ncells):
        H1prime = H1[i] + vcor(i, False)
    H1big = lattice.expand(H1prime)
    ew, ev = la.eigh(H1big)
    return ew, ev

def SimpleDiagK(H1, vcor):
    ncells = H1.shape[0]
    nscsites = H1.shape[1]
    ew = np.empty((ncells, nscsites))
    ev = np.empty((ncells, nscsites, nscsites))

    for i in range(ncells):
        ew[i], ev[i] = la.eigh(H1[i] + vcor(i, True))
    return ew, ev
