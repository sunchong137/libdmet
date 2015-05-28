import libdmet.system.lattice as lat
import libdmet.utils.logger as log
import libdmet.utils.iotools as io
import numpy as np

a = 3.869
c = 7.525

size = np.asarray([[a, 0, 0],[0, a, 0], [0.5*a, 0.5*a, c]])

frac = False

atoms = [
    (np.asarray([0.5*a, 0.5*a, 1.568]), "Ca"),
    (np.asarray([0.5*a, 0.5*a, -1.568]), "Ca"),
    (np.asarray([0, 0, 0]), "Cu"),
    (np.asarray([0.5*a, 0, 0]), "O"),
    (np.asarray([0, 0.5*a, 0]), "O"),
    (np.asarray([0, 0, 2.754]), "Cl"),
    (np.asarray([0, 0, -2.754]), "Cl"),
]

center = np.asarray([0, 0, 0])

scsize = np.asarray([2, 2, 1])

log.result("dump to POSCAR")
io.struct_dump(size, scsize, atoms, "POSCAR", \
        center = center, elements = ["Cu", "O", "Ca", "Cl"])
log.result("dump to XYZ")
io.struct_dump(size, scsize, atoms, "xyz", \
        center = center, elements = ["Cu", "O", "Ca", "Cl"])

