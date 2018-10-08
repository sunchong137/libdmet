import libdmet.utils.logger as log
import libdmet.dmet.Hubbard as dmet
import numpy as np
import numpy.linalg as la
from libdmet.solver.dmrgci import DmrgCI
"""
A script to generate FCIDUMP of Hubbard model without bath.

"""

log.verbose = "DEBUG0"
#np.set_printoptions(3, linewidth =1000)

LatSize = [2, 2]
ImpSize = LatSize
#ImpSize = [2, 2]
norb = np.product(LatSize)

U = 4.0
Filling = 1.0 / 2.0
nelec = int(Filling * norb * 2)
Filling = nelec / (norb * 2.0)

maxM = 400

LMO = True

nelec = int(2 * Filling * np.product(LatSize))
Filling = float(nelec) / (2 * np.product(LatSize))

print "U : ", U
print "Filling : ", Filling
print "lattice size (PBC) : ", LatSize
print 
print "norb (spatial) : ", norb
print "nelec : ", nelec
print 

Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)


vcor = dmet.AFInitGuess(ImpSize, U, Filling)
#vcor_param_zero = np.zeros_like(vcor.param)
#vcor.update(vcor_param_zero)

Mu = U * Filling
print "Solve meanfield for AFM rdm guess."
rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu)


# Full CI
ncas = norb
nelecas = nelec

block = dmet.impurity_solver.StackBlock(nproc = 1, nthread = 28, nnode = 1, \
        bcs = False, tol = 1e-6, maxM = maxM)


if LMO:
    #solver = DmrgCI(ncas, nelecas, MP2natorb = False, spinAverage = False, \
    #            splitloc = True, cisolver = block, mom_reorder = False, tmpDir = "./tmp")
    solver = DmrgCI(ncas, nelecas, MP2natorb = False, spinAverage = False, \
                splitloc = True, cisolver = block, mom_reorder = True, tmpDir = "./tmp")
else:
    solver = block

print "Construct Imp Hamiltonian"
ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor)
# only keep the impurity basis.
basis = basis[:, :, :, :basis.shape[2]]
ImpHam.norb /= 2
ImpHam.H1["cd"] = ImpHam.H1["cd"][:, :ImpHam.norb, :ImpHam.norb]
ImpHam.H2["ccdd"] = ImpHam.H2["ccdd"][:, :ImpHam.norb, :ImpHam.norb, :ImpHam.norb, :ImpHam.norb]


print "Solve by DMRG"
if LMO:
    #solver.run(ImpHam, ci_args = {}, guess = np.asarray((rho[0,0],rho[1,0])), nelec = nelecas, basis = None, similar = False)
    solver.run(ImpHam, ci_args = {}, guess = np.asarray((rho[0,0],rho[1,0])), nelec = nelecas, basis = basis, similar = False)
else:
    solver.run(ImpHam, nelec = nelecas)
