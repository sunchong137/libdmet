import numpy as np
import numpy.linalg as la
from libdmet.system import integral
from libdmet.solver import block

Ham = integral.readFCIDUMP("bcs_pdm/DMETDUMP", 8, False, True)

blk = block.Block()
blk.tmpDir = "/Users/zhengbx/dev/libdmet/block/dmrg_tests/bcs_pdm"
blk.set_system(None, 0, False, True, False)
blk.set_integral(Ham)
blk.optimized = True
GRho = blk.onepdm()
gamma0, gamma2, gamma4 = blk.twopdm(computed = True)
rho = np.asarray([GRho[:8, :8], np.eye(8) - GRho[8:, 8:]])
kappaBA = GRho[8:, :8]

E = Ham.H0 + np.sum(Ham.H1["cd"] * rho) + np.sum(Ham.H1["cc"] * kappaBA.T) * 2 + \
        np.sum(Ham.H2["ccdd"][:2] * gamma0[:2]) * 0.5 + np.sum(Ham.H2["ccdd"][2] * gamma0[2]) + \
        1. * np.sum(Ham.H2["cccd"] * gamma2) + 0.5 * np.sum(Ham.H2["cccc"] * gamma4)

print E

gam0 = np.asarray([
    np.einsum("il,jk->iljk", rho[0], rho[0]) - np.einsum("ik,jl->iljk", rho[0], rho[0]),
    np.einsum("il,jk->iljk", rho[1], rho[1]) - np.einsum("ik,jl->iljk", rho[1], rho[1]),
    np.einsum("il,jk->iljk", rho[0], rho[1]) + np.einsum("ij,lk->iljk", kappaBA.T, kappaBA.T)
])

gam2 = np.asarray([
    np.einsum("il,jk->ijkl", rho[0], kappaBA.T) - np.einsum("ik,jl->ijkl", kappaBA.T, rho[0]),
    np.einsum("il,kj->ijkl", rho[1], -kappaBA.T) - np.einsum("ki,jl->ijkl", -kappaBA.T, rho[1])
])

gam4 = np.einsum("il,jk->ijkl", kappaBA.T, kappaBA.T) - np.einsum("ik,jl->ijkl", kappaBA.T, kappaBA.T)[np.newaxis]

print la.norm(gamma0[0] - gam0[0])
print la.norm(gamma0[1] - gam0[1])
print la.norm(gamma0[2] - gam0[2])
print la.norm(gamma2[0] - gam2[0])
print la.norm(gamma2[1] - gam2[1])
print la.norm(gamma4[0] - gam4[0])
