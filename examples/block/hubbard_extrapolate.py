import numpy as np
from libdmet.solver import block
import libdmet.utils.logger as log
from libdmet.system import integral

log.verbose = "INFO"

block.Block.set_nproc(4)

solver = block.Block()
solver.set_system(18, 0, False, False, True)

Int1e = -np.eye(20, k = 1)
Int1e -= np.eye(20, k = 2) * 0.3
Int1e -= -np.eye(20, k = 3) * 0.6
Int1e[0, 19] = -1
Int1e += Int1e.T
Int2e = np.zeros((20,20,20,20))

for i in range(20):
    Int2e[i,i,i,i] = 4
for i in range(19):
    Int2e[i,i,i+1,i+1] = 3
    Int2e[i+1,i+1,i,i] = 3
for i in range(18):
    Int2e[i,i,i+2,i+2] = 2
    Int2e[i+2,i+2,i,i] = 2
    Int2e[i,i+2,i,i+2] = -1
    Int2e[i+2,i,i+2,i] = -1
    Int2e[i+2,i,i,i+2] = -1
    Int2e[i,i+2,i+2,i] = -1

solver.set_integral(20, 0, {"cd": Int1e}, {"ccdd": Int2e})

solver.extrapolate([40,50,60,70])

solver.cleanup()

