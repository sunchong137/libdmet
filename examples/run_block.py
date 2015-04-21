import numpy as np
from libdmet.solver import block
import libdmet.utils.logger as log
from libdmet.system import integral

log.clock = True
log.verbose = "INFO"

block.Block.nproc = 4

# a Hubbard example
solver = block.Block()
solver.set_system(8, 0, False, False, True)

Int1e = -np.eye(8, k = 1)
Int1e[0, 7] = -1
Int1e += Int1e.T
Int2e = np.zeros((8,8,8,8))

for i in range(8):
    Int2e[i,i,i,i] = 6

solver.set_integral(8, 0, {"cd": Int1e}, {"ccdd": Int2e})

scheduler = block.Schedule()
scheduler.gen_initial(minM = 50, maxM = 400)

solver.set_schedule(scheduler)

solver.optimize()

log.result("E = %20.12f", solver.evaluate(0, {"cd": Int1e}, {"ccdd": Int2e}, op = "Hamiltonian"))

for i in range(8):
    Int2e[i,i,i,i] = 8

solver.integral.H2["ccdd"] = Int2e

log.result("E = %20.12f", solver.restart_optimize(onepdm = False, M = 600)[1])

solver.cleanup()

# a BCS example

solver.createTmp()

solver.set_system(16, 0, False, True, False)
solver.integral = integral.read("../block/dmrg_tests/bcs/DMETDUMP", \
    8, False, True, "FCIDUMP")

scheduler.gen_initial(minM = 100, maxM = 400)

solver.set_schedule(scheduler)

w, E, _ = solver.optimize(onepdm = False)
log.result("E = %20.12f", E)

solver.cleanup()
