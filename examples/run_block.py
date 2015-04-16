import numpy as np
import libdmet.solver.block_iface as block
import libdmet.utils.logger as log

log.clock = True
log.verbose = log.Level["INFO"]

block.Block.nproc = 4
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

log.result("E = %20.12f", solver.evaluate(0, {"cd": Int1e}, {"ccdd": Int2e}))

solver.cleanup()
