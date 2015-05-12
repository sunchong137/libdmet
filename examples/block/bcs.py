import numpy as np
from libdmet.solver import block
import libdmet.utils.logger as log
from libdmet.system import integral

log.verbose = "INFO"

block.Block.set_nproc(4)

solver = block.Block()
solver.createTmp()
solver.set_system(16, 0, False, True, False)

solver.integral = integral.read("../../block/dmrg_tests/bcs/DMETDUMP", \
    8, False, True, "FCIDUMP")
solver.integral_initialized = True

scheduler = block.Schedule()
scheduler.gen_initial(minM = 100, maxM = 400)
solver.set_schedule(scheduler)

w, E, _ = solver.optimize(onepdm = False)
log.result("E = %20.12f", E)

solver.cleanup()
