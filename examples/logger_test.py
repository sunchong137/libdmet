import libdmet.utils.logger as log
import numpy as np

log.verbose = log.Level["DEBUG0"]
log.clock = True

log.result("I'm result %20.12f", 42)
log.warning("I'm warning")
log.info("I'm info")
log.debug(0, "I'm debug level %d", 0)
log.debug(1, "I'm debug level %d, you cannot see me", 1)
log.error("I'm an error!")
log.fatal("I'm a fatal error!")
log.warning("%s", np.eye((3)))
