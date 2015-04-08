import numpy as np
from os.path import * 
from tempfile import mkdtemp
import libdmet.utils.logger as log
from libdmet.utils import integral
from copy import deepcopy

log.clock = True
log.verbose = log.Level["DEBUG0"]

class Schedule(object):
    def __init__(self, maxiter = 30, sweeptol = 1e-8):
        self.initialized = False
        self.twodot_to_onedot = None
        self.maxiter = maxiter
        self.sweeptol = sweeptol

    def initial(self, minM, maxM):
        defaultM = [100, 250, 400, 800, 1500, 2500]
        log.debug(1, "Generate default schedule with startM = %d maxM = %d, maxiter = %d", \
            minM, maxM, self.maxiter)

        self.arrayM = [minM] + [M for M in defaultM if M > minM and M < maxM] + [maxM]
        self.arraySweep = range(0, 6 * len(self.arrayM), 6)
        self.arrayTol = [self.sweeptol * 0.1 * 10.**i for i in range(len(self.arrayM))][::-1]
        self.arrayNoise = deepcopy(self.arrayTol)

        self.arrayM.append(maxM)
        self.arraySweep.append(self.arraySweep[-1] + 2)
        self.arrayTol.append(self.arrayTol[-1])
        self.arrayNoise.append(0)

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)  
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = self.arraySweep[-1] + 2

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        if self.twodot_to_onedot + 4 > self.maxiter:
            log.warning("only %d onedot iterations\nmodify maxiter to %d", \
                self.maxiter - self.twodot_to_onedot, self.twodot_to_onedot + 4)
            self.maxiter = self.twodot_to_onedot + 4
        self.initialized = True

    def restart(self, M):
        log.debug(1, "Generate default schedule with restart calculation M = %d, maxiter = %d", M, self.maxiter)
        self.arrayM = [M, M]
        self.arraySweep = [0, 2]
        self.arrayTol = [self.sweeptol * 0.1] * 2
        self.arrayNoise = [self.sweeptol * 0.1, 0]
        
        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)  
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = self.arraySweep[-1] + 2
        
        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)
        
        if self.twodot_to_onedot + 4 > self.maxiter:
            log.warning("only %d onedot iterations\nmodify maxiter to %d", \
                self.maxiter - self.twodot_to_onedot, self.twodot_to_onedot + 4)
            self.maxiter = self.twodot_to_onedot + 4
        self.initialized = True

    def extrapolate(self, M):
        log.debug(1, "Generate default schedule for truncation error extrapolation M = %d", M)
        self.arrayM = [M]
        self.arraySweep = [0]
        self.arrayTol = [self.sweeptol * 0.1]
        self.arrayNoise = [0]
        
        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)  
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = 2
        self.maxiter = 2

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        self.initialized = True

    def custom(self, arrayM, arraySweep, arrayTol, arrayNoise, twodot_to_onedot = None):
        log.debug(1, "Generate custom schedule")
        nstep = len(arrayM)
        if len(arraySweep) != nstep or len(arrayTol) != nstep or len(arrayNoise) != nstep:
            log.error("The lengths of input arrays are not consistent.")

        self.arrayM, self.arraySweep, self.arrayTol, self.arrayNoise = \
            arrayM, arraySweep, arrayTol, arrayNoise
        
        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)  
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        if twodot_to_onedot is None:
            self.twodot_to_onedot = self.arraySweep[-1] + 2
        else:
            self.twodot_to_onedot = twodot_to_onedot

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        if self.arraySweep[-1]+2 > self.maxiter:
            log.warning("maxiter smaller than scheduled number of sweeps\nmodify maxiter to %d", \
                self.arraySweep[-1]+2)
            self.maxiter = self.arraySweep[-1]+2
        self.initialized = True

    def get_schedule(self):
        if not self.initialized:
            log.error("DMRG schedule has not been generated.")
            raise Exception
        else:
            text = ["", "schedule"]
            nstep = len(self.arrayM)
            text += map(lambda n: "%d %d %.0e %.0e" % \
                (self.arrayM[n], self.arraySweep[n], self.arrayTol[n], self.arrayNoise[n]), range(nstep))
            text.append("end")
            text.append("")
            text.append("maxiter %d" % self.maxiter)
            text.append("twodot_to_onedot %d" % self.twodot_to_onedot)
            text.append("sweep_tol %.0e" % self.sweeptol)
            text.append("")            
            text = "\n".join(text)

            log.debug(2, "Generated schedule in configuration file")
            log.debug(1, text)
            

class Block(object):
    """
    Interface to Block dmrg code

    - take particle number conserving/non-conserving, spin-restricted/unrestricted Hamiltonian    
    - specify sequence of M or only min/max M
    - specify number of iterations or use default
    - specify twodot_to_onedot or use default
    - specify energy tolerance
    - compute 1pdm in patch    
    
    TODO:
    - optimize the wavefunction
    - compute other expectation values one at a time
    - specify reorder or noreorder
    - specify the outputlevel for DMRG itself, and for my output
    - dry run: just generate input files
    - error handling
    - set whether to restart, restart folder, whether or not delete it
    - set temp folder
    - set number of processors, number of nodes
    - back sweep and extrapolation to M=\inf limit
    """

    block_path = realpath(join(dirname(realpath(__file__)), "../block"))
    nproc = 1
    nnode = 1
    
    def __init__(self):
        self.sys_initialized = False
        self.schedule_initialized = False
        self.integral_initialized = False
    
        self.warmup_method = "local_2site"
        self.onepdm = True
        self.outputlevel = 0
        self.restart = False

    def set_system(self, nelec, spin, spinAdapted, bogoliubov):
        self.nelec = nelec
        self.spin = spin
        if spinAdapted and bogoliubov:
            log.fatal("Bogoliubov calculation with spin adaption is not implemented")
            raise Exception
        self.spinAdapted = spinAdapted
        self.bogoliubov = bogoliubov
        self.sys_initialized = True

    def set_integral(self, H0, H1, H2):
        if self.sys_initialized = False:
            log.error("set_integral() should be used after initializing set_system()")
        pass
        integral.dump()
        self.integral_initialized = True

    def copy_restartfile(self, src):
        pass

    def set_schedule(self, schedule):
        pass

    def run(self, rdm = True):
        pass
        return truncation, energy, (rdm)

    def evaluate(self):
        pass

    def extrapolate(self, rdm = True, evaluate = False):
        pass

        
    
    #default = {
    #    # executables
    #    'block_path': realpath(join(dirname(realpath(__file__)), "../block")),
    #    # symmetry
    #    'sym_n': True, # particle number conserving
    #    'sym_s': True, # spin-adapted
    #    # sweep control
    #    'e_tol': 1e-6,
    #    'max_it': 30,
    #    'minM': 250,
    #    'maxM': 400,
    #    'twodot_to_onedot': None,
    #    'schedule': None, # a ([(start_iter, M, tol, noise), (start_iter, M, tol, noise), ...], twodot_to_onedot)
    #    # whether or not compute onepdm
    #    'onepdm': True,
    #    # mpi information
    #    'nproc': 1,
    #    'nnode': 1,
    #    'nelec': None,
    #    'nsites': None,
    #    'temp': None,
    #    'temp_parent': "/tmp"
    #}

    #def __init__(self, **kwargs):
    #    for key in self.default:
    #        if key in kwargs:
    #            self.__dict__[key] = kwargs[key]            
    #        else:
    #            self.__dict__[key] = self.default[key]
    #    if self.temp is None:
    #        self.temp = mkdtemp(prefix = "BLOCK", dir = self.temp_parent)
    #    if self.sym_n is False:
    #        
    #        self.nelec


    #def optimize(self, Int1e, Int2e, nelec = None):
    #    if nelec is not None:
    #        self.nelec = nelec
    #    self.__write_config()

    #def evaluate(self):
    #    pass

    #def __write_config(self):
    #    with open(join(self.temp, "dmrg.conf"), "w") as f:
    #        f.write("nelec %d" % self.nelec)

if __name__ == "__main__":
    log.verbose = log.Level["DEBUG1"]    
    schedule = Schedule()
    
    schedule.initial(minM = 50, maxM = 400)
    schedule.get_schedule()
    
    schedule.maxiter = 12
    schedule.restart(M = 400)
    schedule.get_schedule()
    
    schedule.maxiter = 20
    schedule.sweep_tol = 1e-5
    
    schedule.custom([150, 250, 400, 600, 800, 1000, 1200], [0, 4, 8, 12, 16, 20, 24], \
        [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [0] * 7, 0)
    schedule.get_schedule()
    
    schedule.extrapolate(300)
    schedule.get_schedule()
    
    Block.set()
    print Block.block_path
