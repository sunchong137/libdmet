import numpy as np
from os.path import * 
from tempfile import mkdtemp

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

    default = {
        # executables
        'block_path': realpath(join(dirname(realpath(__file__)), "../block")),
        # symmetry
        'sym_n': True, # particle number conserving
        'sym_s': True, # spin-adapted
        # sweep control
        'e_tol': 1e-6,
        'max_it': 30,
        'minM': 250,
        'maxM': 400,
        'twodot_to_onedot': None,
        'schedule': None, # a ([(start_iter, M, tol, noise), (start_iter, M, tol, noise), ...], twodot_to_onedot)
        # whether or not compute onepdm
        'onepdm': True,
        # mpi information
        'nproc': 1,
        'nnode': 1,
        'nelec': None,
        'nsites': None,
        'temp': None,
        'temp_parent': "/tmp"
    }

    def __init__(self, **kwargs):
        for key in self.default:
            if key in kwargs:
                self.__dict__[key] = kwargs[key]            
            else:
                self.__dict__[key] = self.default[key]
        if self.temp is None:
            self.temp = mkdtemp(prefix = "BLOCK", dir = self.temp_parent)
        if self.sym_n is False:
            
            self.nelec


    def optimize(self, Int1e, Int2e, nelec = None):
        if nelec is not None:
            self.nelec = nelec
        self.__write_config()

    def evaluate(self):
        pass

    def __write_config(self):
        with open(join(self.temp, "dmrg.conf"), "w") as f:
            f.write("nelec %d" % self.nelec)



if __name__ == "__main__":
    Block(block_path = "None")
