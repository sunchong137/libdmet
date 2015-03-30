# temp directory of dmrg calculation 
TmpDir = "/scratch/boxiao/DMETTemp"
StoreDir = "/tigress/boxiao/DMETTemp"
# dmrg executable

#param_mps = {
#  "exec": "/home/boxiao/office/tiger_fs/dmet_bcs/dmrg/dmrg.x",
#  "nproc": 1,
#}

import os
mypath = os.path.dirname(os.path.abspath(__file__))

blockpath = mypath + "/block"

param_block = {
  #"mpi_cmd": "mpirun -bynode --bind-to-core -np",
  "mpi_cmd": "mpirun -np",
  #"mpi_cmd": "srun -n",
  "exec": blockpath + "/block.spin_adapted",
  "exec_OH": "mpirun -np 1 " + blockpath + "/OH", # used on computation nodes
  "exec_COEF": None,
  "nproc": 12,
  "node": 2,
  "bind": False,
  "SharedDir": "/tigress/boxiao/DMETTemp",
}

ArchieveDir = "/home/boxiao/storage/BcsDmetLib"
