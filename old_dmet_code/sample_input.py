import numpy as np

Vloc     = [np.array([[ 5.01204216,  0.09004421,  0.09003157,  0.03841594],
       [ 0.09004421,  1.81313049, -0.04132074,  0.09022601],
       [ 0.09003157, -0.04132074,  1.81320205,  0.09022165],
       [ 0.03841594,  0.09022601,  0.09022165,  5.01191101]]), 
       np.array([[ 1.81291541,  0.09009428,  0.09013999, -0.0413249 ],
       [ 0.09009428,  5.01199536,  0.03843361,  0.09020736],
       [ 0.09013999,  0.03843361,  5.01200194,  0.09021975],
       [-0.0413249 ,  0.09020736,  0.09021975,  1.81337353]])]
Delta    = np.array([[  9.94100000e-05,   1.30544990e-01,  -1.30512310e-01,
          1.62000000e-05],
       [  6.71682400e-02,   1.96920000e-04,  -3.44110000e-04,
         -6.72918700e-02],
       [ -6.72156200e-02,   2.30490000e-04,  -3.97760000e-04,
          6.72456900e-02],
       [ -6.95000000e-06,  -1.30975630e-01,   1.31085600e-01,
          3.18100000e-05]])
mu       = 4.820753306922

from utils import ToSpinOrb
Vcor = (ToSpinOrb(Vloc), Delta)

Common = {}

#Common["Hamiltonian"] = {
#    "U": 4.,
#}

Common["Geometry"] = {
    "ClusterSize": np.array([2, 2]),
    "LatticeSize": np.array([24, 24]),
}
Common["DMET"] = {
    "OrbType": "UHFB",
    "Localize": None,
    "InitGuessType": "MAN",
    "MaxIter": 40,
}
Common["Fitting"] = {
    "TraceStart": 2,
    "MaxIter": 200,
}
Common["ImpSolver"] = {
    "ImpSolver": "BLOCK_DMRG",
    "Restart": True,
    "M": 400,
    "N_QP": 16,
    "TmpDir": "/scratch/gpfs/boxiao/DMETTemp"
}
Common["Format"] = {
    "Verbose": 3,
}

IterOver = [
    ("Hamiltonian", "U", np.linspace(4,5,2)),
    ("DMET", "Filling", np.linspace(0.5, 0.65, 4)),
]

First = [
    ("DMET", "InitGuessType", "AF"),
    #("DMET", "InitGuess", Vcor),
    #("DMET", "InitMu", mu),
    #("DMET", "InitGuessType", "PM")
]

FromPrevious = [
    ("DMET", "InitGuess", "Vcor"),
    ("DMET", "InitMu", "Mu"),
]

Test = True
