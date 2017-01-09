import libdmet.utils.logger as log
import libdmet.dmet.abinitioBCS as dmet
import numpy as np
import numpy.linalg as la
import os

log.verbose = "INFO"

### Electron representation

Ud = 8
Up = 1.79
ed = -7.--3.5
tpd = -1.31
tpp = -0.9
tpp1 = 0.
LatSize = [2, 2]
ImpSize = [2, 2]
M = 400
Lat = dmet.Square3Band(*(LatSize + ImpSize))
Ham = dmet.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1)
Lat.setHam(Ham)
H2 = np.zeros((1,12,12,12,12))
for i in range(0, 12):
    if i % 3 == 0:
        H2[0,i,i,i,i] = Ud
    else:
        H2[0,i,i,i,i] = Up

from libdmet.system import integral
Ham = integral.Integral(12, True, False, 0, \
        {"cd": Lat.expand(Lat.H1)[np.newaxis]}, {"ccdd": H2})

solver = dmet.impurity_solver.StackBlock(nproc = 2, nthread = 2, nnode = 1, maxM = 600, \
        reorder = True, spinAdapted = False, tol = 1e-7)

onepdm, E = solver.run(Ham, nelec = 20)
np.save("onepdm_elec.npy", onepdm)

### Hole representation

#Ud = 7.4
#Up = 1.79
#ed = Up-Ud-ed
#tpd = 1.31
#tpp = 0.9
#tpp1 = 0.
#LatSize = [2, 2]
#ImpSize = [2, 2]
#M = 400
#Lat = dmet.Square3Band(*(LatSize + ImpSize))
#Ham = dmet.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1)
#Lat.setHam(Ham)
#H2 = np.zeros((1,12,12,12,12))
#for i in range(0, 12):
#    if i % 3 == 0:
#        H2[0,i,i,i,i] = Ud
#    else:
#        H2[0,i,i,i,i] = Up
#
#from libdmet.system import integral
#Ham = integral.Integral(12, True, False, 0, \
#        {"cd": Lat.expand(Lat.H1)[np.newaxis]}, {"ccdd": H2})
#
#solver = dmet.impurity_solver.StackBlock(nproc = 2, nthread = 2, nnode = 1, maxM = 600, \
#        reorder = True, spinAdapted = False, tol = 1e-7)
#
#onepdm, E = solver.run(Ham, nelec = 4)
#np.save("onepdm_hole.npy", onepdm)
#
