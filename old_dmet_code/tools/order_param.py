#!/usr/bin/env python2.7

# order parameter tool for 2D Hubbard model

import numpy as np
import sys
import os
import pickle as p
sys.path.append("..")

from inputs import Input
from BCSdmet import BCSDmetResult

folder = sys.argv[1]
files = [f for f in os.listdir(folder) if f.startswith("JobResult")]

r = []

for i, file in enumerate(files):
  filename = os.path.join(folder, file)
  with open(filename, "r") as f:
    result = p.load(f)
  
  Restrict = result[0]["DMET"]["OrbType"] == "RHFB"
  BCS = result[0]["DMET"]["UseDelta"]
  U = result[0]["HAMILTONIAN"]["U"]
  occ = result[0]["DMET"]["Filling"] * 2
  imp = result[0]["GEOMETRY"]["ClusterSize"]
  conv = result[1]["Conv"]
  AF = 0.
  F = 0.
  dwave = 0.
  swave = 0.
  
  if not Restrict:
    rho = np.diag(result[1]["Rho"])
    spins = (rho[::2] - rho[1::2]) * 0.5
    F = abs(np.sum(spins)) / np.product(imp)
    if np.all(imp == np.array([2, 2])):
      upsites = [0, 3]
    elif np.all(imp == np.array([4, 2])):
      upsites = [0, 3, 4, 7]
    elif np.all(imp == np.array([6, 2])):
      upsites = [0, 3, 4, 7, 8, 11]
    elif np.all(imp == np.array([8, 2])):
      upsites = [0, 3, 4, 7, 8, 11, 12, 15]
    elif np.all(imp == np.array([4, 4])):
      upsites = [0, 2, 5, 7, 8, 10, 13, 15]
    else:
      raise Exception("Impurity not supported")
    
    for x in range(np.product(imp)):
      if x in upsites:
        AF += spins[x]
      else:
        AF -= spins[x]
    AF /= np.product(imp)
    AF = abs(AF)
  
  if BCS:
    kappa = result[1]["Kappa"]
    kappa = (kappa + kappa.T) / 2
    if np.all(imp == np.array([2, 2])):
      positive = [(0, 2), (1, 3)]
      negative = [(0, 1), (2, 3)]
    elif np.all(imp == np.array([4, 2])):
      positive = [(0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7)]
      negative = [(0, 1), (2, 3), (2, 3), (4, 5), (4, 5), (6, 7)]
      #positive = [(2, 4), (3, 5)]
      #negative = [(2, 3), (4, 5)]
    for x in positive:
      dwave += kappa[x]
      swave += kappa[x]
    for x in negative:
      dwave -= kappa[x]
      swave += kappa[x]
  
    dwave /= 2*len(positive)
    swave /= 2*len(positive)
    dwave = abs(dwave)
    swave = abs(swave)

  r.append("%3d%10.4f%20.12f%20.12f%20.12f%6s" % (i, U, occ, AF, dwave, conv))

print " Job      U            Nelec               AF.Order          Dwave.Order   Conv"
print "-----------------------------------------------------------------------------------"
print "\n".join(r)

