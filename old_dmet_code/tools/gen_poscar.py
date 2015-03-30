#!/usr/bin/env python2.7

import numpy as np
import sys
sys.path.append("..")

from geometry import BuildLatticeFromInput
from utils import ToClass

def generate_coordinates(input):
  Names = map(lambda s: s[1], input.UnitCell['Sites'])
  uNames = set(Names)
  counts = map(lambda x: Names.count(x), uNames)

  input.Fragments = None
  input.BoundaryCondition = 'pbc'
  lattice = BuildLatticeFromInput(input, verbose = 5)
  assert(lattice.dim == 3)
  compound = "(" + "".join(["%s%d" % (n,c) for n,c in zip(uNames,counts)]) + ")" \
      + "x".join(["%d" % x for x in input.LatticeSize])

  coord = {}
  for n in uNames:
    coord[n] = []
  for n,s in zip(lattice.names, lattice.sites):
    coord[n].append("%20.12f%20.12f%20.12f" % tuple(s))

  with open("POSCAR.out", "w") as f:
    f.write("%s\tGenerated using DMET_BCS.gen_poscar tool\n" % compound)
    f.write("1.0000\n")
    f.write("%20.12f%20.12f%20.12f\n" % tuple(lattice.size[0]))
    f.write("%20.12f%20.12f%20.12f\n" % tuple(lattice.size[1]))
    f.write("%20.12f%20.12f%20.12f\n" % tuple(lattice.size[2]))
    f.write(" ".join(["%d" % (c*np.product(input.LatticeSize)) for c in counts]) + "\n")
    f.write("Cartesian\n")
    for n in uNames:
      f.write("\n".join(coord[n]) + "\n")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise Exception("No input file.")
  np.set_printoptions(precision = 12, suppress = True, threshold = 10000, linewidth = np.inf)
  
  filename = sys.argv[1]
  if filename.endswith(".py"):
    filename = filename[:-3].replace("/", ".")

  exec("from %s import Geometry as geom" % filename)
  geom = ToClass(geom)

  generate_coordinates(geom)

