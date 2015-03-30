#!/usr/bin/env python2.7

import numpy as np
import numpy.linalg as la
import itertools as it
import sys
sys.path.append("..")

from inputs import Input
from geometry import BuildLatticeFromInput, Topology
from ham import Hamiltonian
from mean_field import init_mfd
from BCSdmet import BCSdmet

from dmet_run import default_keywords
from main import ChooseRoutine
from utils import ToClass

def summary(g):
  print "\nDMET MODEL SUMMARY\n"
  for item in g.Common.items():
    print "%-8s" % item[0]
    for subitem in item[1].items():
      print "    %-12s = %s" % (subitem[0], subitem[1])
    print
  assert(len(g.IterOver) == 0)
  assert(len(g.First) == 0)
  assert(len(g.FromPrevious) == 0)


def compute_dispersion(InputDict):
  np.set_printoptions(precision = 12, suppress = True, threshold = 10000, linewidth = np.inf)
  # parse input dictionary
  Inp = Input(InputDict)
  Inp.gen_control_options()

  verbose = Inp.FORMAT.Verbose
  OrbType = Inp.DMET.OrbType

  # build the essential classes
  Lattice = BuildLatticeFromInput(Inp.GEOMETRY, OrbType, verbose)
  Topo = Topology(Lattice)
  Ham = Hamiltonian(Inp.HAMILTONIAN, Inp.CTRL, Lattice)
  Lattice.set_Hamiltonian(Ham)
  Dmet = ChooseRoutine(Inp.DMET, Inp.CTRL, Lattice, Topo)
  
  MfdSolver = init_mfd(Inp.MFD, Inp.CTRL, Lattice, Ham, Dmet.trans_lat_int2e, Inp.DMET.OrbType)

  Vcor = Dmet.GuessVcor(Lattice, Interaction=Ham.get_Int2e())
  Mu = Inp.DMET.InitMu
  MfdSolver.run(Vcor, Mu, verbose)
  MfdResult = MfdSolver.run_k(Vcor, Mu, verbose)
  # now compute dispersion with MfdResult
  dos = []
  all_poles = []
  all_residues = []
  for i in range(Lattice.nscells):
    poles = []
    residues = []
    e = MfdResult.e[i]
    u = MfdResult.u[i]
    v = MfdResult.v[i]
    if OrbType == "UHFB":
      # Mu = 0
      u_a = np.diag(np.dot(u[0].T.conj(), u[0])).real
      u_b = np.diag(np.dot(u[1].T.conj(), u[1])).real
      v_a = np.diag(np.dot(v[0].T.conj(), v[0])).real
      v_b = np.diag(np.dot(v[1].T.conj(), v[1])).real
      for p in range(len(e[0])):
        poles.append(e[0][p]-Mu*(u_a[p]-v_b[p]))
        residues.append(u_a[p])
        poles.append(-e[0][p]+Mu*(u_a[p]-v_b[p]))
        residues.append(v_a[p])
      for p in range(len(e[1])):
        poles.append(e[1][p]-Mu*(u_b[p]-v_a[p]))
        residues.append(u_b[p])
        poles.append(-e[1][p]+Mu*(u_b[p]-v_a[p]))
        residues.append(v_b[p])
    else:
      u_sum = np.diag(np.dot(u.T.conj(), u)).real
      v_sum = np.diag(np.dot(v.T.conj(), v)).real
      for p in range(len(e)):
        poles.append(e[p]-Mu*(u[p]-v[p])-Mu)
        residues.append(u[p]*2)
        poles.append(-e[p]+Mu*(u[p]-v[p])-Mu)
        residues.append(v_a[p]*2)
    
    all_poles.append(poles)
    all_residues.append(residues)
  return ToClass({"k": Lattice.get_kpoints(), "poles": all_poles, "res": all_residues, "n": Inp.DMET.Filling})

class SpectralFunc(object):
  def __init__(self, params):
    self.k = params.k
    self.poles = params.poles
    self.res = params.res
    self.nkpts = np.product(map(lambda kpts: len(kpts), self.k))
    self.filling = params.n

  def __call__(self, w, d = 0.02):
    n = np.zeros(self.nkpts)
    for i in range(self.nkpts):
      n[i] = reduce(lambda x,y: x+y, \
          map(lambda j: self.res[i][j] / (d**2+(w-self.poles[i][j])**2), \
          range(len(self.poles[i])))) * d / np.pi
    return n

  def plot_w(self, w, d = 0.01):
    n = self.__call__(w, d).reshape(tuple(map(lambda kpts:len(kpts), self.k)))
    import matplotlib.pyplot as plt
    plt.imshow(n)
    plt.title("e=%f" % w)
    plt.show()

  def plot_dos(self, d = 0.01, wlim = [-2, 2]):
    ws = np.linspace(wlim[0], wlim[1], 2000)
    ns = map(lambda w: np.average(self.__call__(w, d)), ws)
    import matplotlib.pyplot as plt
    print np.sum(ns)
    n_occ = np.sum(ns) * self.filling
    for i in range(1, len(ns)):
      if np.sum(ns[:i]) > n_occ and np.sum(ns[:i-1]) < n_occ:
        e_f = (wlim[1]-wlim[0]) * float(i) / 2000 + wlim[0]
    print e_f
    plt.plot(ws, ns)
    plt.plot([e_f, e_f], [0, 1])
    plt.show()

if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise Exception("No input file.")
  np.set_printoptions(precision = 12, suppress = True, threshold = 10000, linewidth = np.inf)
  
  filename = sys.argv[1]
  if filename.endswith(".py"):
    filename = filename[:-3].replace("/", ".")

  exec("import %s as g" % filename)
  default_keywords(g)
  assert(g.Common["DMET"]["InitGuessType"] == "MAN")
  summary(g)
  
  spectra = SpectralFunc(compute_dispersion(g.Common))
  
  #for mu in np.linspace(-10, 10, 21):
  #  spectra.plot_w(mu)
  spectra.plot_dos(wlim = [-12, 12], d = 0.05)

  sys.stdout.flush()
