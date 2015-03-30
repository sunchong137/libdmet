import numpy as np
from commands import getoutput

from utils import WriteFile, ReadFile, ToClass, ToSpatOrb, ToSpinOrb
from settings import param_mps

def WriteMatrix(mat, name):
  lines = ["!%s %s" % (name, mat.shape)]
  for i in range(mat.shape[0]):
    line = []
    for j in range(mat.shape[1]):
      line.append("%20.16f" % mat[i, j])
    line = " ".join(line)
    lines.append(line)
  return "\n".join(lines)


def get_cmd(path, M, nproc):
  # prepare command
  if nproc is None:
    nproc = param_mps["nproc"]
  cmd = ["OMP_NUM_THREADS=%d" % nproc]
  cmd.append(param_mps["exec"])
  cmd.append(path)
  cmd.append(str(M))
  return " ".join(cmd)

def gen_inputfile(h0, d0, U, f, g, d, UHFB):
  if UHFB:
    h0 = ToSpatOrb(h0)
    nsites = h0[0].shape[0]
    ntei = len(f[0])
    spins = ["a", "b"]

    # write dmrg input file
    input = []
    input.append("%3d%3d%20.12f UHFB" % (nsites, ntei, U))
    input.append(WriteMatrix(h0[0], "h0a"))
    input.append(WriteMatrix(h0[1], "h0b"))
    input.append(WriteMatrix(d0, "d0")) # asymmetric
    for i in range(ntei):
      for s in range(2):
        input.append(WriteMatrix(f[s][i], "f(site%2d)%s" % (i, spins[s])))
        input.append(WriteMatrix(g[s][i], "g(site%2d)%s" % (i, spins[s])))
        input.append(WriteMatrix(d[s][i], "d(site%2d)%s" % (i, spins[s])))
  else:
    nsites = h0.shape[0]
    ntei = len(f)
    
    # write dmrg input file
    input = []
    input.append("%3d%3d%20.12f" % (nsites, ntei, U))
    input.append(WriteMatrix(h0, "h0"))
    input.append(WriteMatrix(d0, "d0"))
    for i in range(ntei):
      input.append(WriteMatrix(f[i], "f(site%2d)" % i))
      input.append(WriteMatrix(g[i], "g(site%2d)" % i))
      input.append(WriteMatrix(d[i], "d(site%2d)" % i))

  return "\n".join(input) + "\n"


def run_emb(h0, d0, U, f, g, d, path, options, verbose = 0):
  UHFB = options["UHFB"]
  inputfile = gen_inputfile(h0, d0, U, f, g, d, UHFB)
  cmd = get_cmd(path, options["M"], options["nproc"])
  WriteFile(path + "/DMRG.in", inputfile)
  
  # call dmrg
  mpsoutput = getoutput(cmd)
  if verbose > 3:
    print mpsoutput
    print
  
  # read output
  energy = float(mpsoutput.split("!")[-1].split(" ")[-1])
  # rdm
  rdmA = np.loadtxt(path + "/1RDM.A", comments = "!")
  rdmB = np.loadtxt(path + "/1RDM.B", comments = "!")
  if UHFB:
    nsites = rdmA.shape[0]
    rdm = np.zeros((nsites*2, nsites*2))
    rdm = ToSpinOrb([rdmA, rdmB])
  else:
    rdm = (rdmA+rdmB)/2
  # kappa
  kappa = np.loadtxt(path + "/KAPPA", comments = "!")
  return ToClass({"E":energy, "rho":rdm, "kappa":kappa})
