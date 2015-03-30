import numpy as np
import itertools as it
from commands import getoutput
import os
import struct
from tempfile import mkdtemp

from utils import WriteFile, ReadFile, ToClass, ToSpatOrb
from settings import param_block as g

def gen_orbitalfile(Int1e, Int2e, UHFB):

  def empty_line():
    return "%20.16f%4d%4d%4d%4d" % (0., 0, 0, 0, 0)
  
  def insert_ccdd(lines, v, nsites):
    # (ij||kl) convention indices are i k j l
    for idx1, (i, k) in enumerate(it.product(range(nsites), repeat = 2)):
      for idx2, (j, l) in enumerate(it.product(range(nsites), repeat = 2)):
        if v.has_key((i,j,l,k)):
          lines.append("%20.16f%4d%4d%4d%4d" % (v[(i,j,l,k)], i+1,k+1,j+1,l+1))
          del v[(i,j,l,k)]
    assert(len(v) == 0)
    lines.append(empty_line())
  
  def insert_cccd(lines, v, nsites):
    # plain index i j k l
    for i in range(nsites):
      for j in range(i):
        for k, l in it.product(range(nsites), repeat = 2):
          if v.has_key((i,j,k,l)):
            lines.append("%20.16f%4d%4d%4d%4d" % (v[(i,j,k,l)], i+1,j+1,k+1,l+1))
            del v[(i,j,k,l)]
    assert(len(v) == 0)
    lines.append(empty_line())
  
  def insert_cccc(lines, v, nsites, UHFB):
    # index order i j l k, where i>j, l>k
    for i in range(nsites):
      for j in range(i):
        if UHFB:
          for l in range(nsites):
            for k in range(l):
              if v.has_key((i,j,k,l)):
                lines.append("%20.16f%4d%4d%4d%4d" % (v[(i,j,k,l)], i+1,j+1,l+1,k+1))
                del v[(i,j,k,l)]
        else:
          for l in range(i+1):
            for k in range(l):
              if (i > l or j >= k) and v.has_key((i,j,k,l)):
                lines.append("%20.16f%4d%4d%4d%4d" % (v[(i,j,k,l)], i+1,j+1,l+1,k+1))
                del v[(i,j,k,l)]
    assert(len(v) == 0)
    lines.append(empty_line())
  
  def insert_symm_matrix(lines, v, nsites):
    assert(np.allclose(v.T - v, 0.))
    for i in range(nsites):
      for j in range(i+1):
        lines.append("%20.16f%4d%4d%4d%4d" % (v[i, j], i+1,j+1,0,0))
    lines.append(empty_line())
  
  def insert_matrix(lines, v, nsites):
    for i in range(nsites):
      for j in range(nsites):
        lines.append("%20.16f%4d%4d%4d%4d" % (v[i, j], i+1,j+1,0,0))
    lines.append(empty_line())

  h0, d0 = Int1e.cd, Int1e.cc
  vccdd, vcccd, vcccc = Int2e.ccdd, Int2e.cccd, Int2e.cccc
  if UHFB:
    h0 = ToSpatOrb(h0)
    nsites = h0[0].shape[0]
    orbitalfile = []
    orbitalfile.append(" &BCS NORB=%2d," % nsites)
    orbitalfile.append("  ORBSYM=%s" % ("1,"*nsites))
    orbitalfile.append("  ISYM=1,")
    orbitalfile.append("  IUHF=1,")
    orbitalfile.append(" &END")
    # vccdd_alpha, vccdd_beta, vccdd_ab
    for v in vccdd:
      insert_ccdd(orbitalfile, v, nsites)
    # vcccd_a, vcccd_b
    for v in vcccd:
      insert_cccd(orbitalfile, v, nsites)
    # vcccc
    insert_cccc(orbitalfile, vcccc, nsites, True)
    # h0a, h0b
    for h in h0:
      insert_symm_matrix(orbitalfile, h, nsites)
    # d0
    insert_matrix(orbitalfile, d0, nsites)
 
  else:
    nsites = h0.shape[0]
    orbitalfile = []
    orbitalfile.append(" &BCS NORB=%2d," % nsites)
    orbitalfile.append("  ORBSYM=%s" % ("1,"*nsites))
    orbitalfile.append("  ISYM=1,")
    orbitalfile.append(" &END")
    # vccdd
    insert_ccdd(orbitalfile, vccdd, nsites)
    # vcccd (a)
    insert_cccd(orbitalfile, vcccd, nsites)
    # vcccc
    insert_cccc(orbitalfile, vcccc, nsites, False)
    # h0
    insert_symm_matrix(orbitalfile, h0, nsites)
    # d0
    insert_symm_matrix(orbitalfile, d0, nsites)
  
  orbitalfile.append(empty_line()) # core energy is zero
  return "\n".join(orbitalfile) + "\n"


def gen_config(options, path, verbose):
  # prepare "dmrg.conf" file
  configfile = []
  configfile.append("nelec %d\nspin 0\nhf_occ integral" % (options["npar"]))
  if not options["restart"] or options["M"][0] != options["M"][1]:
    configfile.append("schedule default\nStartM %d\nmaxM %d\nsweep_tol %e" % \
        (options["M"][0], options["M"][1], options["tol"]))
  else:
    configfile.append("schedule")
    configfile.append("0 %d %e %e" % (options["M"][0], options["tol"], options["tol"]))
    configfile.append("2 %d %e 0" % (options["M"][0], options["tol"] * 0.1))
    configfile.append("end\n")
    configfile.append("twodot_to_onedot 4\nsweep_tol %e" % options["tol"])
  
  if options["restart"]:
    configfile.append("fullrestart")
  else:
    configfile.append("warmup local_2site")

  configfile.append("maxiter %d" % options["max_it"])
  configfile.append("orbitals %s/DMETDUMP" % path)
  configfile.append("nonspinadapted\nbogoliubov\nonepdm")
  configfile.append("noreorder")
  configfile.append("prefix %s" % path)

  configfile.append("outputlevel %d" % (verbose-4))
  
  return "\n".join(configfile) + "\n"

def gen_config_energy(options, path):
  # prepare "dmrg.conf" file
  configfile = []
  configfile.append("nelec %d\nspin 0\nhf_occ integral" % (options["npar"]))
  if not options["restart"] or options["M"][0] != options["M"][1]:
    configfile.append("schedule default\nStartM %d\nmaxM %d\nsweep_tol %e" % \
        (options["M"][0], options["M"][1], options["tol"]))
  else:
    configfile.append("schedule")
    configfile.append("0 %d %e %e" % (options["M"][0], options["tol"], options["tol"]))
    configfile.append("2 %d %e 0" % (options["M"][0], options["tol"] * 0.1))
    configfile.append("end\n")
    configfile.append("twodot_to_onedot 4\nsweep_tol %e" % options["tol"])
  
  if options["restart"]:
    configfile.append("fullrestart")
  else:
    configfile.append("warmup local_2site")

  configfile.append("maxiter %d" % options["max_it"])
  configfile.append("orbitals %s/DMETDUMP_energy" % path)
  configfile.append("nonspinadapted\nbogoliubov\nonepdm")
  configfile.append("noreorder")
  configfile.append("prefix %s" % path)

  configfile.append("outputlevel 0")
  
  return "\n".join(configfile) + "\n"

def gen_cmd(nproc, node, path, src_path = None):
  if nproc is None:
    nproc = g["nproc"]
  if node is None:
    node = g["node"]

  cmds = []
  cmds.append("mpirun -npernode 1 cp %s/dmrg.conf %s" % (src_path, path))
  cmds.append("mpirun -npernode 1 cp %s/DMETDUMP %s" % (src_path, path))
  cmds.append("rm -rf %s" % src_path)

  if g.has_key("mpi_cmd"):
    run_cmd = [g["mpi_cmd"]]
  else:
    run_cmd = ["mpirun -np"]
  run_cmd.append("%d" % (nproc * node))
  
  if g["bind"]:
    run_cmd.append("--bind-to-socket")
  run_cmd.append(g["exec"])
  run_cmd.append("%s/dmrg.conf > %s/dmrg.out" % (path, path))
  run_cmd = " ".join(run_cmd)
  cmds.append(run_cmd)
  return cmds


def read_results(UHFB, path):
  def readrdm(file):
    with open(file, "r") as f:
      lines = f.readlines()
    
    nsites = int(lines[0])
    rdm = np.zeros((nsites, nsites))
    
    for line in lines[1:]:
      tokens = line.split(" ")
      rdm[int(tokens[0]), int(tokens[1])] = float(tokens[2])
    
    return rdm

  file_e = open("%s/dmrg.e" % path, "rb")
  energy = struct.unpack('d', file_e.read(8))[0]
  # read rdm and kappa
  if UHFB:
    rdm = readrdm("%s/onepdm.0.0.txt" % path)
  else:
    rdm = readrdm("%s/spatial_onepdm.0.0.txt" % path) / 2
  kappa = readrdm("%s/spatial_pairmat.0.0.txt" % path)
  if not UHFB:
    kappa = (kappa + kappa.T) / 2
  return ToClass({"E":energy, "rho":rdm, "kappa":kappa})


def run_emb(Int1e, Int2e, path, options, verbose = 0):
  UHFB = options["UHFB"]
  orbital = gen_orbitalfile(Int1e, Int2e, UHFB)

  # default options
  if not options.has_key("tol"):
    options["tol"] = 1e-6
  if not options.has_key("max_it"):
    if options["restart"]:
      options["max_it"] = 16
    else:
      options["max_it"] = 30

  #dmet_path = os.getcwd()
  config = gen_config(options, path, verbose)
  
  shr_path = mkdtemp(prefix = "BLOCK_SRC", dir = g["SharedDir"])
  cmd = gen_cmd(options["nproc"], options["node"], path, shr_path)
  
  if verbose > 3:
    print "Configure File"
    print config
    print "Block Command"
    for c in cmd:
      print c
    print

  WriteFile("%s/DMETDUMP" % shr_path, orbital)
  WriteFile("%s/dmrg.conf" % shr_path, config)
  
  for c in cmd:
    getoutput(c)
  
  blockoutput = getoutput("grep 'Sweep Energy' %s/dmrg.out" % path)

  if verbose > 1:
    print blockoutput

  results = read_results(UHFB, path)
  
  #os.chdir(dmet_path)
  return results

def run_energy(Int1e, Int2e, path, options):
  UHFB = options["UHFB"]
  orbital = gen_orbitalfile(Int1e, Int2e, UHFB)

  # default options
  if not options.has_key("tol"):
    options["tol"] = 1e-6
  if not options.has_key("max_it"):
    options["max_it"] = 30

  #dmet_path = os.getcwd()
  config = gen_config_energy(options, path)
  
  cmd = " ".join([g["exec_OH"], "%s/dmrg.conf.energy > %s/dmrg.energy.out" % (path, path)])
  
  WriteFile("%s/DMETDUMP_energy" % path, orbital)
  WriteFile("%s/dmrg.conf.energy" % path, config)
  
  getoutput(cmd)
  
  with open("%s/dmrg.energy.out" %path, "r") as f:
    energy = float(f.readlines()[-1].split("=")[-1])

  return energy

#def run_trans_and_overlap(norbs, path, options):
#  UHFB = options["UHFB"]
  
