import numpy as np
import sys
import re

np.set_printoptions(threshold=np.inf)

def findRange(lines, keyword, keyword2 = None):
  r = [0]
  for i, line in enumerate(lines):
    if line.find(keyword) != -1:
      r.append(i)
  if keyword2 is not None:
    Found = False
    for i, line in enumerate(lines):
      if line.find(keyword2) != -1:
        r.append(i)
        found = True
        break
    if not Found:
      r.append(len(lines)-1)
  else:
    r.append(len(lines)-1)
  return r

def find_vloc(lines, uhfb):
  for i, line in enumerate(lines):
    if line.startswith("Local Potential"):
      l_vloc = i
    if line.startswith("Local Pairing Matrix"):
      l_delta = i
    if line.startswith("Mu"):
      mu = float(line.split()[2])
  
  nsites = l_delta - l_vloc - 2

  def read_matrix(start, nsites):
    mat = np.zeros((nsites, nsites))
    for i, line in enumerate(lines[start:start+nsites]):
      mat[i] = map(float, line.strip(" []\n").split())
    return mat

  if uhfb:
    nsites /= 2
    Vloc = read_matrix(l_vloc+1, nsites*2)
    Delta = read_matrix(l_delta+1, nsites)
  else:
    Vloc = read_matrix(l_vloc+1, nsites)
    Delta = read_matrix(l_delta+1, nsites)

  return Vloc, Delta, mu


def get_vcor(filename, target_j = -1, target_it = -1):
  with open(filename, "r") as f:
    lines = f.readlines()

  for line in lines:
    if line.find("OrbType") != -1:
      uhfb = "UHFB" in line
      break

  JobRange = findRange(lines, "Result Table After Finishing")
  print "# Total Number of Finished Jobs ", len(JobRange)-2

  finished = True
  if target_j < 0:
    target_j = len(JobRange)-2 # find the last job

  if target_j == len(JobRange)-2:
    print "# Warning: Try to find potential of unfinished jobs"
    finished = False
  elif target_j > len(JobRange)-2:
    print "# Job doesn't exist"
    abort()
  
  target_lines = lines[JobRange[target_j]:JobRange[target_j+1]]
  
  # switch whether this job is finished
  if not finished:
    IterRange = findRange(target_lines, "DMET Iteration", "Final")
    IterRange = IterRange[1:]
    if IterRange[-1] < len(target_lines)-1: # then all the iterations are finished
      if target_it < 0:
        target_it = len(IterRange) - 2
      elif target_it > len(IterRange) - 2:
        print "# this iteration doesn't exist, abort"
        abort()
      vloc_lines = target_lines[IterRange[target_it]: IterRange[target_it+1]]
    elif len(IterRange) < 3:
      if target_j > 0:
        print "# there's no finished iteration in this job, search for last job"
        target_j -= 1
        target_it = -1
        target_lines = lines[JobRange[target_j]:JobRange[target_j+1]]
        finished = True
      else:
        print "# No finished iteration in this file, abort"
        abort()
    else:
      if target_it < 0:
        target_it = len(IterRange) - 3
      elif target_it > len(IterRange) -3:
        print "# this iteration is not finished, abort"
        abort()
      vloc_lines = target_lines[IterRange[target_it]: IterRange[target_it+1]]
  
  if finished:
    IterRange = findRange(target_lines, "DMET Iteration", "Final")
    IterRange = IterRange[1:]
    if target_it < 0:
      target_it = len(IterRange) - 3
    elif target_it > len(IterRange) - 3:
      print "# this iteration doesn't exist, abort"
      abort()
    vloc_lines = target_lines[IterRange[target_it]: IterRange[target_it+1]]
  
  dmrg_dir = None
  for line in vloc_lines:
    if line.startswith("DMRG restart information"):
      dmrg_dir = line.split()[-1]

  return find_vloc(vloc_lines, uhfb), target_j, target_it, dmrg_dir

if __name__ == "__main__":
  filename = sys.argv[1]
  target_j = -1
  target_it = -1 # default take the last one

  if len(sys.argv) > 2:
    target_j = int(sys.argv[2])
  if len(sys.argv) > 3:
    target_it = int(sys.argv[3])
  
  (Vloc, Delta, mu), target_j, target_it, dmrg_dir = get_vcor(filename, target_j, target_it)

  output = \
"""

# restart from output file %s
# job %d iteration %d

Vloc     = %s
Delta    = %s
mu       = %s
""" % (filename, target_j, target_it, Vloc.__repr__(), Delta.__repr__(), mu.__repr__())

  if "new" in sys.argv:
    if "unres" in sys.argv[1]:
      output += \
"""
from utils import ToSpinOrb
Vloc = ToSpinOrb(Vloc)

"""
  output += \
"""
Vcor = [Vloc, Delta]
Common["DMET"]["InitGuessType"] = "MAN"
Common["DMET"]["InitGuess"] = Vcor
Common["DMET"]["InitMu"] = mu

Common["DMET"]["DiisStart"] = 0
Common["DMET"]["DiisDim"] = 6
Common["Fitting"]["TraceStart"] = 0

#Common["ImpSolver"]["nproc"] = 12
#Common["ImpSolver"]["node"] = 1
"""

  if dmrg_dir is not None:
    output += \
"""
Common["ImpSolver"]["RestartDir"] = "%s"
""" % dmrg_dir

  print output.replace("array", "np.array")
