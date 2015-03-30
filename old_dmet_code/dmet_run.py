#!/usr/bin/env python2.7

import numpy as np
import sys
import itertools as it
from copy import deepcopy
import pickle as p
import inspect

from settings import ArchieveDir
from main import main as dmet
from tempfile import mkdtemp
from inputs import dump_input
from utils import ToClass

def ResultTable(results, id, dmet_result, additional_result = None):
  data_dict = [
    ("ID",     " Job",               "%3d "   ),
    ("U",      "     U  ",           "%8.4f"  ),
    ("V",      "     V  ",           "%8.4f"  ),
    ("J",      "     J  ",           "%8.4f"  ),
    ("t",      "     t  ",           "%8.4f"  ),
    ("V1",     "     V1 ",           "%8.4f"  ),
    ("J1",     "     J1 ",           "%8.4f"  ),
    ("t1",     "     t1 ",           "%8.4f"  ),
    ("System", "           Name   ", "%18s"   ),    
    ("Nelec",  "        Nelec     ", "%18.12f"),
    ("Energy", "          E       ", "%18.12f"),
    ("Dwave",  "     D-Wave.Order ", "%18.12f"),
    ("Swave",  "     S-Wave.Order ", "%18.12f"),    
    ("AF",     "       AF.Order   ", "%18.12f"),
    ("Mu",     "    Chemical.Pot. ", "%18.12f"),
    ("Gap",    "         Gap      ", "%18.12f"),
    ("dVcor",    "   VcorError"     , "%12.5f"),
    ("EmbRdmErr","    RdmError"     , "%12.5f"),
    ("Conv",   "   Conv"            , "%8s"   ),
  ]
  data = []
  names = []
  fmt = []
  value = []
  for term in data_dict:
    if term[0] == "ID":
      data.append(term[0])
      names.append(term[1])
      fmt.append(term[2])
      value.append(id)
    elif term[0] in dmet_result.__dict__:
      data.append(term[0])
      names.append(term[1])
      fmt.append(term[2])
      value.append(getattr(dmet_result, term[0]))
    elif additional_result is not None and term[0] in additional_result.__dict__:
      data.append(term[0])
      names.append(term[1])
      fmt.append(term[2])
      value.append(getattr(additional_result, term[0]))

  names = "".join(names)
  fmt = "".join(fmt)
  
  results.append(fmt % tuple(value))
  print "\n\nResult Table After Finishing %3d Jobs:\n" % len(results)
  print names
  print '-' * (len(names) + 3)
  for r in results:
    print r
  print
  return results

def banner():
  print "***********************************************************************"
  print
  print "                    D M E T   B C S   P r o g r a m                    "
  print
  print "                             08/2014                                   "
  print
  print "***********************************************************************"
  print

def default_keywords(g):
  if not "Common" in g.__dict__.keys():
    setattr(g, "Common", {})
  keywords = ["IterOver", "First", "FromPrevious"]
  for key in keywords:
    if not key in g.__dict__.keys():
      setattr(g, key, [])

def summary(g):
  print "\nDMET MODEL SUMMARY\n"
  print "Common Settings:\n"
  for item in g.Common.items():
    print "%-8s" % item[0]
    for subitem in item[1].items():
      print "    %-12s = %s" % (subitem[0], subitem[1])
    print

  print "\nIteration over Settings:"
  for item in g.IterOver:
    print "    %-12s = %s" % (item[1], item[2])
  if len(g.IterOver) == 0:
    print "  None"
  print "Total Number of Jobs: %d\n" % np.product([len(v[2]) for v in g.IterOver])

  print "Special Settings for First Iteration:"
  for item in g.First:
    print "    %-12s = %s" % (item[1], item[2])
  if len(g.First) == 0:
    print "  None"
  print

  print "Settings From Previous Converged Calculation:"
  for item in g.FromPrevious:
    print "    %-12s = %s" % (item[1], item[2])
  if len(g.FromPrevious ) == 0:
    print "  None"
  print

def expand_iter(iterover):
  keys = []
  vals = []
  for item in iterover:
    assert(len(item) == 3)
    keys.append((item[0], item[1]))
    vals.append(item[2])

  val_expand = list(it.product(*vals))
  return keys, val_expand, vals

if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise Exception("No input file.")
  np.set_printoptions(precision = 12, suppress = True, threshold = 10000, linewidth = np.inf)
  banner()

  filename = sys.argv[1]
  if filename.endswith(".py"):
    filename = filename[:-3].replace("/", ".")
  
  exec("import %s as g" % filename)
  print "Input File\n"
  print ">" * 60
  print inspect.getsource(g)
  print "<" * 60
  print
  
  default_keywords(g)
  summary(g)
  
  if "Test" in g.__dict__ and g.Test == False and ArchieveDir is not None:
    import time as t
    epctime = int(t.time())
    Arch = mkdtemp(prefix = filename + "." + str(epctime) + ".", dir = ArchieveDir)
    with open(Arch+"/input.py", "w") as f:
      f.write(inspect.getsource(g))
    with open(Arch+"/log", "w") as f:
      f.write("# Created @ " + t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(epctime)) + "\n")

  iter_key, iter_val, compact_val = expand_iter(g.IterOver)
  History = []
  
  for i, val in enumerate(iter_val):
    # add common options
    input = deepcopy(g.Common)
    # for each job, add their own special options
    for j, key in enumerate(iter_key):
      if not key[0] in input.keys():
        input[key[0]] = {}
      input[key[0]][key[1]] = val[j]
    # special option for the first run
    if i == 0:
      for item in g.First:
        if not item[0] in input.keys():
          input[item[0]] = {}
        input[item[0]][item[1]] = item[2]
    else: # special option from previous run
      for item in g.FromPrevious:
        if not item[0] in input.keys():
          input[item[0]] = {}
        input[item[0]][item[1]] = getattr(out, item[2])
    
    sys.stdout.flush()
    print "\n---------- Entering Job %03d ----------" % i
    inp, out = dmet(input)
    # FIXME this is a hack, should come up with better ways to deal with it
    if inp.HAMILTONIAN.Type == "Hubbard":
      empty = {}
      for k,v in inp.HAMILTONIAN.Params.items():
        if v != 0. or k in ["U", "V", "J", "t"]:
          empty[k] = v
      History = ResultTable(History, i, out, ToClass(empty))
    else:
      History = ResultTable(History, i, out, ToClass({"System":inp.HAMILTONIAN.Name}))

    if "Test" in g.__dict__ and g.Test == False and ArchieveDir is not None:
      arch_file = Arch+"/JobResult%03d.pickle" % i
      with open(arch_file, "w") as f:
        p.dump([dump_input(inp), out.__dict__], f)
      print "Results archieved to %s" % arch_file
      print

    sys.stdout.flush()    
