import subprocess as sub
import numpy as np

def grep(string, f):
    return sub.check_output(["""grep "%s" %s; exit 0""" % (string, f)], shell = True)[:-1]

def mdot(*args):
    r = args[0]
    for a in args[1:]:
      r = np.dot(r, a)
    return r
