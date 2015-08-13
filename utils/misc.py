import subprocess as sub
import numpy as np

def grep(string, f, A = None, B = None):
    cmd = """grep "%s" %s""" % (string, f)
    if A is not None:
        cmd += " -A %d" % A
    if B is not None:
        cmd += " -B %d" % B
    cmd += "; exit 0"
    return sub.check_output(cmd, shell = True)[:-1]

def mdot(*args):
    r = args[0]
    for a in args[1:]:
      r = np.dot(r, a)
    return r

def find(x, l):
    return [i for i, v in enumerate(l) if v == x]

def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.count+= 1
        return fn(*args, **kwargs)
    wrapper.count= 0
    wrapper.__name__= fn.__name__
    return wrapper
