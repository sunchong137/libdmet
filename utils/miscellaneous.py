import subprocess as sub

def grep(string, f):
    return sub.check_output(["""grep "%s" %s; exit 0""" % (string, f)], shell = True)[:-1]
