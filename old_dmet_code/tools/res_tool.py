#!/usr/bin/env python2.7

import sys
import pickle as p
sys.path.append("..")

from inputs import Input
from BCSdmet import BCSDmetResult


with open(sys.argv[1], "r") as f:
  result = p.load(f)

print "Input\n"
print Input(result[0])
print "Output\n"
print BCSDmetResult(dict = result[1])

