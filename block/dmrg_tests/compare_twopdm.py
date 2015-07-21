import numpy as np
from libdmet.solver.block import read2pdm
import itertools as it

twopdm1 = read2pdm("hubbard/twopdm.0.0.txt")
twopdm2 = read2pdm("hubbard/ref_twopdm.0.0.txt")

count = 0
for i,j,k,l in it.product(range(24), repeat = 4):
    if abs(twopdm1[i,j,k,l] - twopdm2[i,j,k,l]) > 1e-4:
        if count < 10:
            print "%3d%3d%3d%3d%12.6f%12.6f" % (i,j,k,l, twopdm1[i,j,k,l], twopdm2[i,j,k,l])
        count += 1

print "Number of different terms:", count
