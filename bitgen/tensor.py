import numpy as np
import libdmet.utils.logger as log
from copy import deepcopy

__all__ = ["BaseTensor", "fermion", "num_tensor", "delta"]

class BaseTensor(object):
    # the base class for all tensors
    def set_idx(self, idx):
        if idx is not None:
            assert(len(idx) == self.nidx and isindices(idx))
            self.idx = tuple(idx)
        else:
            self.idx = None

    def rm_idx(self):
        # remove index to get the base type of the operator
        self.idx = None

def isidx(i):
    # an index should be a lower case letter, eg. i,j,k,l, ...
    return len(i) == 1 and 'a' <= i and i <= 'z'

def isindices(indices):
    return reduce(lambda a,b: a and b, map(isidx, indices), True)

class fermion(BaseTensor):
    # a single fermion creation / destruction operator
    def __init__(self, cre, spin, idx = None):
        self.nidx = 1
        assert(spin in ['A', 'B'])
        assert(cre in [True, False])
        self.spin = spin
        self.cre = cre
        self.set_idx(idx)

    def dn(self):
        return 1 if self.cre else -1

    def ds(self):
        if self.spin == 'A':
            s = 1
        else:
            s = -1
        return s * self.dn()

    def conj(self):
        return fermion(not self.cre, self.spin, self.idx)

    def replace_idx(self, idx, new_idx):
        if self.idx[0] == idx:
            self.set_idx(new_idx)

    def __hash__(self):
        return hash((self.cre, self.spin, self.idx))

    def __eq__(self, other):
        return self.spin == other.spin and self.cre == other.cre \
                and self.idx == other.idx

    def __repr__(self):
        if self.cre:
            s = 'c_'
        else:
            s = 'd_'
        s += self.spin
        if self.idx is not None:
            s += "(%s)" % self.idx
        return s

class num_tensor(BaseTensor):
    def __init__(self, name, idx = None, nidx = None):
        if nidx is not None:
            self.nidx = nidx
        elif idx is not None:
            self.nidx = len(idx)
        else:
            raise Exception("One of idx and nidx must be provided")
        self.name = name
        self.set_idx(idx)

    def dn(self):
        return 0

    def ds(self):
        return 0

    def conj(self):
        return deepcopy(self)

    def replace_idx(self, idx, new_idx):
        if idx in self.idx:
            self.set_idx([new_idx if x == idx else x for x in list(self.idx)])

    def __hash__(self):
        return hash((self.name, self.nidx, self.idx))

    def __eq__(self, other):
        return self.name == other.name and self.nidx == other.nidx \
                and self.idx == other.idx

    def __repr__(self):
        s = self.name
        if self.idx is not None:
            s += "(" + ",".join(self.idx) + ")"
        return s

class delta(BaseTensor):
    def __init__(self, idx = None):
        self.nidx = 2
        self.set_idx(idx)

    def set_idx(self, idx):
        if idx is not None:
            assert(len(idx) == self.nidx and isindices(idx))
            self.idx = set(idx)
        else:
            self.idx = None

    def replace_idx(self, idx, new_idx):
        if idx in self.idx:
            self.set_idx([new_idx if x == idx else x for x in list(self.idx)])

    def dn(self):
        return 0

    def ds(self):
        return 0

    def conj(self):
        return deepcopy(self)

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return self.idx == other.idx

    def __repr__(self):
        s = "delta"
        if self.idx is not None:
            s += "(%s,%s)" % tuple(sorted(self.idx))
        return s
