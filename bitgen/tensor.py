import numpy as np
from copy import deepcopy
import tensor_symm as symmetry
import libdmet.utils.logger as log

__all__ = ["BaseTensor", "Fermion", "NumTensor", "Delta"]

def isidx(i):
    # an index should be a lower case letter, eg. i,j,k,l, ...
    return len(i) == 1 and 'a' <= i and i <= 'z'

def isindices(indices):
    return reduce(lambda a,b: a and b, map(isidx, indices), True)

class BaseTensor(object):
    # the base class for all tensors
    def set_idx(self, idx):
        if idx is not None:
            assert(len(idx) == self.nidx and isindices(idx))
            self.idx = tuple(idx)
        else:
            self.idx = None

    def replace_idx(self, idx, new_idx):
        if idx in self.idx:
            self.set_idx([new_idx if x == idx else x for x in list(self.idx)])

    def rm_idx(self):
        # remove index to get the base type of the operator
        self.idx = None

    def equiv(self):
        equivs = []
        for idx in self.symm.symm(self.idx):
            tensor1 = deepcopy(self)
            tensor1.set_idx(idx)
            equivs.append(tensor1)
        return equivs
            

class Fermion(BaseTensor):
    # a single fermion creation / destruction operator
    def __init__(self, cre, spin, idx = None):
        assert(spin in ['A', 'B'])
        assert(cre in [True, False])
        self.nidx = 1
        self.symm = symmetry.IdxNoSymm(1)
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
        return Fermion(not self.cre, self.spin, self.idx)

    def replace_idx(self, idx, new_idx):
        if self.idx[0] == idx:
            self.set_idx(new_idx)

    def __hash__(self):
        return hash((self.cre, self.spin, self.idx))

    def __eq__(self, other):
        return self.spin == other.spin and self.cre == other.cre \
                and other.idx in self.symm.symm(self.idx)

    def __repr__(self):
        if self.cre:
            s = 'c_'
        else:
            s = 'd_'
        s += self.spin
        if self.idx is not None:
            s += "(%s)" % self.idx
        return s

class NumTensor(BaseTensor):
    def __init__(self, name, idx = None, nidx = None, symm = None):
        if nidx is not None:
            self.nidx = nidx
        elif idx is not None:
            self.nidx = len(idx)
        else:
            raise Exception("One of idx and nidx must be provided")
        if symm is None:
            self.symm = symmetry.IdxNoSymm(self.nidx)
        else:
            self.symm = symm
        self.name = name
        self.set_idx(idx)

    def dn(self):
        return 0

    def ds(self):
        return 0

    def conj(self):
        return deepcopy(self)

    def __hash__(self):
        return hash((self.name, self.nidx, self.idx))

    def __eq__(self, other):
        # FIXME do we require the have the same symm class?
        return self.name == other.name and self.nidx == other.nidx \
                and other.idx in self.symm.symm(self.idx)

    def __repr__(self):
        s = self.name
        if self.idx is not None:
            s += "(" + ",".join(self.idx) + ")"
        return s

class Delta(BaseTensor):
    def __init__(self, idx = None):
        self.nidx = 2
        self.symm = symmetry.IdxSymm()
        self.set_idx(idx)

    def dn(self):
        return 0

    def ds(self):
        return 0

    def conj(self):
        return deepcopy(self)

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return other.idx in self.symm.symm(self.idx)

    def __repr__(self):
        s = "Delta"
        if self.idx is not None:
            s += "(%s,%s)" % (self.idx)
        return s

if __name__ == "__main__":
    log.section("tensor.py: defines the basic tensor types")
    log.result("This is a fermion creation operator, %s", Fermion(True, 'A', 'i'))
    log.result("This is a delta function, %s", Delta('ij'))
    log.result("Is %s equal to %s? %s", Delta('ij'), \
            Delta('ji'), Delta('ij') == Delta('ji'))
    w = NumTensor('w', 'ijkl', symm = symmetry.Idx8FoldSymm())
    log.result("8-fold symmetry of (ij||kl):\n%s", w.equiv())
