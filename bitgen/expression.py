from tensor import *
import numpy as np
from copy import deepcopy
import libdmet.utils.logger as log

# for simplicity, we have product before sum, not the otherway around
__all__ = ["OpProduct", "OpSum"]

class OpProduct(object):
    # operator strings
    def __init__(self, oplist):
        if isinstance(oplist, BaseTensor):
            self.oplist = [oplist]
        else:
            for op in oplist:
                assert(isinstance(op, BaseTensor))
            self.oplist = oplist
        self._sort()

    def _sort(self):
        oplist = deepcopy(self.oplist)
        tnum, tdelta, tfermion = [], [], []
        for op in oplist:
            if isinstance(op, NumTensor):
                tnum.append(op)
            elif isinstance(op, Delta):
                tdelta.append(op)
            else:
                tfermion.append(op)
        self.n_num = len(tnum)
        self.n_del = len(tdelta)
        self.n_fermion = len(tfermion)
        self.oplist = tnum + tdelta + tfermion
        assert(len(self.oplist) == len(oplist))

    def append(self, other):
        self.oplist += other.oplist
        self._sort()
    
    # acess part of the operator product
    def nums(self):
        return OpProduct(self.oplist[:self.n_num])

    def deltas(self):
        return OpProduct(self.oplist[self.n_num:self.n_num+self.n_del])
    
    def nonfermions(self):
        return OpProduct(self.oplist[:self.n_num+self.n_del])

    def fermions(self):
        return OpProduct(self.oplist[self.n_num+self.n_del:])

    def add_indices(self, indices = None):
        if indices is None:
            indices = "ijklmn"
        count = 0
        for op in self.oplist:
            op.set_idx(indices[count:count+op.nidx])
            count += op.nidx

    def rm_indices(self):
        for op in self.oplist:
            op.rm_idx()

    def replace_indices(self, *pairs):
        oplist = deepcopy(self.oplist)
        for idx, new_idx in pairs:
            for op in oplist:
                op.replace_idx(idx, new_idx)
        return OpProduct(oplist)

    def dn(self):
        return np.sum(map(lambda op: op.dn(), self.oplist))

    def ds(self):
        return np.sum(map(lambda op: op.ds(), self.oplist))

    def conj(self):
        return OpProduct(map(lambda op: op.conj(), self.oplist[::-1]))

    def permute(self, i, j):
        assert(abs(i-j) == 1)
        sums = []
        oplist = deepcopy(self.oplist)
        delta = None
        if isinstance(oplist[i], Fermion) and isinstance(oplist[j], Fermion):
            factor = -1
            if oplist[i].spin == oplist[j].spin and oplist[i].cre != oplist[j].cre:
                assert(oplist[i].idx is not None and oplist[j].idx is not None)
                delta = Delta([oplist[i].idx[0], oplist[j].idx[0]])
        else:
            factor = 1
        oplist[i], oplist[j] = oplist[j], oplist[i]
        sums.append((factor, OpProduct(oplist)))
        if delta is not None:
            sums.append((1, OpProduct([delta] + oplist[:i] + oplist[j+1:])))
        return sums

    def __hash__(self):
        return hash(tuple([op.__hash__() for op in self.oplist]))

    def __eq__(self, other):
        return self.oplist == other.oplist

    def __repr__(self):
        return "*".join(map(lambda op: op.__repr__(), self.oplist))

    def __mul__(self, other):
        res = deepcopy(self)
        res.append(other)
        return res

def rm_indices(ops):
    ops1 = deepcopy(ops)
    ops1.rm_indices()
    return ops1

def equiv(ops1, ops2):
    return set(ops1.fermions()) == set(ops2.fermions())

class OpSum(list):
    def __init__(self, op_terms):
        list.__init__(self, filter(lambda term: term is not None, \
                map(self._format, op_terms)))

    def _format(self, term):
        if isinstance(term, BaseTensor):
            return (1, OpProduct([term]))
        elif isinstance(term, OpProduct):
            return (1, term)
        elif len(term) == 2:
            factor, ops = term
            assert(isinstance(factor, (int, float)))
            if factor == 0:
                return None
            assert(isinstance(ops, OpProduct))
            return tuple(term)
    
    def append(self, term):
        list.append(self, _format(term))

    def replace_indices(self, *pairs):
        return OpSum([(factor, ops.replace_indices(*pairs)) for (factor, ops) in self])

    def __iadd__(self, other):
        list.__iadd__(self, other)
        return self

    def __add__(self, other):
        return OpSum(list.__add__(self, other))

    def conj(self):
        return OpSum(map(lambda (factor, term): \
                (factor, term.conj()), self))

    def __str__(self):
        if len(self) == 0:
            return "0"

        s = ""
        for i, (factor, ops) in enumerate(self):
            if factor > 0:
                if i != 0:
                    s += '+'
            elif factor < 0:
                s += '-'
            else:
                continue
            if abs(factor) != 1:
                s += str(abs(factor)) + "*"
            s += "%s" % ops
        return s

    def __repr__(self):
        return "term(" + list.__repr__(self) + ")"

if __name__ == "__main__":
    log.section("expression.py defines OpProduct and OpSum classes")
    ops = OpProduct([Fermion(False, 'B'), Fermion(False, 'B'), \
            Fermion(False, 'A'), Fermion(True, 'B')])
    ops.add_indices()
    log.result("OpProduct %s", ops)
    log.result("Complex Conjugate is %s", ops.conj())
    from tensor_symm import IdxSymm
    HA = OpProduct(NumTensor('hA', 'ij', symm = IdxSymm)) * \
            OpProduct(Fermion(True, 'A', 'i')) * OpProduct(Fermion(False, 'A', 'j'))
    HB = OpProduct(NumTensor('hB', 'ij', symm = IdxSymm)) * \
            OpProduct(Fermion(True, 'B', 'i')) * OpProduct(Fermion(False, 'B', 'j'))
    log.result("One-body Hamiltonian\n%s", OpSum([HA, HB]))
