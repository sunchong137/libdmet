import tensor
import numpy as np
from copy import deepcopy

__all__ = ["opstring", "terms"]

class opstring(object):
    # operator strings
    def __init__(self, oplist):
        for op in oplist:
            assert(isinstance(op, tensor.BaseTensor))
        self.oplist = oplist
        self.__order()

    def append(self, other):
        self.oplist += other.oplist
        self.__order()

    def __order(self):
        oplist = deepcopy(self.oplist)
        self.oplist = []
        for op in oplist:
            if isinstance(op, tensor.num_tensor):
                self.oplist.append(op)
        self.n_num = len(self.oplist)
        for op in oplist:
            if isinstance(op, tensor.delta):
                self.oplist.append(op)
        self.n_del = len(self.oplist) - self.n_num
        for op in oplist:
            if isinstance(op, tensor.fermion):
                self.oplist.append(op)
        assert(len(self.oplist) == len(oplist))

    def nums(self):
        return opstring(self.oplist[:self.n_num])

    def deltas(self):
        return opstring(self.oplist[self.n_num:self.n_num+self.n_del])
    
    def nonfermions(self):
        return opstring(self.oplist[:self.n_num+self.n_del])

    def fermions(self):
        return opstring(self.oplist[self.n_num+self.n_del:])

    def add_indices(self, indices = None):
        if indices is None:
            indices = "ijklmnopq"
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
        return opstring(oplist)

    def dn(self):
        return np.sum(map(lambda op: op.dn(), self.oplist))

    def ds(self):
        return np.sum(map(lambda op: op.ds(), self.oplist))

    def conj(self):
        return opstring(map(lambda op: op.conj(), self.oplist[::-1]))

    def permute(self, i, j):
        assert(abs(i-j) == 1)
        terms = []
        oplist = deepcopy(self.oplist)
        delta = None
        if isinstance(oplist[i], tensor.fermion) and \
                isinstance(oplist[j], tensor.fermion):
            factor = -1
            if oplist[i].spin == oplist[j].spin and oplist[i].cre != oplist[j].cre:
                assert(oplist[i].idx is not None and oplist[j].idx is not None)
                delta = tensor.delta([oplist[i].idx[0], oplist[j].idx[0]])
        else:
            factor = 1
        oplist[i], oplist[j] = oplist[j], oplist[i]
        terms.append((factor, opstring(oplist)))
        if delta is not None:
            terms.append((1, opstring([delta] + oplist[:i] + oplist[j+1:])))
        return terms

    def __hash__(self):
        return hash(tuple([op.__hash__() for op in self.oplist]))

    def __eq__(self, other):
        if len(self.oplist) != len(other.oplist):
            return False
        else:
            return reduce(lambda a, b: a and b, \
                    map(lambda x, y: x == y, self.oplist, other.oplist), True)

    def __repr__(self):
        return "*".join(map(lambda op: op.__repr__(), self.oplist))

def rm_indices(ops):
    ops1 = deepcopy(ops)
    ops1.rm_indices()
    return ops1

def append(*opstrs):
    res = deepcopy(opstrs[0])
    for ops in opstrs[1:]:
        res.append(ops)
    return res

def equiv(ops1, ops2):
    return set(ops1.oplist) == set(ops2.oplist)

class terms(list):
    def __init__(self, terms):
        for factor, ops in terms:
            assert(isinstance(factor, (int, float)))
            assert(isinstance(ops, opstring))
        list.__init__(self, map(tuple, terms))

    def append(self, term):
        factor, ops = term
        assert(isinstance(factor, (int, float)))
        assert(isinstance(ops, opstring))
        list.append(self, tuple(term))

    def replace_indices(self, *pairs):
        return terms([(factor, ops.replace_indices(*pairs)) for (factor, ops) in self])

    def __iadd__(self, other):
        list.__iadd__(self, other)
        return self

    def __add__(self, other):
        return terms(list.__add__(self, other))

    def conj(self):
        return terms(map(lambda (factor, term): \
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
    ops = opstring([tensor.fermion(False, 'B'), tensor.fermion(False, 'B'), \
            tensor.fermion(False, 'A'), tensor.fermion(True, 'B')])
    ops.add_indices()
    print ops
