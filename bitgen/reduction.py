from tensor import delta, fermion
import opstring
from copy import deepcopy
import libdmet.utils.logger as log

# one can redefine primary in another file to overwrite this
primary = {}
primary[0] = set([opstring.opstring([])])
primary[2] = map(opstring.opstring, [
    [fermion(True, 'A'), fermion(False, 'A')],
    [fermion(True, 'B'), fermion(False, 'B')],
    [fermion(True, 'A'), fermion(True, 'B')],
])
primary[2] += map(lambda ops: ops.conj(), primary[2])
primary[2] = set(primary[2])

primary[4] = map(opstring.opstring, [
    [fermion(True, 'A'), fermion(True, 'A'), fermion(True, 'B'), fermion(True, 'B')],
    [fermion(True, 'A'), fermion(True, 'A'), fermion(True, 'B'), fermion(False, 'A')],
    [fermion(True, 'B'), fermion(True, 'B'), fermion(True, 'A'), fermion(False, 'B')],
    [fermion(True, 'A'), fermion(True, 'A'), fermion(False, 'A'), fermion(False, 'A')],
    [fermion(True, 'B'), fermion(True, 'B'), fermion(False, 'B'), fermion(False, 'B')],
    [fermion(True, 'A'), fermion(True, 'B'), fermion(False, 'B'), fermion(False, 'A')],
])
primary[4] += map(lambda ops: ops.conj(), primary[4])
primary[4] = set(primary[4])

def find_primary(ops):
    # primary representation for the operator
    # require only fermion in the string
    assert(ops.n_num == 0 and ops.n_del == 0)
    ops1 = opstring.rm_indices(ops)
    lenstr = len(ops1.oplist)
    for p in primary[lenstr]:
        if opstring.equiv(ops1, p):
            return p
    raise Exception("Cannot find primary permutaion for operator string %s" % ops)

def isprimary(ops):
    ops1 = opstring.rm_indices(ops)
    lenstr = len(ops1.oplist)
    return ops1 in primary[lenstr]

def find_permutations(ops, target):
    ops1 = opstring.rm_indices(ops)
    assert(opstring.equiv(ops1, target))
    # get order first
    lenstr = len(ops1.oplist)
    mask = [False] * lenstr
    order = []
    # if target is ABCCD, ops is CBADC
    # we obtain order = [2,1,0,4,3]
    for i in range(lenstr):
        for j in range(lenstr):
            if mask[j]:
                continue
            if ops1.oplist[i] == target.oplist[j]:
                order.append(j)
                mask[j] = True
                break
    # the permutations should be
    # [2,1,0,4,3]
    # (1,2) => [2,0,1,4,3]
    # (0,1) => [0,2,1,4,3]
    # (1,2) => [0,1,2,4,3]
    # (3,4) => [0,1,2,3,4]
    perms = []
    for val in range(len(order)):
        idx = order.index(val)
        while idx > val:
            perms.append((idx-1, idx))
            order[idx-1], order[idx] = order[idx], order[idx-1]
            idx -= 1
    return perms

def reduced(ops):
    target = find_primary(ops)
    permutations = find_permutations(ops, target)
    ops1 = deepcopy(ops)
    f = opstring.terms([])
    factor = 1.
    for p in permutations:
        res = ops1.permute(p[0], p[1])
        for term in res[1:]:
            t_factor, t_ops = term
            if isprimary(t_ops.fermions()):
                f.append((factor*t_factor, t_ops))
            else:
                t_f = reduced(t_ops.fermions())
                for t_term in t_f:
                    tt_factor, tt_ops = t_term
                    f.append((factor*t_factor*tt_factor, \
                            opstring.append(t_ops.nums(), t_ops.deltas(), tt_ops)))
        term = res[0]
        factor *= term[0]
        ops1 = term[1]
    f.append((factor, ops1))
    return f

def scale_terms(factor, expression):
    return opstring.terms([(f1*factor, ops1) for (f1, ops1) in expression])


def tensor_scale_terms(ops, expression):
    return opstring.terms([(f1, opstring.append(ops, ops1)) for (f1, ops1) in expression])

def reduced_terms(expression):
    #for factor, ops in expression:
    #    print factor, ops
    #    print ops.fermions(), reduced(ops.fermions())
    return reduce(opstring.terms.__add__, map(lambda (factor, ops): \
            scale_terms(factor, tensor_scale_terms(ops.nonfermions(), \
            reduced(ops.fermions()))), expression))

if __name__ == "__main__":
    import itertools as it
    log.verbose = "INFO"
    log.section("test operator reduction, with all S_z=0 two-electron operators")
    base_all = [(True, 'A'), (False, 'A'), (True, 'B'), (False, 'B')]
    for string in it.product(base_all, repeat = 4):
        ops = opstring.opstring(map(lambda args: fermion(*args), string))
        if ops.ds() == 0:
            ops.add_indices()
            rops = reduced(ops)
            log.result("Operator: %s\nreduced to %s", ops, reduced(ops))
