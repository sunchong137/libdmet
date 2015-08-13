import hamiltonian as h
from op_reduction import scale_terms, reduced_terms, find_primary
from opstring import *
from tensor import *
import itertools as it
from copy import deepcopy

def sub_op(ops, op, *exprs):
    #print "sub_op", ops, op, exprs
    assert(isinstance(ops, opstring))
    assert(isinstance(op, BaseTensor))
    for expr in exprs:
        assert(isinstance(expr, terms))
    found = []
    for i, op1 in enumerate(ops.oplist):
        if type(op) == type(op1) and op == op1:
            found.append(i)
    assert(len(found) <= len(exprs))
    # now replace recursively
    if len(found) == 0:
        return terms([(1, ops)])
    else:
        # replace the first ones
        replaced = terms([])
        for term in exprs[0]:
            factor, ops1 = term
            replaced.append((factor, \
                    opstring(ops.oplist[:found[0]] + ops1.oplist \
                    + ops.oplist[found[0]+1:])))
        # now continue replace in new term
        replaced = terms(list(it.chain.from_iterable(map(lambda (factor, ops1): \
                scale_terms(factor, sub_op(ops1, op, *exprs[1:])), replaced))))
    return replaced

def sub_term(op_terms, op, *exprs):
    return reduce(terms.__add__, map(lambda (factor, ops): \
            scale_terms(factor, sub_op(ops, op, *exprs)), op_terms), terms([]))

def gen_replace(op, pattern, idx_list, idx_candidate, conj = True):
    idx_candidate = set(idx_candidate + [idx_list])
    replace_list = []
    for idx_group in idx_candidate:
        assert(len(idx_group) == len(idx_list))
        terms = pattern.replace_indices(*zip(idx_list, idx_group))
        op1 = deepcopy(op)
        for old, new in zip(idx_list, idx_group):
            op1.replace_idx(old, new)
        replace_list.append((op1, terms))
        if conj:
            replace_list.append((op1.conj(), terms.conj()))
    return replace_list

if __name__ == "__main__":
    H = h.cdterm()# + h.ccterm() + h.ccddterm() + h.cccdterm() + h.ccccterm()
    print "original Hamiltonian: "
    for term in H:
        print term
    replace_list = gen_replace(op = fermion(True, 'A', 'i'),
        pattern = terms([
            (1, opstring([num_tensor('VA','ip'), fermion(True, 'A', 'p')])),
            (1, opstring([num_tensor('UA','ip'), fermion(False, 'B', 'p')])),
        ]),
        idx_list = ('i', 'p'), idx_candidate = [('j','q'), ('k','r'), ('l','s')], 
        conj = True)
    replace_list += gen_replace(op = fermion(True, 'B', 'i'),
        pattern = terms([
            (1, opstring([num_tensor('VB','ip'), fermion(True, 'B', 'p')])),
            (1, opstring([num_tensor('UB','ip'), fermion(False, 'A', 'p')])),
        ]),
        idx_list = ('i', 'p'), idx_candidate = [('j','q'), ('k','r'), ('l','s')], 
        conj = True)
    print "Do the following transformation"
    for op, rep in replace_list:
        print op, " ==> ", rep
    print "We obtain"
    for old, new in replace_list:
        H = sub_term(H, old, new)
    for term in H:
        print term
    print "-------------------"
    print "After normal ordering"
    Hreduced = reduced_terms(H)
    for term in Hreduced:
        print term
    print "-------------------"

    coef_dict = {}
    for term in Hreduced:
        factor, ops = term
        coef, fops = ops.nonfermions(), ops.fermions()
        p = find_primary(fops)
        if not p in coef_dict:
            coef_dict[p] = terms([])
        ridx = zip(map(lambda op: op.idx[0], fops.oplist), "abcd")
        coef1 = coef.replace_indices(*ridx)
        ridx1 = zip("abcd", "pqrs")[:len(ridx)]
        coef1 = coef1.replace_indices(*ridx1)
        coef_dict[p].append((factor, coef1))
    for key, val in coef_dict.items():
        print key
        print val
