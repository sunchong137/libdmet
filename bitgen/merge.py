import basic
import libdmet.utils.logger as log
import itertools as it
import copy
import numpy as np
import numpy.linalg as la

def compute_rep_pairs(_indices, _indices_old):
    if set(_indices_old) == set(_indices):
        return zip(_indices_old, _indices)
    else:
        pairs = zip(_indices_old, _indices)
        additional = zip(set(_indices) - set(_indices_old), \
                set(_indices_old) - set(_indices))
        return pairs + additional

def get_parity(idx1, idx2):
    assert(sorted(idx1) == sorted(idx2))
    assert(len(set(idx1)) == len(idx1))
    if len(idx1) == 0:
        return 1
    m = np.zeros((len(idx1), len(idx2)))
    for i, c in enumerate(idx1):
        m[i, idx2.index(c)] = 1
    return int(la.det(m))

def equiv_idx(redtype, indices):
    redclass = {}
    for i, op in enumerate(redtype):
        if op in redclass:
            redclass[op].append(i)
        else:
            redclass[op] = [i]
    _eqlist = sorted(map(lambda key: redclass[key], redclass))
    eq_indices = []
    for i in it.product(*map(lambda l: it.permutations(l), _eqlist)):
        indices1 = "".join([indices[i] for i in it.chain.from_iterable(i)])
        eq_indices.append((indices1, get_parity(indices, indices1)))
    return eq_indices

def merge_fixed(expr, indices = "pq"):
    merged = {}
    for fac, ops in expr:
        t = basic.get_reduced_type(ops)
        _indices_old = "".join(ops.fermions().get_indices())
        found = False
        for (indices, parity) in equiv_idx(t, _indices_old):
            if basic.add_indices(t, indices) in merged:
                merged[basic.add_indices(t, indices)] += \
                        basic.OpSum([(fac*parity, ops.nonfermions())])
                found = True
                break
        if not found:
            merged[ops.fermions()] = basic.OpSum([(fac, ops.nonfermions())])
    for key in merged:
        merged[key] = merge_num(merged[key])
    return merged

def merge_num(expr):
    merged = {}
    for fac, ops in expr:
        if ops in merged:
            merged[ops] += fac
        else:
            merged[ops] = fac
    return basic.OpSum([(fac, ops) for (ops, fac) in merged.items() if fac != 0])

def merged_to_sum(expr_dict):
    return reduce(basic.OpSum.__add__, [val * basic.OpSum(key) for key, val \
            in expr_dict.items()])

def merge(expr, indices = "pqrs"):
    merged = {}
    for fac, ops in expr:
        t = basic.get_reduced_type(ops)
        _indices_old = "".join(ops.fermions().get_indices())
        log.eassert(len(set(_indices_old)) == len(_indices_old), \
                "assumed fermions indices are all different")
        if t.dn() < 0:
            _indices = indices[:len(t)][::-1]
        else:
            _indices = indices[:len(t)]
        t.add_indices(_indices)
        rep_pairs = compute_rep_pairs(_indices, _indices_old)
        if t in merged:
            merged[t] += basic.OpSum([(fac, ops.nonfermions().\
                    replace_indices(*rep_pairs))])
        else:
            merged[t] = basic.OpSum([(fac, ops.nonfermions().\
                    replace_indices(*rep_pairs))])
    return merged

def eval_delta(merged_expr, indices = ""):
    for key in merged_expr:
        merged_expr[key] = eval_delta_expr(merged_expr[key], \
                indices = indices + "".join(key.get_indices()))
    return merged_expr

def eval_delta_expr(expr, indices = ""):
    for i, (fac, ops) in enumerate(expr):
        assert(len(ops.fermions()) == 0)
        while len(ops.deltas()) > 0:
            # use the knowledge that delta is after num_tensor
            ridx = sorted(ops[-1].idx)[::-1]
            if ridx[0] in indices:
                if ridx[1] in indices:
                    ops = basic.OpProduct(ops[:-1] + \
                            [basic.NumTensor("I", ridx, symm = basic.IdxIdentity())])
                    continue
                else:
                    ridx = ridx[::-1]
            ops = basic.OpProduct(ops[:-1]).replace_indices(ridx)
        expr[i] = (fac, ops)
    return expr


if __name__ == "__main__":
    _eval_deltas(basic.OpProduct([basic.Delta("ij"), basic.Delta("jk")]))
