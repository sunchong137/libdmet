import basic
import libdmet.utils.logger as log
import copy

def compute_rep_pairs(_indices, _indices_old):
    if set(_indices_old) == set(_indices):
        return zip(_indices_old, _indices)
    else:
        pairs = zip(_indices_old, _indices)
        additional = zip(set(_indices) - set(_indices_old), \
                set(_indices_old) - set(_indices))
        return pairs + additional


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

def eval_delta(merged_expr):
    for key, val in merged_expr.items():
        for i, (fac, ops) in enumerate(val):
            assert(len(ops.fermions()) == 0)
            while len(ops.deltas()) > 0:
                # use the knowledge that delta is after num_tensor
                ridx = sorted(ops[-1].idx)[::-1]
                assert(len(set(ridx).intersection(key.get_indices())) == 0)
                ops = basic.OpProduct(ops[:-1]).replace_indices(ridx)
            val[i] = (fac, ops)
    return merged_expr

if __name__ == "__main__":
    _eval_deltas(basic.OpProduct([basic.Delta("ij"), basic.Delta("jk")]))
