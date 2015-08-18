from tensor import *
from expression import *
from ham import C, D
from copy import deepcopy
import libdmet.utils.logger as log

# we define the reduced form of operator products here
_reduced = {}
_reduced[0] = [OpProduct([])]
_reduced[2] = [
    C('A') * D('A'),
    C('B') * D('B'),
    C('A') * C('B'),
    D('B') * D('A')
]
_reduced[4] = [
    # ccdd
    C('A') * C('A') * D('A') * D('A'),
    C('B') * C('B') * D('B') * D('B'),
    C('A') * C('B') * D('B') * D('A'),
    # cccd
    C('A') * C('A') * C('B') * D('A'),
    C('B') * C('B') * C('A') * D('B'),
    # cccc
    C('A') * C('A') * C('B') * C('B'),
    # cddd
    C('A') * D('B') * D('A') * D('A'),
    C('B') * D('A') * D('B') * D('B'),
    # dddd
    D('B') * D('B') * D('A') * D('A'),
]

def get_reduced_type(ops):
    ops1 = rm_indices(ops.fermions())
    for r in _reduced[len(ops1)]:
        if equiv(r, ops1):
            return deepcopy(r)
    raise Exception("Reduced form for this operator %s not defined", ops1)

def is_reduced(ops):
    ops1 = rm_indices(ops.fermions())
    return ops1 in _reduced[len(ops1)]

def get_permutations(ops, target_ops):
    ops1 = rm_indices(ops.fermions())
    shift = len(ops.nonfermions())
    assert(equiv(ops1, target_ops))
    # get ordering first
    mask = [False] * len(ops1)
    order = []
    # if target is ABCCD, ops is CBADC, then we get order = [2,1,0,4,3]
    for i in range(len(ops1)):
        for j in range(len(ops1)):
            if mask[j]:
                continue
            if ops1[i] == target_ops[j]:
                order.append(j)
                mask[j] = True
                break
    # the permutation for [2,1,0,4,3] is
    # (1,2), (0,1), (1,2), (3,4)
    permutations = []
    for val in range(len(ops1)):
        idx = order.index(val)
        while idx > val:
            permutations.append((shift+idx-1, shift+idx))
            order[idx-1], order[idx] = order[idx], order[idx-1]
            idx -= 1
    return permutations

def get_reduce_permutations(ops):
    return get_permutations(ops, get_reduced_type(ops))

def reduced(obj):
    if isinstance(obj, OpSum):
        return reduce(OpSum.__add__, map(lambda (fac, ops): \
                fac * _reduced_OpProduct(ops), obj), OpSum([]))
    elif isinstance(obj, OpProduct):
        return _reduced_OpProduct(obj)
    else:
        raise Exception("Unknown type to reduce")

def _reduced_OpProduct(ops):
    permutations = get_reduce_permutations(ops)
    ops1 = deepcopy(ops)
    factor = 1.
    other = OpSum([])
    for p in permutations:
          raw = ops1.permute(*p)
          other += factor * reduced(raw[1:])
          fac, ops1 = raw[0]
          factor *= fac
    return OpSum([(factor, ops1)]) + other

if __name__ == "__main__":
    log.section("Show operator reduction functions here")
    for key, val in _reduced.items():
        log.result("%d operator product, %d operators\n%s", key, len(val), \
                "\n".join(map(str, val)))

    from ham import *
    for fac, ops in ccddterm():
        log.result("Is term %s in reduced form? %s", ops, is_reduced(ops))

    s = Coeff("w", "ijkl") * D('A','i') * C('A','j') * D('B','k') * C('B','l')
    log.result("Fully reducing %s requires permutations\n%s", s, \
            get_reduce_permutations(s))
    log.result("Reduction result\n%s", reduced(s))
