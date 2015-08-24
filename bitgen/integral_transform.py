import basic
from pycode import *
from substitute import FermionSub, Substitution
from merge import merge, eval_delta
import libdmet.utils.logger as log

def transform(sub, Ham):
    # first replace the terms
    H = sub.replace(Ham)
    # then reduce to primary Operator products
    H = basic.reduced(H)
    # merge terms: classify according to fermion operators
    H = merge(H, indices = "pqrs")
    # evaluate delta functions
    H = eval_delta(H)
    return H

def _addto(dictionary, key, content):
    if key in dictionary:
        dictionary[key].append(content)
    else:
        dictionary[key] = [content]

def generate_code(H, indices = "pqrs"):
    H0fromH0 = {}
    H0fromH1 = {}
    H0fromH2 = {}
    H1fromH1 = {}
    H1fromH2 = {}
    H2fromH2 = {}
    for key, expr in H.items():
        if len(key) == 0: # H0 terms
            for fac, term in expr:
                if len(term) == 1: # H0fromH0
                    _addto(H0fromH0, key, (fac, term))
                elif len(term) == 3:
                    _addto(H0fromH1, key, (fac, term))
                elif len(term) == 5:
                    _addto(H0fromH2, key, (fac, term))
                else:
                    raise Exception()
        elif len(key) == 2:
            for fac, term in expr:
                if len(term) == 3:
                    _addto(H1fromH1, key, (fac, term))
                elif len(term) == 5:
                    _addto(H1fromH2, key, (fac, term))
                else:
                    raise Exception()
        elif len(key) == 4:
            for fac, term in expr:
                if len(term) == 5:
                    _addto(H2fromH2, key, (fac, term))
                else:
                    raise Exception()

    if len(H0fromH0) != 0:
        pyH0_H0 = pyH0fromH0(H0fromH0, indices)
    if len(H0fromH1) != 0:
        pyH0_H1 = pyH0fromH1(H0fromH1, indices)
    if len(H0fromH2) != 0:
        pyH0_H2 = pyH0fromH2(H0fromH2, indices)
    if len(H1fromH1) != 0:
        pyH1_H1 = pyH1fromH1(H1fromH1, indices)
    if len(H1fromH2) != 0:
        pyH1_H2 = pyH1fromH2(H1fromH2, indices)
    if len(H2fromH2) != 0:
        pyH2_H2 = pyH2fromH2(H2fromH2, indices)
    return dump()

def pyH0fromH0(H0_H0, indices = "pqrs"):
    log.result("transform H0 to H0")
    assert(len(H0_H0) == 1)
    terms = H0_H0[H0_H0.keys()[0]]
    toSum = map(lambda (fac, ops): \
            (fac, reg(ops)[0]), terms)
    expr, oplist = sumover(toSum)
    define(expr, oplist, 0, name = "H0_H0", final = True)

def contract_pi_ij_jp(ops, indices = "pqrs"):
    assert(len(ops) == 3)
    left, center, right = None, None, None
    for mat in ops: 
        if set(mat.idx).intersection(indices) == set([]) \
                and center is None:
            center = reg(mat)
        elif indices[0] in mat.idx and left is None:
            left = reg(mat)
        elif indices[0] in mat.idx and right is None:
            right = reg(mat)
        else:
            raise Exception()
    return contract(contract(left, center), right)[0]

def pyH0fromH1(H0_H1, indices = "pqrs"):
    log.result("transform H1 to H0")
    assert(len(H0_H1) == 1)
    terms = H0_H1[H0_H1.keys()[0]]
    toSum = map(lambda (fac, ops): \
            (fac, contract_pi_ij_jp(ops, indices)), terms)
    expr, oplist = sumover(toSum)
    define(expr, oplist, 0, name = "H0_H1", final = True)

def contract_ijkl_ip_jp_kq_lq(ops, indices = "pqrs"):
    assert(len(ops) == 5)
    ipjp, center, kqlq = None, None, None
    for mat in ops:
        if mat.nidx == 4 and center is None:
            center = reg(mat)
    for i in indices:
        temp = [reg(mat) for mat in ops if \
                mat.nidx == 2 and i in mat.idx]
        if len(temp) == 2:
            if ipjp is None:
                ipjp = temp
            elif kqlq is None:
                kqlq = temp
            else:
                raise Exception
    left = contract(*ipjp)
    right = contract(*kqlq)
    if center[1][0] in right[1]:
        left, right = right, left
    return contract(contract(left, center), right, rightT = True)[0]

def pyH0fromH2(H0_H2, indices = "pqrs"):
    log.result("transform H2 to H0")
    assert(len(H0_H2) == 1)
    terms = H0_H2[H0_H2.keys()[0]]
    toSum = map(lambda (fac, ops): \
            (fac, contract_ijkl_ip_jp_kq_lq(ops, indices)), terms)
    expr, oplist = sumover(toSum)
    define(expr, oplist, 0, name = "H0_H2", final = True)

def contract_ij_ip_jq(ops, indices = "pqrs"):
    assert(len(ops) == 3)
    left, center, right = None, None, None
    for mat in ops:
        if set(mat.idx).intersection(indices) == set([]) \
                and center is None:
            center = reg(mat)
        elif indices[0] in mat.idx and left is None:
            left = reg(mat)
        elif indices[1] in mat.idx and right is None:
            right = reg(mat)
        else:
            raise Exception
    return contract(contract(left, center), right)[0]

def pyH1fromH1(H1_H1, indices = "pqrs"):
    log.result("transform H1 to H1")
    for key in H1_H1:
        if key.dn() < 0:
            continue
        if basic.rm_indices(key) == basic.C('A') * basic.C('B'):
            name = "H1D_H1"
        elif basic.rm_indices(key) == basic.C('A') * basic.D('A'):
            name = "H1A_H1"
        elif basic.rm_indices(key) == basic.C('B') * basic.D('B'):
            name = "H1B_H1"
        toSum = map(lambda (fac, ops): \
                (fac, contract_ij_ip_jq(ops, indices)), H1_H1[key])
        expr, oplist = sumover(toSum)
        define(expr, oplist, 2, name = name, final = True)

def contract_ijkl_ip_jq_kr_lr(ops, indices = "pqrs"):
    assert(len(ops) == 5)
    center, krlr, r1, r2 = None, None, None, None
    for mat in ops:
        if mat.nidx == 4 and center is None:
            center = reg(mat)
    for i in indices:
        temp = [reg(mat) for mat in ops if \
                mat.nidx == 2 and i in mat.idx]
        if len(temp) == 2 and krlr is None:
            krlr = temp
        elif i == indices[0] and r1 is None:
            r1 = temp[0]
        elif i == indices[1] and r2 is None:
            r2 = temp[0]
    left = contract(*krlr)
    cidx1 = contract(left, center, indices_only = True)[1]
    cidx2 = contract(left, center, indices_only = True)[0]
    if cidx1[0] + cidx2[1] > 3 or (cidx1[0] + cidx2[1] == 3 and \
            cidx1[1] + cidx2[0] > 3):
        temp = contract(center, left)
    else:
        temp = contract(left, center)
    return contract(contract(r1, temp), r2)[0]

def pyH1fromH2(H1_H2, indices = "pqrs"):
    log.result("transform H2 to H1")
    for key in H1_H2:
        if key.dn() < 0:
            continue
        if basic.rm_indices(key) == basic.C('A') * basic.C('B'):
            name = "H1D_H2"
        elif basic.rm_indices(key) == basic.C('A') * basic.D('A'):
            name = "H1A_H2"
        elif basic.rm_indices(key) == basic.C('B') * basic.D('B'):
            name = "H1B_H2"
        toSum = map(lambda (fac, ops): \
                (fac, contract_ijkl_ip_jq_kr_lr(ops, indices)), H1_H2[key])
        expr, oplist = sumover(toSum)
        define(expr, oplist, 2, name = name, final = True) 

def contract_ijkl_ip_jq_kr_ls(ops, indices = "pqrs"):
    assert(len(ops) == 5)
    center, p, q, r, s = None, None, None, None, None
    for mat in ops:
        if mat.nidx == 4 and center is None:
            center = reg(mat)
        elif mat.nidx == 2:
            if indices[0] in mat.idx and p is None:
                p = reg(mat)
            elif indices[1] in mat.idx and q is None:
                q = reg(mat)
            elif indices[2] in mat.idx and r is None:
                r = reg(mat)
            elif indices[3] in mat.idx and s is None:
                s = reg(mat)
    temp = contract(contract(contract(q, contract(p, center)), r), s)
    op, idx = temp
    order = tuple(map(lambda i: indices.index(i), idx))
    if order in op.symm._symm:
        return op
    elif order in op.symm._antisymm:
        expr = "(-@01)"
        return define(expr, [op], len(indices))
    else:
        expr = "np.transpose(@01, %s)" % str(order)
        return define(expr, [op], len(indices))

def defineH2symmetrized(expr, oplist, name):
    if name == "H2wAB_H2":
        define(expr, oplist, 4, name = name, final = "True")
    elif name in ["H2wA_H2", "H2wB_H2"]:
        op1 = define(expr, oplist, 4)
        expr1 = "@01 + np.transpose(@01, (2,3,0,1))"
        # since we have a factor of 0.5 in the definition
        define(expr1, [op1], 4, name = name, final = True)
    elif name in ["H2yA_H2", "H2yB_H2"]:
        op1 = define(expr, oplist, 4, name)
        expr1 = "@01 - np.transpose(@01, (1,0,2,3))"
        # since we have a factor of 0.5 in the definition
        define(expr1, [op1], 4, name = name, final = True)
    elif name == "H2x_H2":
        op1 = define(expr, oplist, 4, name)
        expr1 = "@01 - np.transpose(@01, (1,0,2,3))"
        op2 = define(expr1, [op1], 4, name)
        expr2 = "@01 - np.transpose(@01, (0,1,3,2))"
        # a factor of 1/4
        define(expr2, [op2], 4, name = name, final = True)
    else:
        raise Exception

def pyH2fromH2(H2_H2, indices = "pqrs"):
    log.result("transform H2 to H2")
    for key in H2_H2:
        if key.dn() < 0:
            continue
        if key.dn() == 4:
            name = "H2x_H2"
        elif key.dn() == 2 and basic.rm_indices(key)[0] == basic.C('A')[0]:
            name = "H2yA_H2"
        elif key.dn() == 2 and basic.rm_indices(key)[0] == basic.C('B')[0]:
            name = "H2yB_H2"
        elif basic.rm_indices(key) == basic.C('A') * basic.C('A') * \
                basic.D('A') * basic.D('A'):
            name = "H2wA_H2"
        elif basic.rm_indices(key) == basic.C('B') * basic.C('B') * \
                basic.D('B') * basic.D('B'):
            name = "H2wB_H2"
        elif basic.rm_indices(key) == basic.C('A') * basic.C('B') * \
                basic.D('B') * basic.D('A'):
            name = "H2wAB_H2"
        if key.dn() == 0:
            _indices = "".join(map(lambda i: indices[i], [0,3,1,2]))
        else:
            _indices = indices
        log.info("working on term %s", name)
        toSum = map(lambda (fac, ops): (fac, \
                contract_ijkl_ip_jq_kr_ls(ops, _indices)), H2_H2[key])
        expr, oplist = sumover(toSum)
        defineH2symmetrized(expr, oplist, name)
