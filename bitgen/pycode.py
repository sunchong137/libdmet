import basic
from copy import deepcopy

def pyH0fromH0(H0_H0, indices = "pqrs"):
    assert(len(H0_H0) == 1)
    terms = H0_H0[H0_H0.keys()[0]]
    return "H0_H0 =" + " + ".join(map(lambda (fac, term): \
            str(fac) + "*" + term[0].name ,terms))

def T(trans):
    if trans:
        return ".T"
    else:
        return ""

def Trace(term, indices = "pqrs"):
    assert(len(term) == 3)
    left, center, right = None, None, None
    transl, transc, transr = False, False, False
    for mat in term:
        if mat.name.startswith(("h", "D", "w", "v", "x")) \
                and center is None:
            center = mat
        elif indices[0] in mat.idx and left is None:
            left = mat
        elif indices[0] in mat.idx and right is None:
            right = mat
        else:
            raise Exception()
    assert(len(left.idx) == 2 and len(right.idx) == 2 \
            and len(center.idx) == 2)
    transl = indices[0] != left.idx[0]
    transr = indices[0] != right.idx[1]
    i1 = left.idx[0] if transl else left.idx[1]
    i2 = right.idx[1] if transr else right.idx[0]
    assert(set([i1, i2]) == set(center.idx))
    transc = (not (1,0) in center.symm._symm) and i1 != center.idx[0]
    text = "np.trace(mdot(%s%s, %s%s, %s%s))"
    return  text % (left.name, T(transl), center.name, \
            T(transc), right.name, T(transr))

def pyH0fromH1(H0_H1, indices = "pqrs"):
    assert(len(H0_H1) == 1)
    terms = H0_H1[H0_H1.keys()[0]]
    textterms = {}
    for fac, term in terms:
        text = Trace(term, indices)[9:-1]
        if text in textterms:
            textterms[text] += fac
        else:
            textterms[text] = fac
    return "H0_H1 = np.trace(%s)" % (" + ".join(map(lambda (text, fac): \
            str(fac) + "*" + text, textterms.items())))

def contract2op(ops, replaced, prodtensor):
    opl, opr = replaced
    del ops.oplist[ops.oplist.index(opl)]
    del ops.oplist[ops.oplist.index(opr)]
    i = list(set(opl.idx) - set(opr.idx))[0]
    j = list(set(opr.idx) - set(opl.idx))[0]
    if opl.name == opr.name and opl.idx.index(i) == opr.idx.index(j):
        symm = basic.IdxSymm()
    else:
        symm = basic.IdxNoSymm(2)
    return ops * basic.OpProduct(basic.NumTensor(prodtensor, \
            [i,j], symm = symm))

def doubleTrace(ops, indices = "pqrs"):
    assert(len(ops) == 3)
    left, center, right = None, None, None
    for mat in ops:
        if len(mat.idx) == 4 and center is None:
            center = mat
    for mat in ops:
        if len(mat.idx) == 2 and center.idx[0] in mat.idx:
            left = mat
        elif len(mat.idx) == 2:
            right = mat
    idx1 = map(center.idx.index, left.idx)
    if idx1[0] > idx1[1]:
        idx1 = ((1,0), (idx1[1], idx1[0]))
    else:
        idx1 = ((0,1), tuple(idx1))
    idx2 = map(center.idx.index, right.idx)
    if idx2[0] > idx2[1]:
        idx2 = ((1,0), (0,1))
    else:
        idx2 = ((0,1), (0,1))
    fmt = "np.tensordot(%s, np.tensordot(%s, %s, axes = %s), axes = %s)"
    return fmt % (right.name, left.name, center.name, idx1, idx2)

def pyH0fromH2(H0_H2, indices = "pqrs"):
    assert(len(H0_H2) == 1)
    terms = H0_H2[H0_H2.keys()[0]]
    terms_precont = []
    precontracts = set([])
    for fac, term in terms:
        newterm = deepcopy(term)
        for idx in indices:
            prod = [mat for mat in term if idx in mat.idx]
            if len(prod) == 2:
                name = prod[0].name + "__" + prod[1].name
                idx1 = prod[0].idx.index(idx)
                idx2 = prod[1].idx.index(idx)
                precontracts.add((name, idx1, idx2))
                newterm = contract2op(newterm, prod, name)
        terms_precont.append((fac, newterm))
    code = []
    fmt = "%s = np.dot(%s%s, %s%s)"
    for pre in precontracts:
        matl, matr = pre[0].split("__")
        transl = pre[1] == 0
        transr = pre[2] == 1
        code.append(fmt % (pre[0], matl, T(transl), matr, T(transr)))
    # FIXME further optimization by combining terms is possible
    code.append("H0_H2 = " + " + \\\n        ".join(map(lambda (fac, ops): \
            str(fac) + "*" + doubleTrace(ops, indices), \
            terms_precont)))
    return "\n".join(code)

def prod3op(term, indices = "pqrs"):
    assert(len(term) == 3)
    left, center, right = None, None, None
    transl, transc, transr = False, False, False
    for mat in term:
        if mat.name.startswith(("h", "D", "w", "v", "x")) \
                and center is None:
            center = mat
        elif indices[0] in mat.idx and left is None:
            left = mat
        elif indices[1] in mat.idx and right is None:
            right = mat
        else:
            raise Exception()
    assert(len(left.idx) == 2 and len(right.idx) == 2 \
            and len(center.idx) == 2)
    transl = indices[0] != left.idx[0]
    transr = indices[1] != right.idx[1]
    i1 = left.idx[0] if transl else left.idx[1]
    i2 = right.idx[1] if transr else right.idx[0]
    assert(set([i1, i2]) == set(center.idx))
    transc = (not (1,0) in center.symm._symm) and i1 != center.idx[0]
    text = "mdot(%s%s, %s%s, %s%s)"
    return text % (left.name, T(transl), center.name, \
            T(transc), right.name, T(transr))

def pyH1fromH1(H1_H1, indices = "pqrs"):
    code = []
    for key in H1_H1:
        if key.dn() < 0:
            continue
        if basic.rm_indices(key) == basic.C('A') * basic.C('B'):
            res = "H1D_H1"
        elif basic.rm_indices(key) == basic.C('A') * basic.D('A'):
            res = "H1A_H1"
        elif basic.rm_indices(key) == basic.C('B') * basic.D('B'):
            res = "H1B_H1"
        code.append(res + " = " + " + ".join(map(lambda (fac, ops): \
                str(fac) + "*" + prod3op(ops, indices), H1_H1[key])))
    return "\n".join(code)
