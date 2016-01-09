from libdmet.bitgen.basic import *
from libdmet.bitgen.merge import merge_fixed, eval_delta, eval_delta_expr, merged_to_sum
from libdmet.bitgen.pycode import *
import libdmet.utils.logger as log

def rm_cici(expr):
    return OpSum(filter(lambda (fac, ops): len(set(ops.fermions())) == \
            len(ops.fermions()), expr))

def eval_h_diag(kernel):
    g = merged_to_sum(eval_delta(merge_fixed(reduced(commutator(Ham, kernel)), \
            indices = "qp"), indices = "qp"))
    h_diag = get_expectations(merged_to_sum(eval_delta(merge_fixed(rm_cici(reduced(\
            commutator(g, kernel))), indices = "pq"), indices = "pq")))
    return h_diag

def eval_hessian(kernel1, kernel2):
    return get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced(
        0.5 * commutator(merged_to_sum(eval_delta(merge_fixed(reduced(
            commutator(Ham, kernel1)), indices = "pq"), indices = "pq")), kernel2) +
        0.5 * commutator(merged_to_sum(eval_delta(merge_fixed(reduced(
            commutator(Ham, kernel2)), indices = "rs"), indices = "rs")), kernel1)
    ), indices = "pqrs"), indices = "pqrs")))

def group_identity(hessian):
    # group according to identity operator
    empty = NumTensor("", "")
    hess_dict = {empty: OpSum([])}
    for fac, ops in hessian:
        found = False
        for i, op in enumerate(ops):
            if op.name == "I":
                if op in hess_dict:
                    hess_dict[op] += OpSum([(fac, OpProduct(ops[:i] + ops[i+1:]))])
                else:
                    hess_dict[op] = OpSum([(fac, OpProduct(ops[:i] + ops[i+1:]))])
                found = True
                break
        if not found:
            hess_dict[empty] += OpSum([(fac, ops)])
    return hess_dict

def pyGrad(expr, name, indices = "pq"):
    log.info("working on term %s", name)
    toSum = map(lambda (fac, ops): (fac, \
            contract_grad_term(ops, indices)), expr)
    expr, oplist = sumover(toSum)
    return define(expr, oplist, 2, name = name, final = True)

def pyH_diag(expr, name, indices = "pq"):
    log.info("working on term %s", name)
    toSum = map(lambda (fac, ops): (fac,) + \
            contract_h_diag_term(ops, indices), expr)
    idx = set(map(lambda (fac, ops, i): "".join(i), toSum))
    toSum1 = [(i, [(fac, ops) for (fac, ops, j) in toSum if "".join(j) == i]) \
            for i in idx]
    toSum1 = map(lambda (i, s): (i, sumover(s)), toSum1)
    toSum1 = dict(map(lambda (i, (expr, oplist)): (i, define(expr, oplist, len(i), \
            name = name + "_" + i, final = True)), toSum1))
    temp = []
    count = 1
    oplist = []
    if indices in toSum1:
        temp.append("@%02d" % count)
        oplist.append(toSum1[indices])
        count += 1
    if indices[0] in toSum1:
        temp.append("np.outer(@%02d, np.ones(@%02d.shape[0]))" % (count, count))
        oplist.append(toSum1[indices[0]])
        count += 1
    if indices[1] in toSum1:
        temp.append("np.outer(np.ones(@%02d.shape[0]), @%02d)" % (count, count))
        oplist.append(toSum1[indices[1]])
        count += 1
    if indices[0].upper() in toSum1:
        temp.append("np.diag(@%02d)" % count)
        oplist.append(toSum1[indices[0].upper()])
        count += 1
    expr = " + ".join(temp)
    return define(expr, oplist, 2, name = name, final = True)

def pyHessian1(expr_dict, t_term, name, indices = "pq"):
    # first order terms
    sumexpr = OpSum([])
    for key, val in expr_dict.items():
        if key.name == "I":
            sumexpr += eval_delta_expr(OpSum(OpProduct([val, Delta(key.idx)])) \
                    * t_term, indices = indices)
    if len(sumexpr) == 0:
        return
    toSum = map(lambda (fac, ops): (fac, \
            contract_hessian1(ops, indices)), sumexpr)
    expr, oplist = sumover(toSum)
    return define(expr, oplist, 2, name = name, final = True)

def pyHessian2(expr, name, indices = "pq"):
    log.info("working on term %s", name)
    toSum = map(lambda (fac, ops): (fac, \
            contract_hessian2(ops, indices)), expr)
    expr, oplist = sumover(toSum)
    return define(expr, oplist, 2, name = name, final = True)

def contract_grad_term(ops, indices = "pq"):
    if len(ops) == 2:
        left, right = None, None
        kwargs = {}
        if ops[0].nidx == ops[1].nidx:
            for mat in ops:
                if indices[0] in mat.idx and left is None:
                    left = reg(mat)
                elif indices[1] in mat.idx and right is None:
                    right = reg(mat)
                else:
                    raise Exception()
        else:
            for mat in ops:
                if indices[0] in mat.idx and indices[1] in mat.idx and left is None:
                    left = reg(mat)
                elif not indices[0] in mat.idx and not indices[1] in mat.idx and right is None:
                    right = reg(mat)
                else:
                    raise Exception
            kwargs["indices"] = indices
        c = contract(left, right, **kwargs)
        return c[0]
    elif len(ops) == 1:
        if "".join(ops[0].idx) == indices:
            return reg(ops[0])[0]
        else:
            ops1 = reg(ops[0])
            return define("@01.T", [ops1[0]], 2)

def contract_h_diag_term(ops, indices):
    if len(ops) == 1:
        mat = reg(ops[0])
        if indices[0] in mat[1] and indices[1] in mat[1]:
            matp = [i for (i,x) in enumerate(mat[1]) if x == indices[0]]
            matq = [i for (i,x) in enumerate(mat[1]) if x == indices[1]]
            assert(len(matp) == 2 and len(matq) == 2)
            mat1 = (define("@01.diagonal(0, %d, %d)" % tuple(matp), [mat[0]], 3), \
                        [x for (i,x) in enumerate(mat[1]) if not i in matp] + [indices[0]])
            mat1q = [i for (i,x) in enumerate(mat1[1]) if x == indices[1]]
            c = (define("@01.diagonal(0, %d, %d)" % tuple(mat1q), [mat1[0]], 2), \
                        [x for (i,x) in enumerate(mat1[1]) if not i in mat1q] + [indices[1]])
        else:
            if indices[0] in mat[1]:
                idx = indices[0]
            elif indices[1] in mat[1]:
                idx = indices[1]
            else:
                raise Exception
            matp = [i for (i,x) in enumerate(mat[1]) if x == idx]
            assert(len(matp) == 2 and len(mat[1]) == 2)
            c = (define("np.diag(@01)", [mat[0]], 1), idx)
    elif len(ops) == 2:
        if indices[0] in ops[0].idx and indices[0] in ops[1].idx and \
                indices[1] in ops[0].idx and indices[1] in ops[1].idx:
            assert(set(ops[0].idx) == set(ops[1].idx))
            c = contract(reg(ops[0]), reg(ops[1]), diag_idx = indices)
        elif indices[0] in "".join(ops.get_indices()) \
                and indices[1] in "".join(ops.get_indices()):
            left, right = map(reg, ops if indices[0] in ops[0].idx else ops[::-1])
            if len(left[1]) == len(right[1]) == 4:
                assert(left[1].count(indices[0]) == 2)
                assert(right[1].count(indices[1]) == 2)
                leftp = [i for (i,x) in enumerate(left[1]) if x == indices[0]]
                rightq = [i for (i,x) in enumerate(right[1]) if x == indices[1]]
                left1 = (define("@01.diagonal(0, %d, %d)" % tuple(leftp), [left[0]], 3), \
                        [x for (i,x) in enumerate(left[1]) if not i in leftp] + [indices[0]])
                right1 = (define("@01.diagonal(0, %d, %d)" % tuple(rightq), [right[0]], 3), \
                        [x for (i,x) in enumerate(right[1]) if not i in rightq] + [indices[1]])
                c = contract(left1, right1)
            elif len(left[1]) == len(right[1]) == 2:
                assert(left[1].count(indices[0]) == 2)
                assert(right[1].count(indices[1]) == 2)
                left1 = define("np.diag(@01)", [left[0]], 1)
                right1 = define("np.diag(@01)", [right[0]], 1)
                c = (define("np.outer(@01, @02)", [left1, right1], 2), "".join(indices))
            elif len(left[1]) == 4 and len(right[1]) == 2:
                if left[1].count(indices[0]) == 2 and left[1].count(indices[1]) == 1 and \
                        right[1].count(indices[1]) == 1:
                    idx1, idx2 = indices
                elif left[1].count(indices[1]) == 2 and left[1].count(indices[0]) == 1 and \
                        right[1].count(indices[0]) == 1:
                    idx2, idx1 = indices
                leftp = [i for (i,x) in enumerate(left[1]) if x == idx1]
                left1 = (define("@01.diagonal(0, %d, %d)" % tuple(leftp), [left[0]], 3), \
                        [x for (i,x) in enumerate(left[1]) if not i in leftp] + [idx1])
                expr = 'np.einsum("%s,%s->%s", @01, @02)' % \
                        ("".join(left1[1]), "".join(right[1]), "".join(indices))
                c = (define(expr, [left1[0], right[0]], 2), "".join(indices))
            else:
                raise Exception
        else:
            if set(ops[0].idx) == set(ops[1].idx):
                if indices[0] in ops[0].idx:
                    idx = indices[0]
                elif indices[1] in ops[0].idx:
                    idx = indices[1]
                else:
                    raise Exception()
                c = contract(reg(ops[0]), reg(ops[1]), diag_idx = idx)
            else:
                assert(indices[0] in ops[0].idx or indices[1] in ops[0].idx)
                assert(not indices[0] in ops[1].idx and not indices[1] in ops[1].idx)
                left, right = map(reg, ops)
                if indices[0] in ops[0].idx:
                    idx = indices[0]
                elif indices[1] in ops[0].idx:
                    idx = indices[1]
                leftp = [i for (i,x) in enumerate(left[1]) if x == idx]
                left1 = (define("@01.diagonal(0, %d, %d)" % tuple(leftp), [left[0]], 3), \
                        [x for (i,x) in enumerate(left[1]) if not i in leftp] + [idx])
                expr = 'np.einsum("%s,%s->%s", @01, @02)' % \
                        ("".join(left1[1]), "".join(right[1]), idx)
                c = (define(expr, [left1[0], right[0]], 1), idx)
    elif len(ops) == 3:
        found = False
        for i, mat in enumerate(ops):
            if mat.name == "I" and sorted(mat.idx) == sorted(indices):
                left, right = map(deepcopy, ops[:i] + ops[i+1:])
                left.replace_idx(*mat.idx)
                right.replace_idx(*mat.idx)
                left, right = reg(left), reg(right)
                c = contract(left, right, diag_idx = mat.idx[1])[0]
                c = (c, indices[0].upper())
                found = True
                break
        if not found:
            raise Exception()
    else:
        raise Exception()
    return c

def contract_hessian1(ops, indices):
    assert(len(ops) == 2)
    left, right = None, None
    for mat in ops:
        if indices[0] in mat.idx and left is None:
            left = reg(mat)
        elif indices[1] in mat.idx and right is None:
            right = reg(mat)
        else:
            raise Exception()
    return contract(left, right)[0]

def contract_hessian2(ops, indices):
    if len(ops) == 3:
        term_i, term_g, term_t = map(reg, ops)
        if indices[0] in term_i[1] and indices[1] in term_g[1]:
            if len(set(term_t[1]).intersection(term_g[1])) > 0:
                return contract(term_i, contract(term_t, term_g))[0]
            else:
                return contract(contract(term_i, term_t), term_g)[0]
        elif indices[1] in term_i[1] and indices[0] in term_g[1]:
            if len(set(term_t[1]).intersection(term_g[1])) > 0:
                c = contract(term_i, contract(term_t, term_g))[0]
            else:
                c = contract(contract(term_i, term_t), term_g)[0]
            return define("@01.T", [c], 2)
        elif indices[0] in term_i[1] and indices[1] in term_i[1] and len(term_g[1]) == 2:
            c = contract(term_i, contract(term_t, term_g))
            if c[1][0] == indices[0] and c[1][1] == indices[1]:
                return c[0]
            elif c[1][0] == indices[1] and c[1][1] == indices[0]:
                return define("@01.T", [c[0]], 2)
            else:
                raise Exception()
        else:
            raise Exception()
    elif len(ops) == 2:
        term_i, term_t = map(reg, ops)
        c = contract(term_i, term_t)
        if c[1][0] == indices[0] and c[1][1] == indices[1]:
            return c[0]
        elif c[1][0] == indices[1] and c[1][1] == indices[0]:
            return define("@01.T", [c[0]], 2)
        else:
            raise Exception()
    else:
        raise Exception()

if __name__ == "__main__":
    log.section("Orbital gradient and hessian for BCS-CASSCF," \
            " used for atomic basis variation")

    Ham = H1(False, True) + H2(True, False)
    print Ham
    assert(0)
    kernel_a = OpSum(C('A','p')) * OpSum(D('A', 'q')) - OpSum(C('A','q')) * OpSum(D('A', 'p'))
    kernel_b = OpSum(C('B','p')) * OpSum(D('B', 'q')) - OpSum(C('B','q')) * OpSum(D('B', 'p'))
    kernel_d = OpSum(C('A','p')) * OpSum(C('B', 'q')) + OpSum(D('A','p')) * OpSum(D('B', 'q'))

    # gradient
    # evaluate the formula
    grad_a = get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced( \
            commutator(Ham, kernel_a)), indices = "pq"), indices = "pq")))
    grad_b = get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced( \
            commutator(Ham, kernel_b)), indices = "pq"), indices = "pq")))
    grad_d = get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced( \
            commutator(Ham, kernel_d)), indices = "pq"), indices = "pq")))

    # evaluate code
    pyGrad(grad_a, "gorb_a")
    pyGrad(grad_b, "gorb_b")
    pyGrad(grad_d, "gorb_d")

    # hessian diagonal
    h_diag_a = eval_h_diag(kernel_a)
    h_diag_b = eval_h_diag(kernel_b)
    h_diag_d = eval_h_diag(kernel_d)

    # evaluate code
    pyH_diag(h_diag_a, "h_diag_a")
    pyH_diag(h_diag_b, "h_diag_b")
    pyH_diag(h_diag_d, "h_diag_d")

    # hessian
    kernel_a1 = OpSum(C('A','r')) * OpSum(D('A', 's')) - OpSum(C('A','s')) * OpSum(D('A', 'r'))
    kernel_b1 = OpSum(C('B','r')) * OpSum(D('B', 's')) - OpSum(C('B','s')) * OpSum(D('B', 'r'))
    kernel_d1 = OpSum(C('A','r')) * OpSum(C('B', 's')) + OpSum(D('A','r')) * OpSum(D('B', 's'))

    # evaluate hessian
    log.info("hessian_aa")
    hess_aa = group_identity(eval_hessian(kernel_a, kernel_a1))
    log.info("hessian_ab")
    hess_ab = group_identity(eval_hessian(kernel_a, kernel_b1))
    log.info("hessian_ad")
    hess_ad = group_identity(eval_hessian(kernel_a, kernel_d1))
    log.info("hessian_bb")
    hess_bb = group_identity(eval_hessian(kernel_b, kernel_b1))
    log.info("hessian_bd")
    hess_bd = group_identity(eval_hessian(kernel_b, kernel_d1))
    log.info("hessian_dd")
    hess_dd = group_identity(eval_hessian(kernel_d, kernel_d1))

    # first evaluate code for 1st order terms
    for h, name in [(hess_aa, "h_aa"), (hess_ab, "h_ab"), (hess_ad, "h_ad"), \
            (hess_bb, "h_bb"), (hess_bd, "h_bd"), (hess_dd, "h_dd")]:
        log.info("evaluate 1st order terms in %s", name)
        for key in h:
            if key.name == "I":
                index = "".join(set("pqrs").difference(key.idx))
                temp = pyGrad(h[key], name + "_" + "".join(key.idx), \
                        indices = index)
                h[key] = NumTensor(temp.name, index, symm = temp.symm)

    # now act the non-identical part of hessian on displacement t
    t_a = OpSum(NumTensor("t_a", "rs", symm = IdxAntisymm()))
    t_b = OpSum(NumTensor("t_b", "rs", symm = IdxAntisymm()))
    t_d = OpSum(NumTensor("t_d", "rs"))
    t_a1 = OpSum(NumTensor("t_a", "pq", symm = IdxAntisymm()))
    t_b1 = OpSum(NumTensor("t_b", "pq", symm = IdxAntisymm()))

    log.info("Acting hessian on displacement")
    hx_aa = hess_aa[NumTensor("", "")] * t_a
    hx_ab = hess_ab[NumTensor("", "")] * t_b
    hx_ad = hess_ad[NumTensor("", "")] * t_d
    hx_ba = hess_ab[NumTensor("", "")] * t_a1
    hx_bb = hess_bb[NumTensor("", "")] * t_b
    hx_bd = hess_bd[NumTensor("", "")] * t_d
    hx_da = hess_ad[NumTensor("", "")] * t_a1
    hx_db = hess_bd[NumTensor("", "")] * t_b1
    hx_dd = hess_dd[NumTensor("", "")] * t_d

    # now evaluate code for hx
    # first order terms
    pyHessian1(hess_aa, t_a, "hx_aa1")
    pyHessian1(hess_ab, t_b, "hx_ab1")
    pyHessian1(hess_ad, t_d, "hx_ad1")
    pyHessian1(hess_ab, t_a1, "hx_ba1", indices = "rs")
    pyHessian1(hess_bb, t_b, "hx_bb1")
    pyHessian1(hess_bd, t_d, "hx_bd1")
    pyHessian1(hess_ad, t_a1, "hx_da1", indices = "rs")
    pyHessian1(hess_bd, t_b1, "hx_db1", indices = "rs")
    pyHessian1(hess_dd, t_d, "hx_dd1")
    # then second order terms
    pyHessian2(hx_aa, "hx_aa2")
    pyHessian2(hx_ab, "hx_ab2")
    pyHessian2(hx_ad, "hx_ad2")
    pyHessian2(hx_ba, "hx_ba2", indices = "rs")
    pyHessian2(hx_bb, "hx_bb2")
    pyHessian2(hx_bd, "hx_bd2")
    pyHessian2(hx_da, "hx_da2", indices = "rs")
    pyHessian2(hx_db, "hx_db2", indices = "rs")
    pyHessian2(hx_dd, "hx_dd2")

    # finally write down the code
    with open("g_hop_atomic_raw.py", "w") as f:
        f.write(dump() + "\n")
