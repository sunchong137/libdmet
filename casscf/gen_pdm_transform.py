from libdmet.bitgen.basic import *
from libdmet.bitgen.substitute import FermionSub, Substitution
from libdmet.bitgen.merge import eval_delta_expr
from libdmet.bitgen.pycode import *
from libdmet.bitgen.integral_transform import contract_ijkl_ip_jq_kr_ls
import libdmet.utils.logger as log

sub = Substitution([
    FermionSub(Fermion(True, 'A','i'), \
            OpSum(Coeff("vc_A",'ip') * C('A','p')) + \
            OpSum(Coeff("uc_A",'ip') * D('B','p')) + \
            OpSum(Coeff("va_A",'ia') * C('A','a')) + \
            OpSum(Coeff("ua_A",'ia') * D('B','a'))),
    FermionSub(Fermion(True, 'B','i'), \
            OpSum(Coeff("vc_B",'ip') * C('B','p')) + \
            OpSum(Coeff("uc_B",'ip') * D('A','p')) + \
            OpSum(Coeff("va_B",'ia') * C('B','a')) + \
            OpSum(Coeff("ua_B",'ia') * D('A','a'))),
])

def get_parity(permutation):
    import numpy as np
    import numpy.linalg as la
    l = len(permutation)
    d = np.zeros((l, l), dtype = int)
    for i, r in enumerate(permutation):
        d[i,r] = 1
    return la.det(d)

def divide_c_a(ops, cidx, aidx):
    c = []
    a = []
    for op in ops:
        assert(isinstance(op, Fermion))
        if op.idx[0] in cidx and not op.idx[0] in aidx:
            c.append(op)
        elif op.idx[0] in aidx and not op.idx[0] in cidx:
            a.append(op)
        else:
            raise Exception()
    return get_parity([ops.oplist.index(op) for op in c+a]), \
            OpProduct(c), OpProduct(a)

def nonzero_exp(ops):
    if ops.dn() != 0:
        return False
    for i in range(len(ops)):
        if OpProduct(ops[i:]).dn() > 0:
            return False
        if OpProduct(ops[i:]).dn() == 0 and OpProduct(ops[i:]).ds() != 0:
            return False
    return True

def core_expr(ops):
    assert(len(ops.fermions()) == len(ops))
    ops1 = rm_indices(ops)
    if len(ops) == 0:
        return OpSum(OpProduct([]))
    elif len(ops) == 2:
        if ops1 == C('A') * D('A') or C('B') * D('B'):
            return OpSum(Delta(ops.get_indices()))
        else:
            assert(0)
    elif len(ops) == 4:
        i,j,k,l = ops.get_indices()
        if ops1 == C('A') * C('A') * D('A') * D('A') or \
                ops1 == C('B') * C('B') * D('B') * D('B'):
            return OpSum(OpProduct([Delta(i+l), Delta(j+k)])) - \
                    OpSum(OpProduct([Delta(i+k), Delta(j+l)]))
        elif ops1 == C('A') * C('B') * D('B') * D('A'):
            return OpSum(OpProduct([Delta(i+l), Delta(j+k)]))
        elif ops1 in [
            C('A') * D('A') * C('A') * D('A'),
            C('B') * D('B') * C('B') * D('B'),
            C('A') * D('A') * C('B') * D('B'),
            C('B') * D('B') * C('A') * D('A')
        ]:
            return OpSum(OpProduct([Delta(i+j), Delta(k+l)]))
        else:
            assert(0)

def contract_pdm_core(ops, indices, cidx):
    if len(ops) == 0:
        return None
    elif len(ops) == 2:
        left, right = None, None
        for mat in ops:
            if indices[0] in mat.idx and left is None:
                left = reg(mat)
            elif indices[1] in mat.idx and right is None:
                right = reg(mat)
            else:
                raise Exception
        return contract(left, right)
    elif len(ops) == 4:
        ops1 = filter(lambda s: len(s) > 0, \
                map(lambda i: [op for op in ops if i in op.idx], cidx))
        (left1, left2), (right1, right2) = ops1
        leftidx = [i for i in left1.idx + left2.idx if i in indices]
        if indices.index(leftidx[0]) < indices.index(leftidx[1]):
            left =contract(reg(left1), reg(left2))
        else:
            left =contract(reg(left2), reg(left1))
        rightidx = [i for i in right1.idx + right2.idx if i in indices]
        if indices.index(rightidx[0]) < indices.index(rightidx[1]):
            right = contract(reg(right1), reg(right2))
        else:
            right = contract(reg(right2), reg(right1))
        expr = 'np.einsum("%s,%s->%s", @01, @02)' % \
                ("".join(left[1]), "".join(right[1]), "".join(indices))
        return (define(expr, [left[0], right[0]], len(indices)), indices)
    else:
        raise Exception

def contract_pdm_active(ops, indices, aidx):
    if len(ops) == 0:
        return None
    elif len(ops) == 2:
        left, right = None, None
        for mat in ops:
            if indices[0] in mat.idx and left is None:
                left = reg(mat)
            elif indices[1] in mat.idx and right is None:
                right = reg(mat)
            else:
                raise Exception
        return contract(left, right)
    elif len(ops) == 3:
        left, center, right = None, None, None
        for mat in ops:
            if indices[0] in mat.idx and left is None:
                left = reg(mat)
            elif indices[1] in mat.idx and right is None:
                right = reg(mat)
            elif len(set(mat.idx).intersection(indices)) == 0 and \
                    center is None:
                center = reg(mat)
            else:
                raise Exception
        return contract(contract(left, center), right)
    elif len(ops) == 4:
        ops1 = filter(lambda s: len(s) > 0, \
                map(lambda i: [op for op in ops if i in op.idx], aidx))
        (left1, left2), (right1, right2) = ops1
        leftidx = [i for i in left1.idx + left2.idx if i in indices]
        if indices.index(leftidx[0]) < indices.index(leftidx[1]):
            left =contract(reg(left1), reg(left2))
        else:
            left =contract(reg(left2), reg(left1))
        rightidx = [i for i in right1.idx + right2.idx if i in indices]
        if indices.index(rightidx[0]) < indices.index(rightidx[1]):
            right = contract(reg(right1), reg(right2))
        else:
            right = contract(reg(right2), reg(right1))
        expr = 'np.einsum("%s,%s->%s", @01, @02)' % \
                ("".join(left[1]), "".join(right[1]), "".join(indices))
        return (define(expr, [left[0], right[0]], len(indices)), indices)
    elif len(ops) == 5:
        for mat in ops:
            if len(mat.idx) == 4:
                return (contract_ijkl_ip_jq_kr_ls(ops, indices), indices)
        # otherwise onepdm is used
        center = None
        c = []
        s = []
        for mat in ops:
            if mat.idx[0] in aidx and mat.idx[1] in aidx \
                    and center is None:
                center = mat
        for mat in ops:
            if mat != center:
                if mat.idx[1] in center.idx:
                    c.append(mat)
                else:
                    s.append(mat)
        assert(len(c) == len(s) == 2)
        center = reg(center)
        left, right = reg(c[0]), reg(c[1])
        if indices.index(left[1][0]) < indices.index(right[1][0]):
            temp1 = contract(contract(left, center), right)
        else:
            temp1 = contract(contract(right, center), left)
        left, right = reg(s[0]), reg(s[1])
        if indices.index(left[1][0]) < indices.index(right[1][0]):
            temp2 = contract(left, right)
        else:
            temp2 = contract(right, left)
        expr = 'np.einsum("%s,%s->%s", @01, @02)' % \
                ("".join(temp1[1]), "".join(temp2[1]), "".join(indices))
        return (define(expr, [temp1[0], temp2[0]], len(indices)), indices)
    else:
        raise Exception

def contract_pdm_term(ops, indices, cidx, aidx):
    core = []
    core_idx = []
    active = []
    active_idx = []
    for mat in ops:
        assert(len(set(mat.idx).intersection(cidx)) == 0 or \
                len(set(mat.idx).intersection(aidx)) == 0)
        if len(set(mat.idx).intersection(cidx)):
            core.append(mat)
            core_idx += list(mat.idx)
        elif len(set(mat.idx).intersection(aidx)):
            active.append(mat)
            active_idx += list(mat.idx)
        else:
            raise Exception

    cpart = contract_pdm_core(core, \
            [i for i in indices if i in core_idx], cidx)
    apart = contract_pdm_active(active, \
            [i for i in indices if i in active_idx], aidx)
    if cpart is None and apart is not None:
        return apart[0]
    elif apart is None and cpart is not None:
        return cpart[0]
    else:
        expr = 'np.einsum("%s,%s->%s", @01, @02)' % \
                ("".join(cpart[1]), "".join(apart[1]), "".join(indices))
        c = define(expr, [cpart[0], apart[0]], len(indices))
        return c

def dump_naive(expr, name, indices):
    code = name + " = " + " + \\\n        ".join(map(lambda (fac, ops): \
            "%f * np.einsum('%s->%s', %s)" % \
            (fac, ",".join(ops.get_indices()), "".join(indices), \
            ", ".join(map(lambda op: op.name, ops))), expr))
    return code

def pyPDM(O, name, indices, cidx = "pqrs", aidx = "abcd"):
    log.info("working on term %s", name)
    O1 = sub.replace(O)
    O2 = OpSum([])
    for fac, ops in O1:
        factor, core, active = divide_c_a(ops.fermions(), cidx, aidx)
        if nonzero_exp(core):
            O2 += OpSum([(fac*factor, ops.nonfermions())]) * \
                    core_expr(core) * OpSum(active)
    
    O3 = eval_delta_expr(OpSum(map(lambda (fac, ops): (fac, \
            ops.nonfermions() * expectation(ops.fermions())), \
            reduced(O2))))
    toSum = map(lambda (fac, ops): (fac, \
        contract_pdm_term(ops, indices, cidx, aidx)), O3)
    expr, oplist = sumover(toSum)
    return define(expr, oplist, len(indices), name = name, final = True)

if __name__ == "__main__":
    cdA = OpSum(C('A', 'i') * D('A', 'j'))
    cdB = OpSum(C('B', 'i') * D('B', 'j'))
    cc  = OpSum(C('A', 'i') * C('B', 'j'))
    ccddA  = OpSum(C('A', 'i') * C('A', 'j') * D('A', 'k') * D('A', 'l'))
    ccddB  = OpSum(C('B', 'i') * C('B', 'j') * D('B', 'k') * D('B', 'l'))
    ccddAB = OpSum(C('A', 'i') * C('B', 'j') * D('B', 'k') * D('A', 'l'))
    cccdA  = OpSum(C('A', 'i') * C('A', 'j') * C('B', 'k') * D('A', 'l'))
    cccdB  = OpSum(C('B', 'i') * C('B', 'j') * C('A', 'k') * D('B', 'l'))
    cccc   = OpSum(C('A', 'i') * C('A', 'j') * C('B', 'k') * C('B', 'l'))
    
    pyPDM(cdA, "rho_a", indices = "ij")
    pyPDM(cdB, "rho_b", indices = "ij")
    pyPDM(cc, "kappa_ba_T", indices = "ij")
    pyPDM(ccddA, "gamma0_a", indices = "iljk")
    pyPDM(ccddB, "gamma0_b", indices = "iljk")
    pyPDM(ccddAB, "gamma0_ab", indices = "iljk")
    pyPDM(cccdA, "gamma2_a", indices = "ijkl")
    pyPDM(cccdB, "gamma2_b", indices = "ijkl")
    pyPDM(cccc, "gamma4", indices = "ijkl")
    
    with open("pdm_transform1.py", "w") as f:
        f.write("import numpy as np\n\n")
        f.write("""def cas_pdm_transform(va_A, va_B, ua_A, ua_B, vc_A, vc_B, uc_A, uc_B, \\
        rho_A, rho_B, kappa_BA, Gamma_aa, Gamma_bb, Gamma_ab, \\
        Gamma_2a, Gamma_2b, Gamma_4):
    """)
        f.write(dump().replace("\n", "\n    ") + "\n    ")
        f.write("""return (np.asarray([rho_a, rho_b]), -kappa_ba_T), \\
            (np.asarray([gamma0_a, gamma0_b, gamma0_ab]), \\
            np.asarray([gamma2_a, gamma2_b]), gamma4)\n""")
