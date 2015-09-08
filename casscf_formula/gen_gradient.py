from libdmet.bitgen.basic import *
from libdmet.bitgen.merge import merge_fixed, eval_delta, merged_to_sum
from libdmet.bitgen.pycode import *

log.section("Orbital gradient for BCS-CASSCF")

Ham = H1(False, True) + H2(False, True)

kernel_aa = OpSum(C('A','p')) * OpSum(D('A', 'q')) - OpSum(C('A','q')) * OpSum(D('A', 'p'))
kernel_bb = OpSum(C('B','p')) * OpSum(D('B', 'q')) - OpSum(C('B','q')) * OpSum(D('B', 'p'))
kernel_ab = OpSum(C('A','p')) * OpSum(C('B', 'q')) + OpSum(D('A','p')) * OpSum(D('B', 'q'))

def get_expectations(expr):
    return OpSum(map(lambda (fac, ops): (fac, ops.nonfermions() * \
            expectation(ops.fermions())), expr))

grad_aa = get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced( \
        commutator(Ham, kernel_aa)), indices = "pq"), indices = "pq")))
grad_bb = get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced( \
        commutator(Ham, kernel_bb)), indices = "pq"), indices = "pq")))
grad_ab = get_expectations(merged_to_sum(eval_delta(merge_fixed(reduced( \
        commutator(Ham, kernel_ab)), indices = "pq"), indices = "pq")))

def pyGrad(expr, name, indices = "pq"):
    log.info("working on term %s", name)
    toSum = map(lambda (fac, ops): (fac, \
            contract_grad_term(ops, indices)), expr)
    expr, oplist = sumover(toSum)
    define(expr, oplist, 2, name = name, final = True)

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

pyGrad(grad_aa, "grad_aa")
pyGrad(grad_bb, "grad_bb")
pyGrad(grad_ab, "grad_ab")

with open("gradient.py", "w") as f:
    f.write("import numpy as np\n\n")
    f.write("""def grad(h_A, h_B, D, w_A, w_B, w_AB, y_A, y_B, x, \\
            rho_A, rho_B, kappa_BA, Gamma_aa, Gamma_bb, Gamma_ab, \\
            Gamma_2a, Gamma_2b, Gamma_4):
    """)    
    f.write(dump().replace("\n", "\n    ") + "\n")
    f.write("""
    return grad_aa, grad_bb, grad_ab""")
