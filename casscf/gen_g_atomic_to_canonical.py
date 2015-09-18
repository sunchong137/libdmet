from libdmet.bitgen.basic import *
from libdmet.bitgen.pycode import *
import libdmet.utils.logger as log
from libdmet.bitgen.substitute import FermionSub, Substitution
from libdmet.bitgen.merge import eval_delta, merge
from libdmet.bitgen.integral_transform import contract_ij_ip_jq, contract_pi_ij_jp

def pyGradTransform(expr, name, indices = "pq"):
    log.info("working on %s", name)
    toSum = map(lambda (fac, ops): \
            (fac, contract_ij_ip_jq(ops, indices)), expr)
    expr, oplist = sumover(toSum)
    return define(expr, oplist, 2, name = name, final = True)

def pyGradTransform0(expr, name, indices = "pq"):
    log.info("working on %s", name)
    toSum = map(lambda (fac, ops): \
            (fac, contract_pi_ij_jp(ops, indices)), expr)
    expr, oplist = sumover(toSum)
    return define(expr, oplist, 0, name = name, final = True)

if __name__ == "__main__":
    K = OpSum(NumTensor("g_A", "ij", symm = IdxSymm())) * \
            OpSum(C('A', 'i')) * OpSum(D('A', 'j')) + \
            OpSum(NumTensor("g_B", "ij", symm = IdxSymm())) * \
            OpSum(C('B', 'i')) * OpSum(D('B', 'j')) + \
            OpSum(NumTensor("g_D", "ij")) * \
            (OpSum(C('A', 'i')) * OpSum(C('B', 'j')) + \
            OpSum(D('A', 'i')) * OpSum(D('B', 'j')))
    
    sub = Substitution([
        FermionSub(Fermion(True, 'A','i'), \
                OpSum(Coeff("v_A",'ip') * C('A','p')) + \
                OpSum(Coeff("u_A",'ip') * D('B','p'))),
        FermionSub(Fermion(True, 'B','i'), \
                OpSum(Coeff("v_B",'ip') * C('B','p')) + \
                OpSum(Coeff("u_B",'ip') * D('A','p'))),
    ])
    K1 = eval_delta(merge(reduced(sub.replace(K)), indices = "pq"))
    for key in K1:
        if rm_indices(key) == C('A') * C('B'):
            pyGradTransform(K1[key], "g_d", indices = key.get_indices())
        elif rm_indices(key) == C('A') * D('A'):
            pyGradTransform(K1[key], "g_a", indices = key.get_indices())
        elif rm_indices(key) == C('B') * D('B'):
            pyGradTransform(K1[key], "g_b", indices = key.get_indices())
        #elif rm_indices(key) == OpProduct([]):
        #    pyGradTransform0(K1[key], "g_diag_0")
    print dump()

