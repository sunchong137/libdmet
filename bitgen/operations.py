from expression import *
from tensor import NumTensor
from tensor_symm import *
from reduction import _reduced, reduced
from ham import C, D

def commutator(A, B):
    return A * B - B * A

def anticommutator(A, B):
    return A * B + B * A

def expectation(ops):
    assert(len(ops.fermions()) == len(ops))
    ops1 = rm_indices(ops)    
    if len(ops) == 0:
        return OpProduct([])
    elif len(ops) == 2:
        if ops1 == C('A') * D('A'):
            return OpProduct(NumTensor("rho_A", ops.get_indices(), symm = IdxSymm()))
        elif ops1 == C('B') * D('B'):
            return OpProduct(NumTensor("rho_B", ops.get_indices(), symm = IdxSymm()))
        elif ops1 == C('A') * C('B'):
            return OpProduct(NumTensor("kappa_BA", ops.get_indices()[::-1], symm = IdxNoSymm(2)))
        elif ops1 == D('B') * D('A'):
            return OpProduct(NumTensor("kappa_BA", ops.get_indices(), symm = IdxNoSymm(2)))
        else:
            assert(0)
    elif len(ops) == 4:
        i,j,k,l = ops.get_indices()
        if ops1 == C('A') * C('A') * D('A') * D('A'):
            return OpProduct(NumTensor("Gamma_aa", i+l+j+k, symm = Idx2PdmAASymm()))
        elif ops1 == C('B') * C('B') * D('B') * D('B'):
            return OpProduct(NumTensor("Gamma_bb", i+l+j+k, symm = Idx2PdmAASymm()))
        elif ops1 == C('A') * C('B') * D('B') * D('A'):
            return OpProduct(NumTensor("Gamma_ab", i+l+j+k, symm = Idx2PdmABSymm()))
        elif ops1 == C('A') * C('A') * C('B') * D('A'):
            return OpProduct(NumTensor("Gamma_2a", i+j+k+l, symm = Idx2FoldAntisymm()))
        elif ops1 == C('B') * C('B') * C('A') * D('B'):
            return OpProduct(NumTensor("Gamma_2b", i+j+k+l, symm = Idx2FoldAntisymm()))
        elif ops1 == C('A') * C('A') * C('B') * C('B'):
            return OpProduct(NumTensor("Gamma_4", i+j+k+l, symm = Idx4FoldAntisymm()))
        elif ops1 == C('A') * D('B') * D('A') * D('A'):
            return OpProduct(NumTensor("Gamma_2a", l+k+j+i, symm = Idx2FoldAntisymm()))
        elif ops1 == C('B') * D('A') * D('B') * D('B'):
            return OpProduct(NumTensor("Gamma_2b", l+k+j+i, symm = Idx2FoldAntisymm()))
        elif ops1 == D('B') * D('B') * D('A') * D('A'):
            return OpProduct(NumTensor("Gamma_4", l+k+j+i, symm = Idx4FoldAntisymm()))
        else:
            assert(0)
    else:
        assert(0)
