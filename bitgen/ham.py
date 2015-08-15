from tensor import *
from expression import *
import tensor_symm as symm
import libdmet.utils.logger as log

# this is a toolbox module to define commonly used terms in Hamiltonian

def Coeff(name, idx, symmetry = None):
    return OpProduct(NumTensor(name, idx, symm = symmetry))

def C(spin, idx = None):
    return OpProduct(Fermion(True, spin, idx))

def D(spin, idx = None):
    return OpProduct(Fermion(False, spin, idx))

def h0term(restricted = False):
    return OpSum(Coeff('H0', ''))

def cdterm(restricted = False):
    if restricted:
        return OpSum(Coeff("h", 'ij', symm.IdxSymm()) * C('A', 'i') * D('A', 'j')) + \
                OpSum(Coeff("h", 'ij', symm.IdxSymm()) * C('B', 'i') * D('B', 'j'))
    else:
        return OpSum(Coeff("h_A", 'ij', symm.IdxSymm()) * C('A', 'i') * D('A', 'j')) + \
                OpSum(Coeff("h_B", 'ij', symm.IdxSymm()) * C('B', 'i') * D('B', 'j'))

def ccterm(restricted = False):
    if restricted:
        D = Coeff("Delta", 'ij', symm.IdxSymm())
    else:
        D = Coeff("Delta", 'ij')
    cc = OpSum(D * C('A', 'i') * C('B', 'j'))
    return cc + cc.conj()

def ccddterm(restricted = False):
    if restricted:
        W = Coeff('w', 'iljk', symm.Idx8FoldSymm())
        return 0.5 * OpSum(W * C('A','i') * C('A','j') * D('A','k') * D('A','l')) + \
                0.5 * OpSum(W * C('B','i') * C('B','j') * D('B','k') * D('B','l')) + \
                OpSum(W * C('A','i') * C('B','j') * D('B','k') * D('A','l'))
    else:
        return 0.5 * OpSum(Coeff('w_A', 'iljk', symm.Idx8FoldSymm()) * C('A','i') * \
                C('A','j') * D('A','k') * D('A','l')) + \
                0.5 * OpSum(Coeff('w_B', 'iljk', symm.Idx8FoldSymm()) * C('B','i') * \
                C('B','j') * D('B','k') * D('B','l')) + \
                OpSum(Coeff('w_AB', 'iljk', symm.Idx4FoldSymm()) * C('A','i') * C('B','j') * \
                D('B','k') * D('A','l')) # not sure about the symmetry


def cccdterm(restricted):
    if restricted:
        return 0.5 * (OpSum(Coeff('y', 'ijkl', symm.Idx2FoldAntisymm()) * C('A','i') * \
                C('A','j') * C('B','k') * D('A','l')) - \
                OpSum(Coeff('y', 'ijkl', symm.Idx2FoldAntisymm()) * C('B','i') * \
                C('B','j') * C('A','k') * D('B','l')))
    else:
        return 0.5 * (OpSum(Coeff('y_A', 'ijkl', symm.Idx2FoldAntisymm()) * C('A','i') * \
                C('A','j') * C('B','k') * D('A','l')) + \
                OpSum(Coeff('y_B', 'ijkl', symm.Idx2FoldAntisymm()) * C('B','i') * \
                C('B','j') * C('A','k') * D('B','l')))

def ccccterm(restricted):
    if restricted:
        X = Coeff('x', 'ijkl', symm.Idx8FoldAntisymm())
    else:
        X = Coeff('x', 'ijkl', symm.Idx4FoldAntisymm())
    return 0.25 * OpSum(X * C('A','i') * C('A','j') * C('B','k') * D('B','l'))

if __name__ == "__main__":
    log.section("ham.py defines components of Hamiltonian")
    log.result("Cre %s %s, Anni %s %s", C('A', 'i'), C('B', 'j'), D('A', 'k'), D('B', 'l'))
    for restricted in [True, False]:
        log.section("Test Hamiltonian terms (spin-restrection: %s)", restricted)
        log.result("spin restriction = %s", restricted)
        log.result("H0    %s", h0term())
        log.result("cd    %s", cdterm(restricted))
        log.result("cc    %s", ccterm(restricted))
        log.result("ccdd  %s", ccddterm(restricted))
        log.result("cccd  %s", cccdterm(restricted))
        log.result("cccc  %s", ccccterm(restricted))
