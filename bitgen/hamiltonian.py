from tensor import *
from opstring import *
import libdmet.utils.logger as log

def h0term(restricted = False):
    return terms([(1, opstring([num_tensor('h0', '')]))])

def cdterm(restricted = False):
    if restricted:
        return terms([
            (1, opstring([num_tensor('h', 'ij'), \
                fermion(True, 'A', 'i'), fermion(False, 'A', 'j')])), 
            (1, opstring([num_tensor('h', 'ij'), \
                fermion(True, 'B', 'i'), fermion(False, 'B', 'j')])), 
        ])
    else:
        return terms([
            (1, opstring([num_tensor('h_A', 'ij'), \
                fermion(True, 'A', 'i'), fermion(False, 'A', 'j')])), 
            (1, opstring([num_tensor('h_B', 'ij'), \
                fermion(True, 'B', 'i'), fermion(False, 'B', 'j')])), 
        ])

def ccterm(restricted = False):
    cc = opstring([num_tensor('Delta', 'ij'), \
                fermion(True, 'A', 'i'), fermion(True, 'B', 'j')])
    return terms([(1, cc), (1, cc.conj())])

def ccddterm(restricted = False):
    if restricted:
        return terms([
            (0.5, opstring([num_tensor('w', 'iljk'), fermion(True, 'A', 'i'), \
                    fermion(True, 'A', 'j'), fermion(False, 'A', 'k'), fermion(False, 'A', 'l')])),
            (0.5, opstring([num_tensor('w', 'iljk'), fermion(True, 'B', 'i'), \
                    fermion(True, 'B', 'j'), fermion(False, 'B', 'k'), fermion(False, 'B', 'l')])),
            (1, opstring([num_tensor('w', 'iljk'), fermion(True, 'A', 'i'), \
                    fermion(True, 'B', 'j'), fermion(False, 'B', 'k'), fermion(False, 'A', 'l')])),
        ])
    else:
        return terms([
            (0.5, opstring([num_tensor('w_A', 'iljk'), fermion(True, 'A', 'i'), \
                    fermion(True, 'A', 'j'), fermion(False, 'A', 'k'), fermion(False, 'A', 'l')])),
            (0.5, opstring([num_tensor('w_B', 'iljk'), fermion(True, 'B', 'i'), \
                    fermion(True, 'B', 'j'), fermion(False, 'B', 'k'), fermion(False, 'B', 'l')])),
            (1, opstring([num_tensor('w_AB', 'iljk'), fermion(True, 'A', 'i'), \
                    fermion(True, 'B', 'j'), fermion(False, 'B', 'k'), fermion(False, 'A', 'l')])),
        ])

def cccdterm(restricted = False):
    if restricted:
        cccd = terms([
            (0.5, opstring([num_tensor('y', 'ijkl'), fermion(True, 'A', 'i'), \
                    fermion(True, 'A', 'j'), fermion(True, 'B', 'k'), fermion(False, 'A', 'l')])),
            (-0.5, opstring([num_tensor('y', 'ijkl'), fermion(True, 'B', 'i'), \
                    fermion(True, 'B', 'j'), fermion(True, 'A', 'k'), fermion(False, 'B', 'l')])),
        ])
    else:
        cccd = terms([
            (0.5, opstring([num_tensor('y_A', 'ijkl'), fermion(True, 'A', 'i'), \
                    fermion(True, 'A', 'j'), fermion(True, 'B', 'k'), fermion(False, 'A', 'l')])),
            (0.5, opstring([num_tensor('y_B', 'ijkl'), fermion(True, 'B', 'i'), \
                    fermion(True, 'B', 'j'), fermion(True, 'A', 'k'), fermion(False, 'B', 'l')])),
        ])
    return cccd + cccd.conj()

def ccccterm(restricted = False):
    cccc = opstring([num_tensor('x', 'ijkl'), fermion(True, 'A', 'i'), fermion(True, 'A', 'j'), \
            fermion(True, 'B', 'k'), fermion(True, 'B', 'l')])
    return terms([(0.25, cccc), (0.25, cccc.conj())])

def buildHam(restricted, h0 = False, cd = False, cc = False, \
        ccdd = False, cccd = False, cccc = False):
    Ham = terms([])
    if h0:
        Ham += h0term(restricted)
    if cd:
        Ham += cdterm(restricted)
    if cc:
        Ham += ccterm(restricted)
    if ccdd:
        Ham += ccddterm(restricted)
    if cccd:
        Ham += cccdterm(restricted)
    if cccc:
        Ham += ccccterm(restricted)
    return Ham

def getHam(restricted, dn):
    if dn == 0:
        return buildHam(restricted, h0 = True, cd = True, ccdd = True)
    elif dn == 2:
        return buildHam(restricted, h0 = True, cd = True, cc = True, ccdd = True)
    elif dn == 4:
        return buildHam(restricted, h0 = True, cd = True, cc = True, ccdd = True, \
                cccd = True, cccc = True)
    else:
        raise Exception


if __name__ == "__main__":
    restricted = False
    log.section("Test Hamiltonian terms")
    log.result("spin restriction = %s", restricted)
    log.result("H0:\n%s", h0term(restricted))
    log.result("cd:\n%s", cdterm(restricted))
    log.result("cc:\n%s", ccterm(restricted))
    log.result("ccdd:\n%s", ccddterm(restricted))
    log.result("cccd:\n%s", cccdterm(restricted))
    log.result("cccc:\n%s", ccccterm(restricted))
    log.result("unrestricted full Hamiltonian:\n%s", getHam(False, 4))
    

