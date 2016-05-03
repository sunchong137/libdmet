import libdmet.utils.logger as log
from libdmet.bitgen import basic
from libdmet.bitgen import merge
import itertools as it
import numpy as np

log.section("Compute terms in Fock matrix for general HFB calculations")

Ham = basic.H2(False, True)
bcs = True

log.result("H = %s\n", Ham)

def idx_order(idx, standard = "ijkl"):
    length = len(standard)
    rotations = map(lambda x: idx.index(x), standard)
    s = reduce(lambda x, y: x+y, \
            map(lambda (a, s): a*(length**s), \
            zip(rotations, range(4)[::-1])))
    return s

def reduce_dm_terms(prodstring, replaced = False):
    if not replaced:
        pstring =  prodstring.replace_indices(*zip("".join(prodstring.get_indices()[1:]), 'ijkl'))
    else:
        pstring = prodstring
    int_tensor = pstring[0]
    symm_order = map(idx_order, int_tensor.symm.symm(int_tensor.idx))
    antisymm_order = map(idx_order, int_tensor.symm.antisymm(int_tensor.idx))
    if len(antisymm_order) == 0 or min(symm_order) < min(antisymm_order):
        int_tensor.set_idx(int_tensor.symm.symm(int_tensor.idx)[np.argmin(symm_order)])
        return 1, pstring        
    else:
        int_tensor.set_idx(int_tensor.symm.antisymm(int_tensor.idx)[np.argmin(antisymm_order)])
        return -1, pstring

def merge_terms(expression):
    term_dict = {}
    orderbook = []
    for (factor, opstring) in expression:
        if not opstring in term_dict:
            term_dict[opstring] = factor
            orderbook.append(opstring)
        else:
            term_dict[opstring] += factor
    return basic.OpSum([(term_dict[opstring], opstring) for opstring in orderbook])

# compute energy contraction
energy = basic.OpSum([])
term_order = ["rho_A", "rho_B", "kappa_BA"]
for term in Ham:
    f = term[1].fermions()
    i = 0
    for j in range(1, 4):
        k, l = [s for s in range(1, 4) if s != j]
        try:
            ops1 = basic.reduced(basic.OpProduct([f[i], f[j]]))
            ops2 = basic.reduced(basic.OpProduct([f[k], f[l]]))
            assert(len(ops1) == 1 and len(ops2) == 1)
            sign1, opstring1 = ops1[0]
            sign2, opstring2 = ops2[0]            
            t1 = basic.expectation(opstring1)
            t2 = basic.expectation(opstring2)
            if not bcs and ("kappa" in t1[0].name or "kappa" in t2[0].name):
                continue
            sign = 1 if j in [1, 3] else -1
            if term_order.index(t1[0].name) > term_order.index(t2[0].name):
                t1, t2 = t2, t1
            temp = term[1].nonfermions() * t1 * t2
            sign3, temp = reduce_dm_terms(temp)
            energy += sign1 * sign2 * sign3 * sign * term[0] * basic.OpSum(temp)
        except:
            pass

# merge terms
energy = merge_terms(energy)

log.result("Energy expression\n%s\n", energy)

# compute Fock matrix
FA, FB, FD = basic.OpSum([]), basic.OpSum([]), basic.OpSum([])

for (factor, opstring) in energy:
    temp = basic.OpProduct([opstring[0], opstring[2]]).replace_indices(*zip("ijkl", "klij"))
    d1opstring = factor * basic.OpSum([reduce_dm_terms(temp, replaced = True)])
    d2opstring = basic.OpSum([(factor, basic.OpProduct([opstring[0], opstring[1]]))])
 
    if opstring[1].name == "rho_A":
        FA += d1opstring
    elif opstring[1].name == "rho_B":
        FB += d1opstring
    elif opstring[1].name == "kappa_BA":
        FD += 0.5*d1opstring
 
    if opstring[2].name == "rho_A":
        FA += d2opstring
    elif opstring[2].name == "rho_B":
        FB += d2opstring
    elif opstring[2].name == "kappa_BA":
        FD += 0.5*d2opstring

log.result("Fock (alpha)\n%s\n", merge_terms(FA))
log.result("Fock (beta)\n%s\n", merge_terms(FB))
log.result("Fock (pairing)\n%s\n", merge_terms(FD))

