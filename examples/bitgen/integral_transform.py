import libdmet.utils.logger as log
import libdmet.bitgen as gen

Ham = gen.h0term() + gen.H1(False, True) + gen.H2(False, True)

log.result("H = %s", Ham)

sub = gen.Substitution([
    gen.FermionSub(gen.Fermion(True, 'A','i'), gen.OpSum(gen.Coeff("v_A",'ip') * \
            gen.C('A','p')) + gen.OpSum(gen.Coeff("u_A",'ip') * gen.D('B','p'))),
    gen.FermionSub(gen.Fermion(True, 'B','i'), gen.OpSum(gen.Coeff("v_B",'ip') * \
            gen.C('B','p')) + gen.OpSum(gen.Coeff("u_B",'ip') * gen.D('A','p'))),
])

log.result("Substitute:\n%s", sub)

def transform(sub, Ham):
    # first replace the terms
    H = sub.replace(Ham)
    # then reduce to primary Operator products
    H = gen.reduced(H)
    # merge terms: classify according to fermion operators
    H = gen.merge(H, indices = "pqrs")
    # evaluate delta functions
    H = gen.eval_delta(H)
    return H
    #H0, H1, H2 = [], [], []
    #for key in H.keys():
    #    if len(key) == 0:
    #        H0.append((key, H[key]))
    #    elif len(key) == 2:
    #        H1.append((key, H[key]))
    #    else:
    #        H2.append((key, H[key]))
    #return H0, H1, H2

H = transform(sub, Ham)


def doubleTrace(term, indices = "pqrs"):
    pass


def addto(dictionary, key, content):
    if key in dictionary:
        dictionary[key].append(content)
    else:
        dictionary[key] = [content]

# now further simplify the terms by classify according to integrals
def gencode(H, indices = "pqrs"):
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
                    addto(H0fromH0, key, (fac, term))
                elif len(term) == 3:
                    addto(H0fromH1, key, (fac, term))
                elif len(term) == 5:
                    addto(H0fromH2, key, (fac, term))
                else:
                    raise Exception()
        elif len(key) == 2:
            for fac, term in expr:
                if len(term) == 3:
                    addto(H1fromH1, key, (fac, term))
                elif len(term) == 5:
                    addto(H1fromH2, key, (fac, term))
                else:
                    raise Exception()
        elif len(key) == 4:
            for fac, term in expr:
                if len(term) == 5:
                    addto(H2fromH2, key, (fac, term))
                else:
                    raise Exception()
    
    pyH0_H0 = gen.pyH0fromH0(H0fromH0, indices)
    pyH0_H1 = gen.pyH0fromH1(H0fromH1, indices)
    pyH0_H2 = gen.pyH0fromH2(H0fromH2, indices)
    pyH1_H1 = gen.pyH1fromH1(H1fromH1, indices)
    print pyH0_H0
    print pyH0_H1
    print pyH0_H2
    print pyH1_H1
    #for key, expr in H.items():
    #    if len(key) == 0: # H0 terms
    #        for fac, term in expr:
    #            if len(term) == 1:
    #                if term[0].name in H0fromH01.keys():
    #                    H0fromH01[term[0].name] += fac
    #                else:
    #                    H0fromH01[term[0].name] = fac
    #            elif len(term) == 3: # H0 from H1
    #                prg = Trace(term, indices)
    #                if prg in H0fromH01.keys():
    #                    H0fromH01[prg] += fac
    #                else:
    #                    H0fromH01[prg] = fac
    #            elif len(term) == 5: # H0 from H2
    #                print fac, term
    #                doubleTrace(term)
    #
    #print "H_0 = " + " + ".join(map(lambda (term, fac): \
    #        str(fac) + "*" + term, H0fromH01.items()))

gencode(H, indices = "pqrs")
