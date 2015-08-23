import libdmet.utils.logger as log
from libdmet.bitgen import integral_transform as trans
from libdmet.bitgen import basic

log.section("The most general integral transformation code")

Ham = basic.h0term() + basic.H1(False, True) + basic.H2(False, True)

log.result("H = %s", Ham)

sub = trans.Substitution([
    trans.FermionSub(basic.Fermion(True, 'A','i'), basic.OpSum(basic.Coeff("v_A",'ip') * \
            basic.C('A','p')) + basic.OpSum(basic.Coeff("u_A",'ip') * basic.D('B','p'))),
    trans.FermionSub(basic.Fermion(True, 'B','i'), basic.OpSum(basic.Coeff("v_B",'ip') * \
            basic.C('B','p')) + basic.OpSum(basic.Coeff("u_B",'ip') * basic.D('A','p'))),
])

log.result("Substitute:\n%s", sub)

H = trans.transform(sub, Ham)

# now further simplify the terms by classify according to integrals

with open("TransUBCS2e.py", "w") as f:
    f.write(trans.generate_code(H, indices = "pqrs") + "\n")
