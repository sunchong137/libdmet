import basic
import libdmet.utils.logger as log

class FermionSub(object):
    def __init__(self, f, expr, idx_range = None):
        if isinstance(expr, basic.OpProduct):
            expr = basic.OpSum(expr)
        assert(isinstance(expr, basic.OpSum))
        assert(isinstance(f, basic.Fermion))
        self.f = f
        if idx_range is None:
            self.range = map(chr, range(ord(self.f.idx[0]), \
                    ord(self.f.idx[0]) + 6))
        else:
            self.range = idx_range
        self.expr = expr
        self.indices = map(lambda (fac, ops): ops.get_indices(), expr)
        for indices in self.indices:
            assert("".join(indices).count(f.idx[0]) == 1)
        self.sum_indices = map(lambda indices: \
                list(set("".join(indices)).difference(set(['i']))), \
                self.indices)
        for dumb, indices in zip(self.sum_indices, self.indices):
            for idx in dumb:
                assert("".join(indices).count(idx) > 1)
                assert(not idx in self.range)
        self.reset()

    def replace(self, formula):
        assert(isinstance(formula, basic.OpSum))
        return reduce(basic.OpSum.__add__, map(lambda (fac, ops): \
                fac * self._replace_p(ops), formula), basic.Zero)

    def _replace_p(self, ops):
        assert(isinstance(ops, basic.OpProduct))
        self.reset()
        return reduce(basic.OpSum.__mul__, map(lambda op: \
                self._replace_t(op), ops), basic.Unity)

    def _replace_t(self, op):
        assert(isinstance(op, basic.BaseTensor))
        if isinstance(op, basic.Fermion) and op.spin == self.f.spin \
                and op.idx[0] in self.range:
            # first deal with explicit index
            expr = self.expr.replace_indices((self.f.idx[0], op.idx[0]))
            # then do dumb indices
            for i in range(len(expr)):
                dumb_replace = []         
                for c in self.sum_indices[i]:
                    c_rep = c
                    while c_rep in self.used[i]:
                        c_rep = chr(ord(c_rep) + 1)
                    dumb_replace.append(c_rep)
                    self.used[i].append(c_rep)
                expr[i] = (expr[i][0], expr[i][1].replace_indices(\
                        *zip(self.sum_indices[i], dumb_replace)))
            if op.cre != self.f.cre:
                expr = expr.conj()
            return expr
        else:
            return basic.OpSum(op)

    def reset(self):
        self.used = map(lambda *args: [], self.sum_indices)

    def __str__(self):
        return self.f.__str__() + " ==> " + self.expr.__str__()

class Substitution(object):
    def __init__(self, sublist):
        for s in sublist:
            assert(isinstance(s, FermionSub))
        self.sublist = sublist

    def replace(self, formula):
        assert(isinstance(formula, basic.OpSum))
        return reduce(lambda f, sub: sub.replace(f), self.sublist, formula)

    def __str__(self):
        return "\n".join(map(lambda sub: sub.__str__(), self.sublist))

if __name__ == "__main__":
    sub = Substitution([
        FermionSub(basic.C('A','i')[0], basic.Coeff("u_A", 'ip') * \
            basic.C('A','p')),
        FermionSub(basic.C('B','i')[0], basic.Coeff("u_B", 'ip') * \
            basic.C('B','p'))
    ])
    log.section("Substitute operators in an expression")
    I = basic.ccddterm()
    log.result("two-electron integrals\n%s", I)
    log.result("transformed by\n%s", sub)
    log.result("yields\n%s", sub.replace(I))
