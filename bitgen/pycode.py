import basic
from copy import deepcopy
import numpy as np
import itertools as it
import libdmet.utils.logger as log

__all__ = ["define", "reg", "contract", "sumover", "dump", "registered", "find_term"]

dim = 6
np.random.seed(123)
registered = []

def find_term(expr):
    for reg in registered:
        if reg.get_expr() == expr:
            return reg

def randMatrix(nidx, symm, antisymm):
    mat = np.random.rand(*([dim] * nidx)) * 2 - 1
    mat1 = np.zeros_like(mat)
    for s in symm:
        mat1 += np.transpose(mat, s)
    mat1 /= len(symm)
    for a in antisymm:
        mat1 = 0.5 * (mat1 - np.transpose(mat1, a))
    return mat1

class Intermediate(basic.NumTensor):
    def __init__(self, expr, ops, nidx, symm = None, name = None, \
            val = None, final = False):
        self._dup = -1
        for i in range(len(ops)):
            assert("@%02d" % (i+1) in expr)
        if len(ops) == 0:
            # then is an initial operator
            for i, r in enumerate(registered):
                if r.expr == expr:
                    self._dup = i
                    return
            # if not registered yet
            if name is None:
                name = expr
            basic.NumTensor.__init__(self, None, nidx = nidx, symm = symm)
            self.val = randMatrix(nidx, symm._symm, symm._antisymm)
            self.expr = expr
            self.ops = ops
            self.refcount = 0
            self.final = final
        else:
            for op in ops:
                assert(op in registered)
            if val is None:
                # then compute
                expr1 = deepcopy(expr)
                for i in range(len(ops)):
                    expr1 = expr1.replace("@%02d" % (i+1), "ops[%d].val" % i)
                val = eval(expr1)
            # now check existence
            if not final:
                for i, r in enumerate(registered):
                    if np.allclose(val, r.val):
                        self._dup = i
                        return
            found = []
            for i, r in enumerate(registered):
                if np.allclose(val, r.val):
                    found.append((r.refcount, "@01", [r]))
                    break
                if np.allclose(val, -r.val):
                    found.append((r.refcount, "(-@01)", [r]))
                    break
            for i, r in enumerate(registered):
                for reorder in it.permutations(range(r.nidx)):
                    if np.allclose(val, np.transpose(r.val, reorder)):
                        if reorder == (1,0):
                            found.append((r.refcount, "@01.T", [r]))
                        else:
                            found.append((r.refcount, "np.transpose(@01, %s)" \
                                    % str(reorder), [r]))
                        break
                    elif np.allclose(-val, np.transpose(r.val, reorder)):
                        if reorder == (1,0):
                            found.append((r.refcount, "(-@01.T)", [r]))
                        else:
                            found.append((r.refcount, "(-np.transpose(@01, %s))" \
                                    % str(reorder), [r]))
                        break
            if len(found) > 0:
                _, expr, ops = sorted(found, key = lambda (c, e, _): c * 50 - len(e))[-1]

            # now find symmetry
            if symm is None:
                s = []
                a = []
                for reorder in it.permutations(range(nidx)):
                    if np.allclose(val, np.transpose(val, reorder)):
                        s.append(reorder)
                    elif np.allclose(-val, np.transpose(val, reorder)):
                        a.append(reorder)
                symm = basic.IdxSymmetry(s, a)
            basic.NumTensor.__init__(self, name, nidx = nidx, symm = symm)
            for i, op in enumerate(ops):
                op.refcount += expr.count("@%02d" % (i+1))
            self.val = val
            if final:
                assert(name is not None)
                self.name = name
            else:
                if name is not None:
                    self.name = name
            self.expr = expr
            self.ops = ops
            self.refcount = 0
            self.final = final
        registered.append(self)

    def __eq__(self, other):
        if len(self.ops) == 0 and len(other.ops) == 0:
            return self.name == other.name
        else:
            return np.allclose(self.val, other.val)

    def __hash__(self):
        if len(self.ops) == 0:
            return hash(self.name)
        else:
            return hash(tuple(self.val.ravel()))

    def __str__(self):
        if self.name is None:
            return self.get_expr()
        else:
            return self.name + " = " + self.get_expr()

    def __repr__(self):
        return self.__str__() + \
                " (refcount = %d, symm = %s, antisymm = %s)" % \
                (self.refcount, self.symm._symm, self.symm._antisymm)

    def get_expr(self):
        # generate name using expr
        _name = deepcopy(self.expr)
        for i, op in enumerate(self.ops):
            _name = _name.replace("@%02d" % (i+1), op._get_name())
        return _name

    def _get_name(self):
        if self.name is None:
            return self.get_expr()
        else:
            return self.name

# really badly written
def define(*args, **kwargs):
    instance = Intermediate(*args, **kwargs)
    if instance._dup < 0:
        return instance
    else:
        return registered[instance._dup]

def reg(Op, idx = None, **kwargs):
    if isinstance(Op, Intermediate):
        return (Op, idx)
    elif isinstance(Op, basic.NumTensor):
        return (define(Op.name, [], \
                Op.nidx, symm = Op.symm, **kwargs), list(Op.idx))
    elif isinstance(Op, basic.OpProduct) and len(Op) == 1:
        return reg(Op[0], idx)
    else:
        raise Exception

def T(trans):
    if trans:
        return ".T"
    else:
        return ""

def build_tuple(idx1, idx2):
    if len(idx2) > len(idx1):
        _idx1, _idx2 = idx2, idx1
    else:
        _idx1, _idx2 = idx1, idx2
    t1, t2 = [], []
    for i, idx in enumerate(_idx1):
        if idx in _idx2:
            t1.append(i)
            t2.append(_idx2.index(idx))
    if len(idx2) > len(idx1):
        return tuple(t2), tuple(t1)
    else:
        return tuple(t1), tuple(t2)

def contract(_Op1, _Op2, **kwargs):
    Op1, idx1 = _Op1
    Op2, idx2 = _Op2
    sumover = set(idx1).intersection(idx2)
    if len(sumover) == 0:
        raise Exception("No common indices")
    elif len(sumover) == 1 and len(idx1) == 2 and len(idx2) == 2:
        # use np.dot
        idx_sum = list(sumover)[0]
        transl =  (idx1.index(idx_sum) == 0) and (not (1,0) in Op1.symm._symm)
        transr = (idx2.index(idx_sum) == 1) and (not (1,0) in Op2.symm._symm)
        expr = "np.dot(@01%s, @02%s)" % (T(transl), T(transr))
        idx = [i for i in idx1 + idx2 if i != idx_sum]
        return define(expr, [Op1, Op2], len(idx), **kwargs), idx
    elif len(sumover) == 2 and len(idx1) == 2 and len(idx2) == 2:
        # this is trace
        trans = (idx1 == idx2) and not ((1,0) in Op1.symm._symm or \
                (1,0) in Op2.symm._symm)
        if "rightT" in kwargs:
            expr = "np.trace(np.dot(@01, @02%s))" % T(trans)
            del kwargs["rightT"]
        else:
            expr = "np.trace(np.dot(@01%s, @02))" % T(trans)
        return define(expr, [Op1, Op2], 0, **kwargs), ""
    elif len(sumover) == 1 and min(len(idx1), len(idx2)) == 2:
        # use tensordot
        idx_sum = list(sumover)[0]
        best1, best2 = None, None
        for ridx1 in Op1.symm.symm(idx1) + Op1.symm.antisymm(idx1):
            if ridx1 in Op1.symm.symm(idx1):
                fac1 = 1
            else:
                fac1 = -1
            for ridx2 in Op2.symm.symm(idx2) + Op2.symm.antisymm(idx2):
                if ridx2 in Op2.symm.symm(idx2):
                    fac2 = 1
                else:
                    fac2 = -1
                pos1 = ridx1.index(idx_sum)
                pos2 = ridx2.index(idx_sum)
                if len(idx1) == 2:
                    if best2 is None or pos2 < best2: # as left as possible
                        best1, best2 = pos1, pos2
                        bidx1, bidx2 = ridx1, ridx2
                        factor = fac1 * fac2
                else:
                    if best1 is None or pos1 > best1: # as right as possible
                        best1, best2 = pos1, pos2
                        bidx1, bidx2 = ridx1, ridx2
                        factor = fac1 * fac2
        expr = "np.tensordot(@01, @02, axes=(%s,%s))" % \
                (best1, best2)
        if factor == -1:
            expr = "(-" + expr + ")"
        idx = [i for i in bidx1 + bidx2 if not i in sumover]
        return define(expr, [Op1, Op2], len(idx), **kwargs), idx
    elif len(sumover) == 2 and min(len(idx1), len(idx2)) == 2:
        best1, best2 = None, None
        if len(idx1) == 2:
            def compare(_idx1,_idx2):
                if _idx1[0] < _idx2[0]:
                    return -1
                elif _idx1[0] == _idx2[0] and _idx1[1] < _idx2[1]:
                    return -1
                elif _idx1 == _idx2:
                    return 0
                else:
                    return 1
            def better(_t1, _t2, _best1, _best2):
                return _best1 is None or compare(_t2, _best2) < 0 or \
                        (compare(_t2, _best2) == 0 and compare(_t1, _best1) < 0)
        elif len(idx2) == 2:
            # FIXME also use antisymmetry
            def compare(_idx1,_idx2):
                if _idx1[1] < _idx2[1]:
                    return -1
                elif _idx1[1] == _idx2[1] and _idx1[0] < _idx2[0]:
                    return -1
                elif _idx1 == _idx2:
                    return 0
                else:
                    return 1
            def better(_t1, _t2, _best1, _best2):
                return _best1 is None or compare(_t1, _best1) > 0 or \
                        (compare(_t1, _best1) == 0 and compare(_t2, _best2) > 0)

        for ridx1 in Op1.symm.symm(idx1) + Op1.symm.antisymm(idx1):
            if ridx1 in Op1.symm.symm(idx1):
                fac1 = 1
            else:
                fac1 = -1
            for ridx2 in Op2.symm.symm(idx2) + Op2.symm.antisymm(idx2):
                if ridx2 in Op2.symm.symm(idx2):
                    fac2 = 1
                else:
                    fac2 = -1
                t1, t2 = build_tuple(ridx1, ridx2)
                if better(t1, t2, best1, best2):
                    best1, best2 = t1, t2
                    bidx1, bidx2 = ridx1, ridx2
                    factor = fac1 * fac2
        if "indices_only" in kwargs:
            return best1, best2
        expr = "np.tensordot(@01, @02, axes=(%s,%s))" % \
                (best1, best2)
        idx = [i for i in bidx1 + bidx2 if not i in sumover]
        if "indices" in kwargs:
            assert(sorted(kwargs["indices"]) == sorted(idx))
            if kwargs["indices"] != idx:
                order = tuple(map(lambda i: idx.index(i), kwargs["indices"]))
                if len(idx) == 2:
                    expr = "%s.T" % expr
                else:
                    expr = "np.transpose(%s, %s)" % (expr, order)
            del kwargs["indices"]
        if factor == -1:
            expr = "(-" + expr + ")"
        return define(expr, [Op1, Op2], len(idx), **kwargs), idx
    elif len(sumover) == 3 and len(idx1) == 4 and len(idx2) == 4:
        i1 = filter(lambda i: not i in sumover, idx1)[0]
        i2 = filter(lambda i: not i in sumover, idx2)[0]
        best1, best2 = None, None
        bidx1, bidx2 = None, None
        for ridx1 in Op1.symm.symm(idx1) + Op1.symm.antisymm(idx1):
            if ridx1 in Op1.symm.symm(idx1):
                fac1 = 1
            else:
                fac1 = -1
            for ridx2 in Op2.symm.symm(idx2) + Op2.symm.antisymm(idx2):
                if ridx2 in Op2.symm.symm(idx2):
                    fac2 = 1
                else:
                    fac2 = -1
                t1, t2 = build_tuple(ridx1, ridx2)
                if bidx1 is None or bidx1.index(i1) + bidx2.index(i2) > \
                        ridx1.index(i1) + ridx2.index(i2):
                    best1, best2 = t1, t2
                    bidx1, bidx2 = ridx1, ridx2
                    factor = fac1 * fac2
        expr = "np.tensordot(@01, @02, axes=(%s, %s))" % \
                (best1, best2)
        if factor == -1:
            expr = "(-" + expr + ")"
        idx = [i1, i2]
        return define(expr, [Op1, Op2], len(idx), **kwargs), idx
    else:
        raise Exception

def sumover(toSum):
    # deal with repeated terms
    sum_dict = {}
    def add_to_dict(key, val):
        if key in sum_dict:
            sum_dict[key] += val
        else:
            sum_dict[key] = val

    for (fac, ops) in toSum:
        if ops.expr.startswith("(-") and ops.expr.endswith(")"):
            ops1 = define(ops.expr[2:-1], ops.ops, ops.nidx)
            add_to_dict(ops1, -fac)
        else:
            add_to_dict(ops, fac)

    expr = " + \\\n        ".join([str(sum_dict[ops]) + "*@%02d" % (i+1) \
            for i, ops in enumerate(sum_dict)])
    oplist = sum_dict.keys()
    return expr, oplist


def add_intermediates():
    log.info("Removing unused intermediates")
    while True:
        unused = []
        # firt find unused intermediates
        for i, inter in enumerate(registered):
            if not inter.final and inter.refcount == 0:
                unused.append(i)

        if len(unused) == 0:
            break

        log.info("remove %d intermediates", len(unused))
        # delete their reference count
        for i in unused:
            inter = registered[i]
            for j, op in enumerate(inter.ops):
                op.refcount -= inter.expr.count("@%02d" % (j+1))

        # delete them
        for k, i in enumerate(unused):
            del registered[i-k]

    count = 0
    for inter in registered:
        if not (inter.final or len(inter.ops) == 0) \
                and inter.refcount > 1:
            count += 1
            inter.name = "val%03d" % count
    log.info("%d intermediates will be stored", count)

def dump():
    add_intermediates()

    log.result("dump code")
    code = []
    for inter in registered:
        if inter.final or inter.name is not None:
            code.append(inter.__str__())
    return "\n".join(code)
