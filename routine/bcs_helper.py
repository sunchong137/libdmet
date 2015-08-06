import numpy as np
import libdmet.utils.logger as log

def extractRdm(GRho):
    norbs = GRho.shape[0] / 2
    log.eassert(norbs * 2 == GRho.shape[0], \
            "generalized density matrix dimension error")
    rhoA = GRho[:norbs, :norbs]
    rhoB = np.eye(norbs) - GRho[norbs:, norbs:]
    kappaAB = - GRho[norbs:,:norbs].T
    return rhoA, rhoB, kappaAB

def combineRdm(rhoA, rhoB, kappaAB):
    norbs = rhoA.shape[0]
    GRho = np.empty((2*norbs, 2*norbs))
    GRho[:norbs, :norbs] = rhoA
    GRho[norbs:, norbs:] = np.eye(norbs) - rhoB
    GRho[norbs:, :norbs] = -kappaAB.T
    GRho[:norbs, norbs:] = -kappaAB
    return GRho

def mono_fit(fn, y0, x0, thr, increase = True):
    if not increase:
        return mono_fit(lambda x: -fn(x), -y0, x0, thr, True)

    count = 0
    log.debug(0, "target f(x) = %20.12f", y0)
    def evaluate(xx):
        yy = fn(xx)
        log.debug(0, "Iter %2d, x = %20.12f, f(x) = %20.12f", \
                count, xx, yy)
        return yy

    # first section search
    x = x0
    y = evaluate(x)
    if abs(y - y0) < thr:
        return x

    if y > y0:
        dx = -1.
    else:
        dx = 1.

    while 1:
        x1 = x + dx
        y1 = evaluate(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y-y0) * (y1-y0) < 0:
            break
        else:
            x = x1
            y = y1

    if x < x1:
        sec_x, sec_y = [x, x1], [y, y1]
    else:
        sec_x, sec_y = [x1, x], [y1, y]

    while sec_x[1] - sec_x[0] > 0.1 * thr:
        f = (y0-sec_y[0]) / (sec_y[1] - sec_y[0])
        x1 = sec_x[0] * (1.-f) + sec_x[1] * f
        y1 = evaluate(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y1 - y0) * (sec_y[0] - y0) < 0:
            sec_x = [sec_x[0], x1]
            sec_y = [sec_y[0], y1]
        else:
            sec_x = [x1, sec_x[1]]
            sec_y = [y1, sec_y[1]]

    return 0.5 * (sec_x[0] + sec_x[1])
