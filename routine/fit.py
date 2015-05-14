import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
from scipy.optimize import fmin
from scipy.linalg import lstsq


def minimize(fn, x0, MaxIter = 300, fgrad = None, **kwargs):
    nx = x0.shape[0]
    if "serial" in kwargs.keys() and kwargs["serial"]:
        multi = False
    else:
        multi = True
        try:
            from pathos.multiprocessing import ProcessingPool, cpu_count
        except ImportError:
            multi = False

    if multi:
        log.debug(1, "Fitting: using %d cores to evaluate objective function", cpu_count())
        p = ProcessingPool()

    def grad(x):
        g = np.zeros_like(x)
        step = 1e-5
        def gix(ix):
            dx = np.zeros_like(x)
            dx[ix] = step
            return (0.5/step) * (fn(x+dx) - fn(x-dx))
        if multi:
            g = np.asarray(p.map(gix, range(nx)))
        else:
            g = np.asarray(map(gix, range(nx)))
        return g
    if fgrad is None:
        fgrad = grad

    #def GetDir(y, g):
    #    g2 = np.empty((1+nx, nx))
    #    g2[0] = g
    #    g2[1:] = 0.1 * y * np.eye(nx)
    #    y2 = np.zeros(1+nx)
    #    y2[0] = y
    #    dx2, fitresid, rank, sigma = lstsq(g2, y2)
    #    return dx2
    
    def GetDir(y, g):
        h = 10 * g / y
        h2 = np.sum(h*h)
        return h * 10 / (1+h2)

    x = x0

    log.debug(2, "  Iter           Value               Grad                 Step\n"
        "---------------------------------------------------------------------")

    y = fn(x)

    steps = [1.]

    for iter in range(MaxIter):
        if (y < 1e-6 and iter != 0):
            break
        g = fgrad(x)
        if la.norm(g) < 1e-5:
            break

        dx = GetDir(y, g)

        LineSearchFn = lambda step: fn(x - step * dx)

        def FindStep():
            scale = np.average(steps[-2:])
            grid = list(np.arange(0.,2.001,0.1) * scale)
            if multi:
                val = p.map(LineSearchFn, grid)
            else:
                val = map(LineSearchFn, grid)
            s = grid[np.argmin(val)]
            if abs(s) > 1e-4:
                return s
            else:
                return fmin(LineSearchFn, np.array([0.001]), disp = 0, xtol = 1e-10)[0]

        step = FindStep()
        steps.append(step)
        dx *= step
        y_new = fn(x - dx)

        if y_new > y * 1.5 or abs(y - y_new) < 1e-7 or la.norm(dx) < 1e-6:
            break

        x -= dx
        y = y_new
        log.debug(2, "%4d %20.12f %20.12f %20.12f", iter, y, la.norm(g), la.norm(dx))

    return x, y
