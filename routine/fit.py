import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
from scipy.optimize import fmin
from scipy.linalg import lstsq


def minimize(fn, x0, MaxIter = 300, fgrad = None, callback = None, **kwargs):
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
        log.debug(0, "Fitting: using %d cores to evaluate objective function", \
                cpu_count())
        p = ProcessingPool()
    else:
        log.debug(0, "Fitting: serial specified or failed" \
                " to load multiprocessing module, using single core")

    def grad(x):
        if callback is not None:
            ref = callback(x)
            fn1 = lambda x1: fn(x1, ref = ref)
        else:
            fn1 = fn

        g = np.zeros_like(x)
        step = 1e-5
        def gix(ix):
            #log.debug(1, "Gradient: %d of %d", ix, x.shape[0])
            dx = np.zeros_like(x)
            dx[ix] = step
            return (0.5/step) * (fn1(x+dx) - fn1(x-dx))
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

    log.debug(1, "  Iter           Value               Grad                 Step\n"
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

        if callback is None:
            LineSearchFn = lambda step: fn(x - step * dx)
        else:
            ref_ = callback(x)
            LineSearchFn = lambda step: fn(x - step * dx, ref_)

        def FindStep():
            scale = abs(np.average(steps[-2:]))
            grid = list(np.arange(0.,2.001,0.1) * scale)
            val = map(LineSearchFn, grid)
            s = grid[np.argmin(val)]
            if abs(s) > 1e-4:
                return s
            else:
                xopt, fopt, _, _, _ = fmin(LineSearchFn, np.array([0.001]), disp = 0, \
                        xtol = 1e-10, full_output = True)
                # in case fmin doesn't work
                if fopt < np.min(val):
                    return xopt
                else:
                    return s

        def FindStep2():
            scale = abs(np.average(steps[-2:])) * 2.
            
            def find(a, b):
                log.debug(2, "find(a, b)  a = %f   b = %f", a, b)
                grid = list(np.linspace(a, b, 21))
                val = p.map(LineSearchFn, grid)
                idxmin = np.argmin(val)
                s = grid[idxmin]

                if abs(a-b) < 1e-9 or abs(s) > 1e-3:
                    return s
                else:
                    if idxmin == 0:
                        return find(grid[idxmin], grid[idxmin+1])
                    elif idxmin == len(grid) - 1:
                        return find(grid[idxmin-1], grid[idxmin])
                    else:
                        return find(grid[idxmin-1], grid[idxmin+1])

            return find(-0.1*scale, 1.9*scale)

        if multi:
            step = FindStep2()
        else:
            step = FindStep()
        steps.append(step)
        dx *= step
        y_new = fn(x - dx)
 
        if y_new > y * 1.5 or abs(y - y_new) < 1e-7 or la.norm(dx) < 1e-6:
            break

        x -= dx
        y = y_new

        log.debug(1, "%4d %20.12f %20.12f %20.12f", iter, y, la.norm(g), la.norm(dx))

    return x, y

if __name__ == "__main__":
    log.verbose = "DEBUG1"
    x, y = minimize(lambda x: x[0]**2 + x[1]**4 + 2*x[1]**2 + 2*x[0] + 2., np.asarray([10., 20.]), MaxIter = 300)
    log.result("x = %s\ny=%20.12f", x, y)
