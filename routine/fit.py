import numpy as np
import numpy.linalg as la
import libdmet.utils.logger as log
from scipy.optimize import fmin
from scipy.linalg import lstsq

def minimize(fn, x0, MaxIter = 300, **kwargs):
    nx = x0.shape[0]

    def grad(x):
        g = np.zeros_like(x)
        step = 1e-6
        for ix in range(nx):
            dx = np.zeros_like(x)
            dx[ix] = step
            g[ix] = (0.5/step) * (fn(x+dx) - fn(x-dx))
        return g

    def GetDir(y, g):
        g2 = np.empty((1+nx, nx))
        g2[0] = g
        g2[1:] = 0.1 * y * np.eye(nx)
        y2 = np.zeros(1+nx)
        y2[0] = y
        dx2, fitresid, rank, sigma = lstsq(g2, y2)
        dx = dx2[:nx]
        return dx

    x = x0

    log.debug(2, "  Iter           Value               Grad                 Step\n"
        "---------------------------------------------------------------------")

    y = fn(x)

    for iter in range(MaxIter):
        if (y < 1e-6 and iter != 0):
            break

        g = grad(x)
        dx = GetDir(y, g)
        
        LineSearchFn = lambda step: fn(x - step * dx)

        def FindStep():
            grid = list(np.arange(0,4.001,0.2))
            val = map(LineSearchFn, grid)
            s = grid[np.argmin(val)]
            if abs(s) > 1e-4:
                return s
            else:
                return fmin(LineSearchFn, np.array([0.001]), disp = 0, xtol = 1e-10)

        step = FindStep()
        dx *= step
        y_new = fn(x - dx)

        if y_new > y * 1.5 or la.norm(dx) < 1e-7:
            break

        x -= dx
        y = y_new
        log.debug(2, "%4d %20.12f %20.12f %20.12f", iter, y, la.norm(g), la.norm(dx))

    return x, y
