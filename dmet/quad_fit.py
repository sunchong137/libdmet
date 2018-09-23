#! /usr/bin/env python
'''
Quadratic fitting of three points.
Zhihao Cui zcui@caltech.edu
'''


import math
import cmath
from scipy import stats
import numpy as np

def calc_parabola_vertex(x, y, tol = 1e-12):
    x1, x2, x3 = x
    y1, y2, y3 = y
    denom = float((x1 - x2) * (x1 - x3) * (x2 - x3))
    if abs(denom) < tol:
        status = False
        return 0, 0, 0, status
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    status = True
    return a, b, c, status

def quadsolver(a, b, c, tol = 1e-12):
    #print('Equation: {0}x**2 + {1}x + {2}'.format(a,b,c))
    if abs(a) <tol and abs(b) < tol:
        if abs(c) > tol:
            print('Not a valid equation')
        else:
            print(' 0 = 0 is not an interesting equation')
        status = 0
        return [], status 

    if abs(a) < tol:
        print('WARNING: colinear, single solution is x =', -c/b)
        status = 2
        return [-c/float(b)], status

    discriminant = b**2 - 4 * a * c
    if discriminant > 0:
        root1 = (-b + math.sqrt(discriminant))/ (2.0 * a)
        root2 = (-b - math.sqrt(discriminant))/ (2.0 * a)
        #print('Has two roots:')
        #print(root1)
        #print(root2)
        status = 1
        return [root1, root2], status
    elif discriminant == 0:
        root1 = float(-b + math.sqrt(discriminant))/ (2.0 * a)
        #print('Has a double root:')
        #print(root1)
        status = 1
        return [root1, root1], status
    elif discriminant < 0:
        root1 = (-b + cmath.sqrt(discriminant))/ (2.0 * a)
        root2 = (-b - cmath.sqrt(discriminant))/ (2.0 * a)
        print('WARNING: quadratic eq. has two complex roots:')
        print(root1)
        print(root2)
        status = 3
        return [root1, root2], status


def quad_fit(mu, deltaN, tol = 1e-12):
    mu_lst = np.asarray(mu).copy()
    deltaN_lst = np.asarray(deltaN).copy()
    assert(len(mu_lst) == len(deltaN_lst) and len(mu_lst) == 3)
    idx1 = np.argsort(mu_lst)
    idx2 = np.argsort(deltaN_lst)
    if not (idx1 == idx2).all():
        print "WARNING: deltaN is not a monotonic function of mu"
    mu_lst = mu_lst[idx1]
    deltaN_lst = deltaN_lst[idx1]
    
    a, b, c, status = calc_parabola_vertex(mu_lst, deltaN_lst, tol = tol)
    if not status:
        print "Duplicated points among three dots"
        return 0, False
    roots, status = quadsolver(a, b, c, tol = tol)
    #print "ROOTS, ", roots
    if status == 0:
        print "Root finding error"
        return 0, False
    elif status == 2:
        mu_new = roots[0]
        return mu_new, True
    elif status == 3:
        if abs(roots[0].imag) + abs(roots[1].imag) > 1e-3:
            print "Complex root finding"
            return 0, False
        else:
            roots = [roots[0].real, roots[1].real]
    
    if deltaN_lst[0] >= 0.0:
        left = -99999
        right = mu_lst[0]
    elif deltaN_lst[1] >= 0.0:
        left = mu_lst[0]
        right = mu_lst[1]
    elif deltaN_lst[2] >= 0.0:
        left = mu_lst[1]
        right = mu_lst[2]
    else:
        left = mu_lst[2]
        right = 99999
    
    if roots[0] < right and roots[0] > left:
        if roots[1] < right and roots[1] > left: 
            if abs(roots[0] - mu[0]) < abs(roots[1] - mu[0]):
                return roots[0], True
            else:
                return roots[1], True
        else:
            return roots[0], True
    else:
        if roots[1] < right and roots[1] > left: 
            return roots[1], True
        else:
            print "Can not find proper root within the range, [%15.6f, %15.6f] " %(left, right)
            print "roots: ", roots
            return 0, False


    

if __name__ == '__main__':


    x1, y1 = 6.035156250e-01, 1.508072e+00
    x2, y2 = 5.432540625e-01, 1.488130e+00
    x3, y3 = 4.890186562e-01, 1.465602e+00
    #x1, y1 = 0.0, 1.0
    #x2, y2 = 1.0, 2.0
    #x3, y3 = 2.0, 3.0

    x = [x1, x2, x3]
    y = [y1, y2, y3]

    #a,b,c, status = calc_parabola_vertex(x, y)

    #print quadsolver(a, b, c, tol = 1e-18)

    #slope, intercept, r_value, p_value, std_err = stats.linregress(y,x)
    #newmu = slope*0.0 + intercept
    #print newmu

    mu = x
    deltaN = y
    mu_new, status = quad_fit(mu, deltaN, tol = 1e-12)

    print mu_new
    print status
