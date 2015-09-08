import numpy as np

def grad(h_A, h_B, D, w_A, w_B, w_AB, y_A, y_B, x, \
            rho_A, rho_B, kappa_BA, Gamma_aa, Gamma_bb, Gamma_ab, \
            Gamma_2a, Gamma_2b, Gamma_4):
    val001 = (-np.tensordot(y_A, Gamma_2a, axes=((1, 2, 3), (1, 2, 3))))
    val002 = np.tensordot(Gamma_4, x, axes=((1, 2, 3), (1, 3, 2)))
    val003 = np.tensordot(Gamma_2a, y_A, axes=((0, 1, 2), (0, 1, 2)))
    val004 = np.dot(h_A, rho_A)
    val005 = np.dot(D, kappa_BA)
    val006 = np.tensordot(y_B, Gamma_2b, axes=((0, 1, 3), (0, 1, 3)))
    val007 = np.tensordot(Gamma_ab, w_AB, axes=((1, 2, 3), (1, 2, 3)))
    val008 = np.tensordot(w_A, Gamma_aa, axes=((1, 2, 3), (3, 2, 1)))
    grad_aa = -1.0*val006.T + \
            2.0*val008.T + \
            -2.0*val005.T + \
            -2.0*val004.T + \
            -1.0*val003 + \
            2.0*val001.T + \
            -1.0*val002.T + \
            -2.0*val007 + \
            2.0*val005 + \
            1.0*val003.T + \
            -1.0*val001 + \
            1.0*val006 + \
            1.0*(-val001) + \
            -2.0*val008 + \
            2.0*val007.T + \
            2.0*val004 + \
            1.0*val002
    val009 = np.dot(D.T, kappa_BA.T)
    val010 = np.tensordot(Gamma_bb, w_B, axes=((1, 2, 3), (1, 2, 3)))
    val011 = (-np.tensordot(Gamma_2b, y_B, axes=((1, 2, 3), (1, 2, 3))))
    val012 = np.dot(rho_B, h_B)
    val013 = np.tensordot(x, Gamma_4, axes=((0, 1, 3), (0, 1, 3)))
    val014 = np.tensordot(Gamma_2a, y_A, axes=((0, 1, 3), (0, 1, 3)))
    val015 = np.tensordot(w_AB, Gamma_ab, axes=((0, 1, 3), (0, 1, 3)))
    val016 = np.tensordot(Gamma_2b, y_B, axes=((0, 1, 2), (0, 1, 2)))
    grad_bb = 1.0*val014.T + \
            1.0*val013 + \
            -1.0*val013.T + \
            -2.0*val009.T + \
            -1.0*val016 + \
            -1.0*(-val011) + \
            -2.0*val012 + \
            -2.0*val011.T + \
            -2.0*val015.T + \
            2.0*val009 + \
            2.0*val012.T + \
            2.0*val015 + \
            -2.0*val010 + \
            1.0*val011 + \
            -1.0*val014 + \
            1.0*val016.T + \
            2.0*val010.T
    val017 = (-np.tensordot(x, kappa_BA, axes=((1, 3),(1, 0))).T)
    val018 = (-np.tensordot(y_A, Gamma_ab, axes=((1, 2, 3), (1, 3, 0))))
    val019 = (-np.tensordot(y_A, rho_A, axes=((1, 3),(0, 1))).T)
    grad_ab = -2.0*np.dot(rho_A, D) + \
            -2.0*np.tensordot(Gamma_ab, y_B, axes=((1, 2, 3), (2, 3, 1))) + \
            1.0*(-val018) + \
            -1.0*np.tensordot(Gamma_2a, x, axes=((0, 1, 2), (1, 0, 3))) + \
            2.0*np.tensordot(Gamma_2b, w_B, axes=((0, 1, 3), (1, 2, 3))) + \
            1.0*np.tensordot(y_A, Gamma_4, axes=((0, 1, 2), (1, 0, 3))) + \
            2.0*np.dot(h_A, kappa_BA.T) + \
            -1.0*np.tensordot(Gamma_4, y_B, axes=((1, 2, 3), (2, 1, 0))) + \
            -2.0*(-(-np.tensordot(Gamma_2a, w_AB, axes=((1, 2, 3), (1, 3, 0))))) + \
            2.0*np.tensordot(w_AB, kappa_BA, axes=((1, 3),(1, 0))).T + \
            -2.0*np.dot(D, rho_B) + \
            -1.0*val018 + \
            1.0*np.tensordot(x, Gamma_2b, axes=((1, 2, 3), (2, 1, 0))) + \
            2.0*np.tensordot(w_A, Gamma_2a, axes=((1, 2, 3), (1, 3, 0))) + \
            -1.0*np.tensordot(Gamma_aa, y_A, axes=((1, 2, 3), (1, 3, 0))) + \
            -1.0*(-val017) + \
            2.0*(-(-np.tensordot(w_AB, Gamma_2b, axes=((1, 2, 3), (2, 3, 1))))) + \
            1.0*val017 + \
            -1.0*(-val019) + \
            2.0*np.tensordot(y_B, rho_B, axes=((1, 3),(0, 1))).T + \
            1.0*val019 + \
            2.0*np.dot(kappa_BA.T, h_B) + \
            1.0*np.tensordot(y_B, Gamma_bb, axes=((0, 1, 3), (3, 1, 2))) + \
            2.0*D

    return grad_aa, grad_bb, grad_ab