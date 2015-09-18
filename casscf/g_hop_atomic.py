import numpy as np

def g_hop_atomic(h_A, h_B, D, w, rho_A, rho_B, kappa_BA, Gamma_aa, Gamma_bb, \
        Gamma_ab, Gamma_2a, Gamma_2b, Gamma_4):
    # gradient
    val001 = np.dot(D, kappa_BA)
    val002 = val001.T
    val003 = np.tensordot(Gamma_ab, w, axes=((1, 2, 3), (1, 3, 2)))
    val004 = val003.T
    val005 = np.dot(h_A, rho_A)
    val006 = np.tensordot(w, Gamma_aa, axes=((1, 2, 3), (3, 2, 1)))
    val007 = val006.T
    val008 = val005.T
    gorb_a = 2.0*val007 + \
            2.0*val005 + \
            2.0*val004 + \
            2.0*val001 + \
            -2.0*val006 + \
            -2.0*val002 + \
            -2.0*val003 + \
            -2.0*val008
    val009 = np.dot(D.T, kappa_BA.T)
    val010 = np.tensordot(Gamma_bb, w, axes=((1, 2, 3), (1, 2, 3)))
    val011 = np.dot(rho_B, h_B)
    val012 = val011.T
    val013 = np.tensordot(w, Gamma_ab, axes=((1, 2, 3), (3, 1, 0)))
    val014 = val010.T
    val015 = val009.T
    val016 = val013.T
    gorb_b = -2.0*val010 + \
            2.0*val012 + \
            -2.0*val015 + \
            2.0*val013 + \
            2.0*val009 + \
            -2.0*val016 + \
            2.0*val014 + \
            -2.0*val011
    val017 = np.dot(h_A, kappa_BA.T)
    val018 = np.tensordot(Gamma_2b, w, axes=((0, 1, 3), (2, 1, 3)))
    val019 = np.tensordot(w, Gamma_2a, axes=((1, 2, 3), (1, 0, 3)))
    val020 = np.dot(D, rho_B)
    val021 = np.tensordot(Gamma_2a, w, axes=((1, 2, 3), (2, 1, 3)))
    val022 = np.tensordot(w, kappa_BA, axes=((1, 3),(0, 1))).T
    val023 = np.dot(kappa_BA.T, h_B)
    val024 = np.tensordot(w, Gamma_2b, axes=((1, 2, 3), (2, 3, 1)))
    val025 = np.dot(rho_A, D)
    gorb_d = 2.0*val017 + \
            2.0*val022 + \
            -2.0*val018 + \
            -2.0*val025 + \
            -2.0*val020 + \
            -2.0*val021 + \
            2.0*val023 + \
            2.0*val019 + \
            2.0*val024 + \
            2.0*D
    # h_diag
    val026 = Gamma_aa.diagonal(0, 0, 1)
    val027 = w.diagonal(0, 2, 3)
    val028 = np.tensordot(val026, val027, axes=((0, 1), (0, 1)))
    val029 = np.sum(w*Gamma_ab, axis=(1, 2, 3))
    val030 = np.sum(w*Gamma_aa, axis=(1, 2, 3))
    val031 = np.sum(h_A*rho_A, axis=(1,))
    val032 = w.diagonal(0, 1, 2)
    val033 = np.tensordot(val032, val026, axes=((0, 1), (0, 1)))
    val034 = np.sum(D*np.transpose(kappa_BA, (1, 0)), axis=(1,))
    val035 = np.diag(rho_A)
    val036 = np.diag(h_A)
    val037 = np.outer(val035, val036)
    val038 = Gamma_ab.diagonal(0, 0, 1)
    val039 = np.tensordot(val027, val038, axes=((0, 1), (0, 1)))
    val040 = val039.T
    h_diag_a_q = -2.0*val034 + \
            -2.0*val030 + \
            -2.0*val029 + \
            -2.0*val031
    h_diag_a_P = 4.0*val034 + \
            4.0*val030 + \
            4.0*val029 + \
            4.0*val031
    h_diag_a_pq = 2.0*val040 + \
            -2.0*val033 + \
            2.0*val028 + \
            4.0*np.sum(w*np.transpose(Gamma_aa, (0, 2, 1, 3)), axis=(1, 3)) + \
            -2.0*val033.T + \
            2.0*val039 + \
            2.0*val037 + \
            -4.0*np.sum(w*Gamma_ab, axis=(2, 3)) + \
            2.0*val037.T + \
            2.0*val028.T + \
            4.0*np.sum(w*Gamma_aa, axis=(1, 3)) + \
            -4.0*np.sum(h_A*rho_A, axis=()) + \
            -4.0*np.sum(w*Gamma_aa, axis=(2, 3))
    h_diag_a_p = h_diag_a_q
    h_diag_a = h_diag_a_pq + np.outer(h_diag_a_p, np.ones(h_diag_a_p.shape[0])) + np.outer(np.ones(h_diag_a_q.shape[0]), h_diag_a_q) + np.diag(h_diag_a_P)
    val041 = np.diag(h_B)
    val042 = np.diag(rho_B)
    val043 = np.outer(val041, val042)
    val044 = np.sum(D*np.transpose(kappa_BA, (1, 0)), axis=(0,))
    val045 = np.sum(w*Gamma_bb, axis=(1, 2, 3))
    val046 = Gamma_bb.diagonal(0, 0, 1)
    val047 = np.tensordot(val032, val046, axes=((0, 1), (1, 0)))
    val048 = np.sum(h_B*rho_B, axis=(1,))
    val049 = Gamma_ab.diagonal(0, 2, 3)
    val050 = np.tensordot(val027, val049, axes=((0, 1), (0, 1)))
    val051 = np.sum(w*np.transpose(Gamma_ab, (2, 3, 0, 1)), axis=(1, 2, 3))
    val052 = np.tensordot(val027, val046, axes=((0, 1), (0, 1)))
    h_diag_b_q = -2.0*val048 + \
            -2.0*val051 + \
            -2.0*val044 + \
            -2.0*val045
    h_diag_b_P = 4.0*val048 + \
            4.0*val051 + \
            4.0*val044 + \
            4.0*val045
    h_diag_b_pq = -4.0*np.sum(w*np.transpose(Gamma_ab, (2, 3, 0, 1)), axis=(2, 3)) + \
            -2.0*val047 + \
            2.0*val052 + \
            2.0*val043.T + \
            2.0*val043 + \
            2.0*val050 + \
            -2.0*val047.T + \
            2.0*val052.T + \
            -4.0*np.sum(h_B*rho_B, axis=()) + \
            4.0*np.sum(w*Gamma_bb, axis=(1, 3)) + \
            -4.0*np.sum(w*Gamma_bb, axis=(2, 3)) + \
            4.0*np.sum(w*np.transpose(Gamma_bb, (0, 2, 1, 3)), axis=(1, 3)) + \
            2.0*val050.T
    h_diag_b_p = h_diag_b_q
    h_diag_b = h_diag_b_pq + np.outer(h_diag_b_p, np.ones(h_diag_b_p.shape[0])) + np.outer(np.ones(h_diag_b_q.shape[0]), h_diag_b_q) + np.diag(h_diag_b_P)
    val053 = np.einsum("jkp,jk->p", val027, rho_A)
    val054 = np.einsum("jkp,jk->p", val027, rho_B)
    h_diag_d_q = -2.0*val051 + \
            2.0*val053 + \
            2.0*val054 + \
            -2.0*val048 + \
            -2.0*val044 + \
            2.0*val041 + \
            -2.0*np.einsum("ljq,jl->q", val032, rho_B) + \
            -2.0*val045
    h_diag_d_p = -2.0*val034 + \
            -2.0*val030 + \
            2.0*val053 + \
            2.0*val036 + \
            -2.0*val031 + \
            -2.0*val029 + \
            2.0*val054 + \
            -2.0*np.einsum("ikp,ik->p", val032, rho_A)
    h_diag_d_pq = -4.0*np.einsum("plq,pl->pq", val027, rho_A) + \
            2.0*val027.diagonal(0, 0, 1) + \
            -2.0*val052 + \
            -2.0*val040 + \
            2.0*np.tensordot(val038, val032, axes=((0, 1), (0, 1))) + \
            -2.0*val028 + \
            -2.0*np.outer(val036, val042) + \
            4.0*np.sum(w*np.transpose(Gamma_ab, (0, 1, 3, 2)), axis=(1, 3)) + \
            -2.0*val050 + \
            -4.0*np.einsum("qkp,qk->pq", val027, rho_B) + \
            4.0*np.sum(w*Gamma_4, axis=(1, 3)) + \
            -2.0*np.outer(val035, val041) + \
            2.0*np.tensordot(val032, val049, axes=((0, 1), (0, 1))) + \
            -4.0*np.sum(D*np.transpose(kappa_BA, (1, 0)), axis=())
    h_diag_d = h_diag_d_pq + np.outer(h_diag_d_p, np.ones(h_diag_d_p.shape[0])) + np.outer(np.ones(h_diag_d_q.shape[0]), h_diag_d_q)
    # first order hessian terms
    h_aa_rp = 1.0*val007 + \
            -1.0*val002 + \
            -1.0*val005 + \
            -1.0*val004 + \
            1.0*val006 + \
            -1.0*val001 + \
            -1.0*val008 + \
            -1.0*val003
    h_aa_rq = (-h_aa_rp)
    h_aa_sq = h_aa_rp
    h_aa_sp = (-h_aa_rp)
    h_ad_rp = -1.0*val017 + \
            -1.0*val022 + \
            -1.0*val018 + \
            -1.0*val025 + \
            1.0*val020 + \
            -1.0*val021 + \
            -1.0*val024 + \
            -1.0*val019 + \
            1.0*val023 + \
            -1.0*D
    h_ad_rq = (-h_ad_rp)
    h_bb_rp = -1.0*val010 + \
            -1.0*val012 + \
            -1.0*val015 + \
            -1.0*val013 + \
            -1.0*val009 + \
            -1.0*val016 + \
            -1.0*val014 + \
            -1.0*val011
    h_bb_rq = (-h_bb_rp)
    h_bb_sq = h_bb_rp
    h_bb_sp = (-h_bb_rp)
    h_bd_sq = -1.0*val018.T + \
            1.0*val022 + \
            1.0*val023.T + \
            -1.0*val021.T + \
            1.0*D.T + \
            -1.0*val017.T + \
            -1.0*val025.T + \
            -1.0*val024.T + \
            1.0*val020.T + \
            -1.0*val019.T
    h_bd_sp = (-h_bd_sq)
    val055 = np.tensordot(w, rho_B, axes=((2, 3),(0, 1))).T
    val056 = np.tensordot(w, rho_A, axes=((2, 3),(0, 1))).T
    h_dd_rp = -1.0*val010 + \
            -1.0*val012 + \
            -1.0*val015 + \
            -2.0*np.tensordot(w, rho_B, axes=((1, 3),(0, 1))).T + \
            2.0*val055 + \
            -1.0*val013 + \
            2.0*val056 + \
            -1.0*val009 + \
            -1.0*val016 + \
            -1.0*val014 + \
            -1.0*val011 + \
            2.0*h_B
    h_dd_sq = 1.0*val007 + \
            -1.0*val002 + \
            -1.0*val005 + \
            2.0*val055 + \
            -1.0*val004 + \
            1.0*val006 + \
            -1.0*val001 + \
            2.0*val056 + \
            -2.0*np.tensordot(w, rho_A, axes=((1, 3),(0, 1))).T + \
            -1.0*val003 + \
            2.0*h_A + \
            -1.0*val008

    # h * t operator
    def hop(t_a, t_b, t_d):
        hx_aa1 = 1.0*np.dot(h_aa_sq, t_a) + \
                1.0*np.dot(t_a, h_aa_rp) + \
                1.0*np.dot(t_a.T, h_aa_sp) + \
                1.0*np.dot(h_aa_rq, t_a.T)
        hx_ad1 = 1.0*np.dot(t_d, h_ad_rp.T) + \
                1.0*np.dot(h_ad_rq, t_d.T)
        hx_bb1 = 1.0*np.dot(h_bb_rq, t_b.T) + \
                1.0*np.dot(t_b.T, h_bb_sp) + \
                1.0*np.dot(t_b, h_bb_rp) + \
                1.0*np.dot(h_bb_sq, t_b)
        hx_bd1 = 1.0*np.dot(t_d.T, h_bd_sp.T) + \
                1.0*np.dot(h_bd_sq, t_d)
        hx_da1 = 1.0*np.dot(t_a, h_ad_rp) + \
                1.0*np.dot(t_a.T, h_ad_rq)
        hx_db1 = 1.0*np.dot(h_bd_sq.T, t_b) + \
                1.0*np.dot(h_bd_sp.T, t_b.T)
        hx_dd1 = 1.0*np.dot(h_dd_sq, t_d) + \
                1.0*np.dot(t_d, h_dd_rp)
        val057 = np.tensordot(t_a, Gamma_aa, axes=(0,0))
        val058 = np.tensordot(w, val057, axes=((1, 2, 3), (0, 2, 3)))
        val059 = (-val057)
        val060 = np.tensordot(w, val059, axes=((1, 2, 3), (2, 0, 3)))
        val061 = np.dot(t_a.T, rho_A)
        val062 = np.dot(h_A, val061)
        val063 = np.tensordot(w, val059, axes=((1, 2, 3), (1, 3, 0)))
        val064 = np.tensordot(t_a, Gamma_ab, axes=(0,0))
        val065 = np.tensordot(w, val064, axes=((1, 2, 3), (0, 3, 2)))
        hx_aa2 = -4.0*val058 + \
                4.0*val058.T + \
                -4.0*val060 + \
                4.0*val063.T + \
                -4.0*val063 + \
                -4.0*val062 + \
                -4.0*val065 + \
                4.0*val060.T + \
                4.0*val062.T + \
                4.0*val065.T
        val066 = np.tensordot(t_b, Gamma_ab, axes=(0,2))
        val067 = np.tensordot(w, val066, axes=((1, 2, 3), (2, 3, 0)))
        val068 = np.dot(t_b, kappa_BA)
        val069 = np.dot(D, val068)
        val070 = np.tensordot(w, val066, axes=((1, 2, 3), (1, 0, 3)))
        hx_ab2 = 4.0*val069 + \
                4.0*val070.T + \
                -4.0*val070 + \
                -4.0*val067 + \
                -4.0*val069.T + \
                4.0*val067.T
        val071 = np.dot(t_d, kappa_BA)
        val072 = np.dot(h_A, val071)
        val073 = np.dot(t_d.T, rho_A)
        val074 = np.dot(D, val073)
        val075 = np.tensordot(t_d, Gamma_2a, axes=(1,2))
        val076 = np.tensordot(w, val075, axes=((1, 2, 3), (0, 3, 2)))
        val077 = np.tensordot(w, val075, axes=((1, 2, 3), (1, 2, 0)))
        val078 = np.tensordot(w, val075, axes=((1, 2, 3), (3, 0, 2)))
        val079 = (-np.tensordot(t_d, Gamma_2b, axes=(1,0)))
        val080 = np.tensordot(w, val079, axes=((1, 2, 3), (0, 1, 3)))
        val081 = np.tensordot(w, t_d, axes=((1, 3),(0, 1)))
        val082 = np.dot(val081, kappa_BA)
        val083 = (-np.tensordot(t_d, Gamma_2a, axes=(0,0)))
        val084 = np.tensordot(w, val083, axes=((1, 2, 3), (1, 2, 0)))
        val085 = np.tensordot(w, val083, axes=((1, 2, 3), (3, 0, 2)))
        hx_ad2 = 2.0*val085 + \
                2.0*val072 + \
                -2.0*val078.T + \
                2.0*val080.T + \
                2.0*val078 + \
                -2.0*val076 + \
                -2.0*val072.T + \
                -2.0*val084.T + \
                2.0*val074.T + \
                -2.0*val080 + \
                -2.0*val085.T + \
                2.0*val077 + \
                2.0*val076.T + \
                -2.0*val082.T + \
                2.0*val084 + \
                -2.0*val074 + \
                2.0*val082 + \
                -2.0*val077.T
        val086 = np.tensordot(w, val064, axes=((1, 2, 3), (3, 1, 0)))
        val087 = np.dot(t_a, kappa_BA.T)
        val088 = np.dot(D.T, val087)
        val089 = np.tensordot(w, val064, axes=((1, 2, 3), (2, 1, 0)))
        hx_ba2 = 4.0*val089.T + \
                -4.0*val089 + \
                4.0*val088 + \
                4.0*val086.T + \
                -4.0*val088.T + \
                -4.0*val086
        val090 = np.tensordot(t_b, Gamma_bb, axes=(0,0))
        val091 = np.tensordot(w, val090, axes=((1, 2, 3), (2, 3, 0)))
        val092 = np.tensordot(w, (-val090), axes=((1, 2, 3), (0, 2, 3)))
        val093 = np.tensordot(w, val090, axes=((1, 2, 3), (1, 3, 0)))
        val094 = np.tensordot(w, val066, axes=((1, 2, 3), (0, 2, 1)))
        val095 = np.dot(t_b, rho_B)
        val096 = np.dot(h_B, val095)
        hx_bb2 = 4.0*val094.T + \
                4.0*val096 + \
                4.0*val093 + \
                4.0*val091 + \
                -4.0*val092.T + \
                -4.0*val096.T + \
                -4.0*val091.T + \
                -4.0*val094 + \
                4.0*val092 + \
                -4.0*val093.T
        val097 = np.tensordot(w, val083, axes=((1, 2, 3), (0, 1, 3)))
        val098 = np.tensordot(t_d, Gamma_2b, axes=(0,2))
        val099 = np.tensordot(w, val098, axes=((1, 2, 3), (3, 0, 2)))
        val100 = np.tensordot(w, val079, axes=((1, 2, 3), (3, 0, 2)))
        val101 = np.dot(val081.T, kappa_BA.T)
        val102 = np.tensordot(w, val079, axes=((1, 2, 3), (1, 2, 0)))
        val103 = np.dot(t_d, rho_B)
        val104 = np.dot(D.T, val103)
        val105 = np.tensordot(w, val098, axes=((1, 2, 3), (0, 2, 3)))
        val106 = np.dot(t_d.T, kappa_BA.T)
        val107 = np.dot(h_B, val106)
        val108 = np.tensordot(w, val098, axes=((1, 2, 3), (1, 2, 0)))
        hx_bd2 = 2.0*val100.T + \
                2.0*val099.T + \
                -2.0*val097.T + \
                -2.0*val107.T + \
                -2.0*val101.T + \
                2.0*val102.T + \
                -2.0*val102 + \
                2.0*val107 + \
                -2.0*val104 + \
                2.0*val104.T + \
                -2.0*val108 + \
                2.0*val097 + \
                -2.0*val100 + \
                -2.0*val099 + \
                2.0*val105 + \
                2.0*val108.T + \
                -2.0*val105.T + \
                2.0*val101
        val109 = np.tensordot(t_a, Gamma_2a, axes=(0,0))
        val110 = np.tensordot(t_a, Gamma_2a, axes=(1,3))
        hx_da2 = 4.0*np.tensordot(w, val087, axes=((1, 3),(0, 1))) + \
                -4.0*np.tensordot(w, val110, axes=((1, 2, 3), (3, 2, 0))).T + \
                4.0*np.dot(h_A, val087) + \
                -4.0*np.tensordot(w, np.tensordot(t_a, Gamma_2b, axes=(0,2)), axes=((1, 2, 3), (0, 2, 3))) + \
                4.0*np.dot(D.T, val061).T + \
                4.0*np.tensordot(w, val109, axes=((1, 2, 3), (0, 3, 1))) + \
                -4.0*np.tensordot(w, val109, axes=((1, 2, 3), (2, 3, 0))).T + \
                4.0*np.tensordot(w, (-val109), axes=((1, 2, 3), (1, 3, 0))) + \
                4.0*np.tensordot(w, val110, axes=((1, 2, 3), (2, 1, 0)))
        val111 = np.tensordot(t_b, Gamma_2b, axes=(0,0))
        val112 = np.tensordot(t_b, Gamma_2b, axes=(0,3))
        hx_db2 = -4.0*np.tensordot(w, val112, axes=((1, 2, 3), (3, 2, 0))) + \
                -4.0*np.tensordot(w, (-val112), axes=((1, 2, 3), (2, 1, 0))).T + \
                4.0*np.tensordot(w, val111, axes=((1, 2, 3), (2, 3, 0))) + \
                -4.0*np.tensordot(w, (-val068), axes=((1, 3),(0, 1))).T + \
                4.0*np.dot(h_B, val068).T + \
                4.0*np.tensordot(w, np.tensordot(t_b, Gamma_2a, axes=(0,2)), axes=((1, 2, 3), (0, 2, 3))).T + \
                4.0*np.tensordot(w, (-val111), axes=((1, 2, 3), (0, 1, 3))).T + \
                4.0*np.tensordot(w, val111, axes=((1, 2, 3), (1, 3, 0))).T + \
                4.0*np.dot(D, (-val095))
        val113 = np.tensordot(t_d, Gamma_ab, axes=(1,2))
        val114 = np.tensordot(t_d, Gamma_ab, axes=(0,0))
        val115 = np.tensordot(t_d, Gamma_4, axes=(1,2))
        val116 = np.tensordot(t_d, Gamma_4, axes=(0,0))
        hx_dd2 = -2.0*np.dot(D, val106) + \
                2.0*val081 + \
                2.0*np.tensordot(w, val114, axes=((1, 2, 3), (1, 2, 0))) + \
                2.0*np.tensordot(w, val114, axes=((1, 2, 3), (3, 0, 2))).T + \
                -2.0*np.dot(val081.T, rho_A).T + \
                -2.0*np.tensordot(w, val114, axes=((1, 2, 3), (0, 3, 2))).T + \
                2.0*np.tensordot(w, val116, axes=((1, 2, 3), (2, 0, 3))).T + \
                -2.0*np.dot(h_B, val073).T + \
                -2.0*np.tensordot(w, np.tensordot(t_d, Gamma_bb, axes=(1,0)), axes=((1, 2, 3), (0, 2, 3))) + \
                -2.0*np.tensordot(w, val073, axes=((1, 3),(0, 1))).T + \
                -2.0*np.tensordot(w, np.tensordot(t_d, Gamma_aa, axes=(0,0)), axes=((1, 2, 3), (0, 2, 3))).T + \
                -2.0*np.tensordot(w, val103, axes=((1, 3),(0, 1))) + \
                2.0*np.tensordot(w, val115, axes=((1, 2, 3), (3, 0, 2))).T + \
                2.0*np.tensordot(w, val113, axes=((1, 2, 3), (3, 0, 1))).T + \
                -2.0*np.dot(val081, rho_B) + \
                2.0*np.tensordot(w, val116, axes=((1, 2, 3), (1, 3, 0))) + \
                2.0*np.tensordot(w, val113, axes=((1, 2, 3), (2, 0, 1))) + \
                -2.0*np.tensordot(w, val113, axes=((1, 2, 3), (0, 2, 1))) + \
                -2.0*np.dot(D.T, val071).T + \
                -2.0*np.tensordot(w, val115, axes=((1, 2, 3), (2, 1, 0))) + \
                -2.0*np.dot(h_A, val103)
        return 0.5 * (hx_aa1+hx_aa2+hx_ab2)+hx_ad1+hx_ad2, \
                0.5 * (hx_ba2+hx_bb1+hx_bb2)+hx_bd1+hx_bd2, \
                0.5 * (hx_da1+hx_da2+hx_db1+hx_db2)+hx_dd1+hx_dd2

    # final return
    return (gorb_a, gorb_b, gorb_d), (h_diag_a, h_diag_b, h_diag_d), hop
