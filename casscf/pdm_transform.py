import numpy as np

def cas_pdm_transform(va_A, va_B, ua_A, ua_B, vc_A, vc_B, uc_A, uc_B, \
        rho_A, rho_B, kappa_BA, Gamma_aa, Gamma_bb, Gamma_ab, \
        Gamma_2a, Gamma_2b, Gamma_4):
    val001 = np.dot(vc_A, vc_A.T)
    val002 = np.dot(va_A, rho_A)
    val003 = np.dot(val002, va_A.T)
    val004 = np.dot(va_A, kappa_BA.T)
    val005 = np.dot(val004, ua_A.T)
    val006 = val005.T
    val007 = np.dot(ua_A, rho_B)
    val008 = np.dot(val007, ua_A.T)
    val009 = np.dot(ua_A, ua_A.T)
    rho_a = -1.0*val008 + \
            1.0*val001 + \
            1.0*val005 + \
            1.0*val003 + \
            1.0*val009 + \
            1.0*val006
    val010 = np.dot(vc_B, vc_B.T)
    val011 = np.dot(np.dot(va_B, rho_B), va_B.T)
    val012 = np.dot(np.dot(va_B, kappa_BA), ua_B.T)
    val013 = val012.T
    val014 = np.dot(np.dot(ua_B, rho_A), ua_B.T)
    val015 = np.dot(ua_B, ua_B.T)
    rho_b = 1.0*val015 + \
            1.0*val011 + \
            1.0*val010 + \
            -1.0*val012 + \
            -1.0*val014 + \
            -1.0*val013
    val016 = np.dot(vc_A, uc_B.T)
    val017 = np.dot(val004, va_B.T)
    val018 = np.dot(val002, ua_B.T)
    val019 = np.dot(val007, va_B.T)
    val020 = np.dot(ua_A, va_B.T)
    val021 = np.dot(np.dot(ua_A, kappa_BA), ua_B.T)
    kappa_ba_T = 1.0*val020 + \
            1.0*val017 + \
            -1.0*val019 + \
            1.0*val021 + \
            1.0*val016 + \
            1.0*val018
    val022 = np.einsum("il,jk->iljk", val001, val001)
    val023 = np.einsum("ik,lj->iljk", val001, val003)
    val024 = np.einsum("ik,lj->iljk", val001, val006)
    val025 = np.einsum("ik,lj->iljk", val001, val008)
    val026 = np.einsum("ik,lj->iljk", val001, val009)
    val027 = np.tensordot(va_A, Gamma_aa, axes=(1,0))
    val028 = np.tensordot(va_A, val027, axes=(1,1))
    val029 = np.tensordot(va_A, Gamma_2a, axes=(1,0))
    val030 = np.tensordot(ua_A, val029, axes=(1,2))
    val031 = np.tensordot(np.tensordot(val030, va_A, axes=(2,1)), va_A, axes=(2,1))
    val032 = np.tensordot(va_A, Gamma_4, axes=(1,0))
    val033 = (-np.tensordot(ua_A, val032, axes=(1,2)))
    val034 = np.tensordot(np.tensordot(val033, va_A, axes=(2,1)), ua_A, axes=(2,1))
    val035 = np.tensordot(va_A, Gamma_ab, axes=(1,0))
    val036 = np.tensordot(ua_A, val035, axes=(1,2))
    val037 = np.tensordot(np.tensordot(val036, ua_A, axes=(3,1)), va_A, axes=(2,1))
    val038 = np.einsum("ik,lj->iljk", val003, val009)
    val039 = np.tensordot(va_A, Gamma_2b, axes=(1,2))
    val040 = (-np.tensordot(ua_A, val039, axes=(1,1)))
    val041 = np.tensordot(np.tensordot(val040, ua_A, axes=(3,1)), ua_A, axes=(2,1))
    val042 = np.einsum("il,jk->iljk", val005, val009)
    val043 = np.tensordot(ua_A, Gamma_bb, axes=(1,0))
    val044 = np.tensordot(ua_A, val043, axes=(1,1))
    val045 = np.einsum("il,jk->iljk", val008, val009)
    val046 = np.einsum("il,jk->iljk", val009, val009)
    gamma0_a = 1.0*np.tensordot(np.tensordot(val044, ua_A, axes=(3,1)), ua_A, axes=(2,1)) + \
            -1.0*np.transpose(val024, (1, 0, 3, 2)) + \
            1.0*np.transpose(val024, (0, 3, 1, 2)) + \
            1.0*np.transpose(val042, (2, 3, 0, 1)) + \
            -1.0*np.transpose(val042, (0, 2, 3, 1)) + \
            -1.0*np.transpose(val045, (2, 3, 0, 1)) + \
            -1.0*np.transpose(val038, (1, 0, 3, 2)) + \
            1.0*np.transpose(val038, (1, 2, 0, 3)) + \
            -1.0*np.transpose(val041, (0, 2, 3, 1)) + \
            -1.0*val045 + \
            1.0*np.transpose(val024, (2, 1, 0, 3)) + \
            -1.0*np.transpose(val025, (0, 3, 1, 2)) + \
            -1.0*np.transpose(val031, (1, 0, 2, 3)) + \
            1.0*np.transpose(val045, (2, 0, 1, 3)) + \
            1.0*np.transpose(val025, (1, 0, 3, 2)) + \
            -1.0*val024 + \
            1.0*np.transpose(val034, (1, 0, 2, 3)) + \
            1.0*np.transpose(val038, (0, 3, 1, 2)) + \
            -1.0*np.transpose(val023, (1, 0, 3, 2)) + \
            -1.0*np.transpose(val042, (2, 0, 1, 3)) + \
            -1.0*np.transpose(val022, (0, 2, 3, 1)) + \
            -1.0*val023 + \
            1.0*np.transpose(val024, (1, 2, 0, 3)) + \
            1.0*np.transpose(val042, (2, 3, 1, 0)) + \
            1.0*np.transpose(val042, (1, 0, 2, 3)) + \
            1.0*val042 + \
            1.0*np.transpose(val031, (1, 3, 2, 0)) + \
            -1.0*np.transpose(val042, (2, 1, 0, 3)) + \
            1.0*np.transpose(val037, (1, 0, 2, 3)) + \
            1.0*np.transpose(val045, (0, 2, 3, 1)) + \
            1.0*np.transpose(val026, (0, 3, 1, 2)) + \
            -1.0*np.transpose(val037, (1, 3, 2, 0)) + \
            -1.0*np.transpose(val025, (1, 2, 0, 3)) + \
            -1.0*np.transpose(val026, (1, 0, 3, 2)) + \
            1.0*np.transpose(val041, (0, 1, 3, 2)) + \
            -1.0*np.transpose(val042, (1, 2, 3, 0)) + \
            -1.0*np.transpose(val046, (0, 2, 3, 1)) + \
            -1.0*np.transpose(val031, (0, 1, 3, 2)) + \
            1.0*val046 + \
            1.0*np.tensordot(np.tensordot(val028, va_A, axes=(3,1)), va_A, axes=(2,1)) + \
            1.0*np.transpose(val041, (1, 0, 2, 3)) + \
            1.0*np.transpose(val023, (0, 3, 1, 2)) + \
            -1.0*np.transpose(val024, (0, 2, 1, 3)) + \
            -1.0*np.transpose(val037, (0, 2, 3, 1)) + \
            -1.0*val038 + \
            1.0*np.transpose(val031, (3, 1, 0, 2)) + \
            1.0*np.transpose(val024, (0, 3, 2, 1)) + \
            1.0*val022 + \
            1.0*val025 + \
            1.0*np.transpose(val026, (1, 2, 0, 3)) + \
            -1.0*val026 + \
            -1.0*np.transpose(val041, (2, 0, 1, 3)) + \
            -1.0*np.transpose(val024, (2, 0, 3, 1)) + \
            1.0*np.transpose(val034, (0, 1, 3, 2)) + \
            1.0*np.transpose(val037, (0, 1, 3, 2)) + \
            1.0*np.transpose(val023, (1, 2, 0, 3))
    val047 = np.einsum("il,jk->iljk", val010, val010)
    val048 = np.einsum("ik,lj->iljk", val010, val011)
    val049 = np.einsum("ik,lj->iljk", val010, val013)
    val050 = np.einsum("ik,lj->iljk", val010, val014)
    val051 = np.einsum("ik,lj->iljk", val010, val015)
    val052 = np.tensordot(va_B, Gamma_bb, axes=(1,0))
    val053 = np.tensordot(va_B, Gamma_2b, axes=(1,0))
    val054 = np.tensordot(np.tensordot(np.tensordot(ua_B, val053, axes=(1,2)), va_B, axes=(2,1)), va_B, axes=(2,1))
    val055 = np.tensordot(va_B, Gamma_4, axes=(1,2))
    val056 = np.tensordot(np.tensordot((-np.tensordot(ua_B, val055, axes=(1,1))), va_B, axes=(3,1)), ua_B, axes=(2,1))
    val057 = np.tensordot(va_B, Gamma_ab, axes=(1,2))
    val058 = np.tensordot(np.tensordot(np.tensordot(ua_B, val057, axes=(1,1)), ua_B, axes=(2,1)), va_B, axes=(2,1))
    val059 = np.einsum("ik,lj->iljk", val011, val015)
    val060 = np.tensordot(va_B, Gamma_2a, axes=(1,2))
    val061 = np.tensordot(np.tensordot((-np.tensordot(ua_B, val060, axes=(1,1))), ua_B, axes=(3,1)), ua_B, axes=(2,1))
    val062 = np.einsum("il,jk->iljk", val012, val015)
    val063 = np.tensordot(ua_B, Gamma_aa, axes=(1,0))
    val064 = np.einsum("il,jk->iljk", val014, val015)
    val065 = np.einsum("il,jk->iljk", val015, val015)
    gamma0_b = -1.0*np.transpose(val059, (1, 0, 3, 2)) + \
            1.0*np.transpose(val061, (0, 1, 3, 2)) + \
            1.0*np.transpose(val062, (2, 0, 1, 3)) + \
            1.0*np.transpose(val056, (0, 1, 3, 2)) + \
            1.0*np.transpose(val049, (1, 0, 3, 2)) + \
            -1.0*np.transpose(val061, (2, 0, 1, 3)) + \
            1.0*np.transpose(val051, (0, 3, 1, 2)) + \
            1.0*np.transpose(val061, (1, 0, 2, 3)) + \
            1.0*val047 + \
            1.0*np.transpose(val058, (0, 1, 3, 2)) + \
            1.0*np.transpose(val051, (1, 2, 0, 3)) + \
            1.0*np.transpose(val049, (2, 0, 3, 1)) + \
            1.0*np.transpose(val048, (1, 2, 0, 3)) + \
            1.0*np.transpose(val050, (1, 0, 3, 2)) + \
            1.0*np.transpose(val056, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val047, (0, 2, 3, 1)) + \
            1.0*val049 + \
            -1.0*val062 + \
            -1.0*np.transpose(val051, (1, 0, 3, 2)) + \
            -1.0*np.transpose(val049, (0, 3, 2, 1)) + \
            -1.0*np.transpose(val062, (1, 0, 2, 3)) + \
            1.0*np.transpose(val062, (2, 1, 0, 3)) + \
            1.0*np.transpose(val048, (0, 3, 1, 2)) + \
            1.0*np.transpose(val059, (0, 3, 1, 2)) + \
            -1.0*val064 + \
            1.0*np.tensordot(np.tensordot(np.tensordot(va_B, val052, axes=(1,1)), va_B, axes=(3,1)), va_B, axes=(2,1)) + \
            -1.0*val059 + \
            -1.0*np.transpose(val049, (0, 3, 1, 2)) + \
            1.0*np.transpose(val062, (0, 2, 3, 1)) + \
            1.0*np.transpose(val054, (3, 1, 0, 2)) + \
            -1.0*np.transpose(val054, (1, 0, 2, 3)) + \
            1.0*np.transpose(val064, (2, 0, 1, 3)) + \
            1.0*val050 + \
            -1.0*np.transpose(val065, (0, 2, 3, 1)) + \
            -1.0*np.transpose(val061, (0, 2, 3, 1)) + \
            1.0*np.transpose(val049, (0, 2, 1, 3)) + \
            -1.0*np.transpose(val058, (0, 2, 3, 1)) + \
            1.0*np.transpose(val058, (1, 0, 2, 3)) + \
            1.0*val065 + \
            -1.0*np.transpose(val050, (0, 3, 1, 2)) + \
            1.0*np.transpose(val064, (0, 2, 3, 1)) + \
            -1.0*np.transpose(val062, (2, 3, 0, 1)) + \
            1.0*np.transpose(val062, (1, 2, 3, 0)) + \
            1.0*np.tensordot(np.tensordot(np.tensordot(ua_B, val063, axes=(1,1)), ua_B, axes=(3,1)), ua_B, axes=(2,1)) + \
            -1.0*np.transpose(val064, (2, 3, 0, 1)) + \
            -1.0*np.transpose(val058, (1, 3, 2, 0)) + \
            -1.0*np.transpose(val048, (1, 0, 3, 2)) + \
            -1.0*np.transpose(val062, (2, 3, 1, 0)) + \
            -1.0*np.transpose(val050, (1, 2, 0, 3)) + \
            1.0*np.transpose(val054, (1, 3, 2, 0)) + \
            -1.0*val048 + \
            -1.0*np.transpose(val054, (0, 1, 3, 2)) + \
            -1.0*val051 + \
            -1.0*np.transpose(val049, (2, 1, 0, 3)) + \
            1.0*np.transpose(val059, (1, 2, 0, 3)) + \
            -1.0*np.transpose(val049, (1, 2, 0, 3))
    val066 = np.einsum("ij,lk->iljk", val016, val016)
    val067 = np.einsum("il,jk->iljk", val001, val012)
    val068 = np.einsum("ij,lk->iljk", val016, val017)
    val069 = np.einsum("ij,lk->iljk", val016, val018)
    val070 = np.einsum("ij,lk->iljk", val016, val019)
    val071 = np.einsum("ij,lk->iljk", val016, val020)
    val072 = np.einsum("ij,lk->iljk", val016, val021)
    val073 = np.transpose(val068, (1, 0, 3, 2))
    val074 = np.transpose(val069, (1, 0, 3, 2))
    val075 = np.tensordot(np.tensordot(np.tensordot(va_A, val029, axes=(1,3)), va_B, axes=(3,1)), ua_B, axes=(2,1))
    val076 = np.einsum("jk,il->iljk", val010, val005)
    val077 = np.tensordot(np.tensordot(val040, va_B, axes=(2,1)), va_B, axes=(2,1))
    val078 = np.einsum("ij,lk->iljk", val017, val020)
    val079 = np.tensordot(np.tensordot(val033, va_B, axes=(3,1)), ua_B, axes=(2,1))
    val080 = np.tensordot(np.tensordot(val036, ua_B, axes=(2,1)), va_B, axes=(2,1))
    val081 = np.einsum("ij,lk->iljk", val018, val020)
    val082 = np.tensordot(np.tensordot(val030, ua_B, axes=(3,1)), ua_B, axes=(2,1))
    val083 = np.einsum("il,jk->iljk", val005, val015)
    val084 = np.transpose(val070, (1, 0, 3, 2))
    val085 = np.transpose(val071, (1, 0, 3, 2))
    val086 = np.transpose(val072, (1, 0, 3, 2))
    val087 = np.transpose(val078, (1, 0, 3, 2))
    val088 = np.transpose(val081, (1, 0, 3, 2))
    val089 = np.einsum("lk,ij->iljk", val019, val020)
    val090 = np.einsum("ij,lk->iljk", val020, val020)
    val091 = np.transpose(val089, (1, 0, 3, 2))
    val092 = np.tensordot(np.tensordot((-np.tensordot(ua_A, np.tensordot(ua_A, Gamma_2b, axes=(1,3)), axes=(1,1))), va_B, axes=(2,1)), ua_B, axes=(2,1))
    val093 = np.einsum("lk,ij->iljk", val021, val020)
    val094 = np.einsum("jk,il->iljk", val012, val009)
    val095 = np.transpose(val093, (1, 0, 3, 2))
    gamma0_ab = -1.0*val070 + \
            1.0*val093 + \
            1.0*np.einsum("il,jk->iljk", val001, val011) + \
            -1.0*val094 + \
            -1.0*np.transpose(val067, (0, 1, 3, 2)) + \
            1.0*val066 + \
            1.0*np.einsum("il,jk->iljk", val009, val015) + \
            1.0*np.einsum("jk,il->iljk", val011, val009) + \
            1.0*np.einsum("il,jk->iljk", val001, val010) + \
            1.0*np.transpose(val082, (1, 0, 2, 3)) + \
            1.0*np.transpose(val092, (1, 0, 2, 3)) + \
            -1.0*np.einsum("jk,il->iljk", val014, val009) + \
            -1.0*np.transpose(val079, (1, 0, 2, 3)) + \
            1.0*np.einsum("jk,il->iljk", val010, val009) + \
            1.0*np.transpose(val083, (1, 0, 2, 3)) + \
            1.0*val086 + \
            1.0*val095 + \
            1.0*val074 + \
            -1.0*np.transpose(np.tensordot(np.tensordot(val028, ua_B, axes=(3,1)), ua_B, axes=(2,1)), (0, 1, 3, 2)) + \
            1.0*np.tensordot(np.tensordot(np.tensordot(ua_A, np.tensordot(ua_A, Gamma_ab, axes=(1,2)), axes=(1,3)), ua_B, axes=(3,1)), ua_B, axes=(2,1)) + \
            1.0*val083 + \
            1.0*val076 + \
            -1.0*np.transpose(val079, (0, 1, 3, 2)) + \
            1.0*np.transpose(val082, (0, 1, 3, 2)) + \
            1.0*np.einsum("il,jk->iljk", val003, val015) + \
            -1.0*val067 + \
            1.0*val069 + \
            -1.0*np.transpose(np.tensordot(np.tensordot(val044, va_B, axes=(3,1)), va_B, axes=(2,1)), (0, 1, 3, 2)) + \
            1.0*val090 + \
            -1.0*np.transpose(val075, (1, 0, 2, 3)) + \
            -1.0*np.einsum("il,jk->iljk", val008, val015) + \
            -1.0*np.transpose(val080, (0, 1, 3, 2)) + \
            1.0*val085 + \
            1.0*np.transpose(val076, (1, 0, 2, 3)) + \
            -1.0*val091 + \
            -1.0*val089 + \
            1.0*np.tensordot(np.tensordot(np.tensordot(va_A, val035, axes=(1,1)), va_B, axes=(3,1)), va_B, axes=(2,1)) + \
            1.0*val073 + \
            1.0*val078 + \
            -1.0*np.transpose(val080, (1, 0, 2, 3)) + \
            1.0*val081 + \
            -1.0*val084 + \
            1.0*val068 + \
            -1.0*np.einsum("jk,il->iljk", val010, val008) + \
            -1.0*np.transpose(val077, (0, 1, 3, 2)) + \
            -1.0*np.transpose(val075, (0, 1, 3, 2)) + \
            -1.0*np.einsum("il,jk->iljk", val001, val014) + \
            -1.0*np.transpose(val077, (1, 0, 2, 3)) + \
            1.0*np.transpose(val092, (0, 1, 3, 2)) + \
            1.0*val087 + \
            1.0*val072 + \
            1.0*np.einsum("jk,il->iljk", val010, val003) + \
            1.0*val088 + \
            -1.0*np.transpose(val094, (0, 1, 3, 2)) + \
            1.0*np.einsum("il,jk->iljk", val001, val015) + \
            1.0*val071
    val096 = np.einsum("il,jk->ijkl", val001, val016)
    val097 = np.einsum("il,jk->ijkl", val001, val017)
    val098 = np.einsum("il,jk->ijkl", val001, val018)
    val099 = np.einsum("ik,jl->ijkl", val016, val003)
    val100 = np.einsum("ik,jl->ijkl", val016, val005)
    val101 = np.einsum("il,jk->ijkl", val001, val019)
    val102 = np.einsum("il,jk->ijkl", val001, val020)
    val103 = np.einsum("il,jk->ijkl", val001, val021)
    val104 = np.einsum("ik,jl->ijkl", val016, val008)
    val105 = np.einsum("ik,jl->ijkl", val016, val009)
    val106 = np.tensordot(va_A, val029, axes=(1,1))
    val107 = np.tensordot(val106, va_B, axes=(2,1))
    val108 = np.tensordot(np.tensordot(va_A, val027, axes=(1,2)), ua_B, axes=(3,1))
    val109 = np.tensordot(np.tensordot(va_A, val032, axes=(1,1)), va_B, axes=(3,1))
    val110 = np.tensordot(ua_A, val035, axes=(1,3))
    val111 = np.tensordot(val110, va_B, axes=(3,1))
    val112 = np.tensordot(val111, va_A, axes=(2,1))
    val113 = np.einsum("il,jk->ijkl", val003, val020)
    val114 = np.tensordot(np.tensordot(ua_A, np.tensordot(va_A, Gamma_2a, axes=(1,3)), axes=(1,3)), ua_B, axes=(3,1))
    val115 = np.tensordot(val114, va_A, axes=(2,1))
    val116 = (-np.tensordot(np.tensordot(ua_A, val039, axes=(1,3)), va_B, axes=(3,1)))
    val117 = np.tensordot(val116, ua_A, axes=(2,1))
    val118 = np.einsum("il,jk->ijkl", val005, val020)
    val119 = np.einsum("ik,jl->ijkl", val017, val009)
    val120 = np.tensordot(np.tensordot(val110, ua_B, axes=(2,1)), ua_A, axes=(2,1))
    val121 = np.einsum("ik,jl->ijkl", val018, val009)
    val122 = np.tensordot(ua_A, (-np.tensordot(ua_A, Gamma_2b, axes=(1,0))), axes=(1,1))
    val123 = np.tensordot(val122, va_B, axes=(3,1))
    val124 = np.tensordot(np.tensordot(ua_A, np.tensordot(ua_A, Gamma_4, axes=(1,2)), axes=(1,3)), ua_B, axes=(3,1))
    val125 = np.tensordot(np.tensordot(ua_A, val043, axes=(1,2)), va_B, axes=(3,1))
    val126 = np.einsum("il,jk->ijkl", val008, val020)
    val127 = np.einsum("il,jk->ijkl", val009, val020)
    val128 = np.einsum("ik,jl->ijkl", val019, val009)
    val129 = np.einsum("ik,jl->ijkl", val021, val009)
    gamma2_a = -1.0*np.transpose(val128, (1, 0, 2, 3)) + \
            1.0*np.transpose(val115, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val096, (1, 0, 2, 3)) + \
            1.0*np.transpose(val119, (1, 0, 2, 3)) + \
            1.0*val127 + \
            1.0*val102 + \
            1.0*np.transpose(val120, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val102, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val104, (1, 0, 2, 3)) + \
            1.0*np.transpose(val100, (1, 0, 2, 3)) + \
            -1.0*np.tensordot(val107, va_A, axes=(2,1)) + \
            1.0*val097 + \
            -1.0*val117 + \
            1.0*np.tensordot(val109, ua_A, axes=(2,1)) + \
            1.0*np.transpose(val117, (1, 0, 2, 3)) + \
            1.0*np.transpose(val126, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val127, (1, 0, 2, 3)) + \
            1.0*np.transpose(val099, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val113, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val112, (1, 0, 2, 3)) + \
            1.0*val096 + \
            -1.0*np.transpose(val118, (1, 0, 2, 3)) + \
            -1.0*val126 + \
            1.0*val113 + \
            1.0*np.transpose(val118, (3, 1, 2, 0)) + \
            -1.0*val119 + \
            1.0*np.transpose(val129, (1, 0, 2, 3)) + \
            1.0*val104 + \
            1.0*np.transpose(val101, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val097, (1, 0, 2, 3)) + \
            1.0*val118 + \
            1.0*np.transpose(val105, (1, 0, 2, 3)) + \
            1.0*np.transpose(val121, (1, 0, 2, 3)) + \
            1.0*np.tensordot(np.tensordot(val106, ua_B, axes=(3,1)), ua_A, axes=(2,1)) + \
            -1.0*val115 + \
            1.0*val098 + \
            1.0*val128 + \
            -1.0*val101 + \
            1.0*np.tensordot(np.tensordot(val122, ua_B, axes=(2,1)), ua_A, axes=(2,1)) + \
            1.0*val112 + \
            -1.0*val099 + \
            -1.0*np.transpose(val098, (1, 0, 2, 3)) + \
            -1.0*val129 + \
            -1.0*np.tensordot(val108, va_A, axes=(2,1)) + \
            1.0*np.transpose(val100, (3, 0, 2, 1)) + \
            -1.0*val100 + \
            1.0*val103 + \
            -1.0*val120 + \
            -1.0*val121 + \
            -1.0*np.transpose(val118, (1, 3, 2, 0)) + \
            -1.0*np.tensordot(val125, ua_A, axes=(2,1)) + \
            -1.0*val105 + \
            -1.0*np.transpose(val103, (1, 0, 2, 3)) + \
            1.0*np.tensordot(val124, va_A, axes=(2,1)) + \
            -1.0*np.transpose(val100, (0, 3, 2, 1)) + \
            -1.0*np.tensordot(val123, va_A, axes=(2,1))
    val130 = np.dot(vc_B, uc_A.T)
    val131 = np.einsum("jk,il->ijkl", val130, val010)
    val132 = np.einsum("ik,jl->ijkl", val130, val011)
    val133 = np.einsum("ik,jl->ijkl", val130, val012)
    val134 = np.einsum("ik,jl->ijkl", val130, val014)
    val135 = np.einsum("ik,jl->ijkl", val130, val015)
    val136 = val017.T
    val137 = np.einsum("il,jk->ijkl", val010, val136)
    val138 = val018.T
    val139 = np.einsum("il,jk->ijkl", val010, val138)
    val140 = np.dot(ua_B, va_A.T)
    val141 = np.einsum("il,jk->ijkl", val010, val140)
    val142 = np.tensordot(va_B, val053, axes=(1,1))
    val143 = np.tensordot(ua_B, val057, axes=(1,2))
    val144 = np.tensordot(np.tensordot(val143, va_A, axes=(2,1)), va_B, axes=(2,1))
    val145 = np.einsum("il,jk->ijkl", val011, val140)
    val146 = np.tensordot((-np.tensordot(np.tensordot(ua_B, val060, axes=(1,3)), va_A, axes=(3,1))), ua_B, axes=(2,1))
    val147 = np.einsum("il,jk->ijkl", val012, val140)
    val148 = np.einsum("ik,jl->ijkl", val136, val015)
    val149 = np.tensordot(ua_B, (-np.tensordot(ua_B, Gamma_2a, axes=(1,0))), axes=(1,1))
    val150 = np.einsum("il,jk->ijkl", val014, val140)
    val151 = np.einsum("jk,il->ijkl", val140, val015)
    val152 = np.einsum("ik,jl->ijkl", val138, val015)
    val153 = val019.T
    val154 = np.einsum("il,jk->ijkl", val010, val153)
    val155 = val021.T
    val156 = np.einsum("il,jk->ijkl", val010, val155)
    val157 = np.tensordot(np.tensordot(np.tensordot(ua_B, np.tensordot(va_B, Gamma_2b, axes=(1,3)), axes=(1,3)), ua_A, axes=(3,1)), va_B, axes=(2,1))
    val158 = np.tensordot(np.tensordot(val143, ua_A, axes=(3,1)), ua_B, axes=(2,1))
    val159 = np.einsum("ik,jl->ijkl", val153, val015)
    val160 = np.einsum("ik,jl->ijkl", val155, val015)
    gamma2_b = 1.0*np.transpose(val135, (1, 0, 2, 3)) + \
            -1.0*val157 + \
            -1.0*val156 + \
            1.0*np.tensordot(np.tensordot(np.tensordot(ua_B, np.tensordot(ua_B, Gamma_4, axes=(1,0)), axes=(1,1)), ua_A, axes=(3,1)), va_B, axes=(2,1)) + \
            -1.0*val137 + \
            -1.0*np.tensordot(np.tensordot(np.tensordot(ua_B, val063, axes=(1,2)), va_A, axes=(3,1)), ua_B, axes=(2,1)) + \
            1.0*np.transpose(val147, (1, 3, 2, 0)) + \
            -1.0*np.transpose(val145, (1, 0, 2, 3)) + \
            -1.0*np.tensordot(np.tensordot(val149, va_A, axes=(3,1)), va_B, axes=(2,1)) + \
            -1.0*val139 + \
            -1.0*val146 + \
            1.0*np.transpose(val132, (1, 0, 2, 3)) + \
            1.0*np.transpose(val133, (0, 3, 2, 1)) + \
            1.0*val145 + \
            -1.0*np.transpose(val133, (3, 0, 2, 1)) + \
            1.0*val148 + \
            -1.0*val132 + \
            1.0*np.transpose(val156, (1, 0, 2, 3)) + \
            1.0*val134 + \
            -1.0*val150 + \
            1.0*np.transpose(val147, (1, 0, 2, 3)) + \
            -1.0*val159 + \
            -1.0*np.tensordot(np.tensordot(np.tensordot(va_B, val052, axes=(1,2)), ua_A, axes=(3,1)), va_B, axes=(2,1)) + \
            -1.0*np.transpose(val141, (1, 0, 2, 3)) + \
            1.0*np.transpose(val139, (1, 0, 2, 3)) + \
            -1.0*val135 + \
            1.0*np.tensordot(np.tensordot(val149, ua_A, axes=(2,1)), ua_B, axes=(2,1)) + \
            1.0*np.transpose(val150, (1, 0, 2, 3)) + \
            1.0*np.transpose(val158, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val147, (3, 1, 2, 0)) + \
            1.0*val160 + \
            1.0*np.transpose(val157, (1, 0, 2, 3)) + \
            1.0*np.transpose(val137, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val152, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val154, (1, 0, 2, 3)) + \
            1.0*np.transpose(val146, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val133, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val144, (1, 0, 2, 3)) + \
            -1.0*val147 + \
            -1.0*np.transpose(val131, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val148, (1, 0, 2, 3)) + \
            1.0*val144 + \
            1.0*np.tensordot(np.tensordot(val142, ua_A, axes=(3,1)), ua_B, axes=(2,1)) + \
            -1.0*np.transpose(val160, (1, 0, 2, 3)) + \
            -1.0*np.tensordot(np.tensordot(val142, va_A, axes=(2,1)), va_B, axes=(2,1)) + \
            1.0*val131 + \
            1.0*val141 + \
            1.0*val152 + \
            -1.0*np.transpose(val134, (1, 0, 2, 3)) + \
            1.0*val133 + \
            1.0*np.transpose(val159, (1, 0, 2, 3)) + \
            1.0*val151 + \
            1.0*np.tensordot(np.tensordot(np.tensordot(va_B, val055, axes=(1,3)), va_A, axes=(3,1)), ua_B, axes=(2,1)) + \
            1.0*val154 + \
            -1.0*val158 + \
            -1.0*np.transpose(val151, (1, 0, 2, 3))
    val161 = np.tensordot(val107, ua_B, axes=(2,1))
    val162 = np.tensordot(val116, va_B, axes=(2,1))
    val163 = np.tensordot(val111, ua_B, axes=(2,1))
    val164 = np.tensordot(val114, ua_B, axes=(2,1))
    val165 = np.tensordot(val123, ua_B, axes=(2,1))
    gamma4 = -1.0*val066 + \
            1.0*val070 + \
            1.0*np.transpose(val068, (1, 0, 2, 3)) + \
            -1.0*val164 + \
            -1.0*np.transpose(val163, (1, 0, 2, 3)) + \
            1.0*np.transpose(val078, (1, 0, 2, 3)) + \
            1.0*np.transpose(val072, (0, 1, 3, 2)) + \
            1.0*np.transpose(val068, (0, 1, 3, 2)) + \
            1.0*np.transpose(val162, (1, 0, 2, 3)) + \
            1.0*np.transpose(val069, (0, 1, 3, 2)) + \
            -1.0*val071 + \
            1.0*np.transpose(val081, (1, 0, 2, 3)) + \
            1.0*np.transpose(val090, (0, 1, 3, 2)) + \
            1.0*val084 + \
            -1.0*val086 + \
            -1.0*val095 + \
            -1.0*val074 + \
            -1.0*val087 + \
            1.0*np.transpose(val165, (0, 1, 3, 2)) + \
            1.0*np.tensordot(val109, va_B, axes=(2,1)) + \
            1.0*np.transpose(val161, (0, 1, 3, 2)) + \
            -1.0*np.transpose(val089, (1, 0, 2, 3)) + \
            -1.0*val085 + \
            -1.0*val093 + \
            -1.0*val088 + \
            -1.0*np.transpose(val089, (0, 1, 3, 2)) + \
            1.0*np.transpose(val078, (0, 1, 3, 2)) + \
            1.0*np.transpose(val071, (1, 0, 2, 3)) + \
            -1.0*val162 + \
            -1.0*val069 + \
            1.0*np.transpose(val163, (1, 0, 3, 2)) + \
            1.0*np.transpose(val164, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val070, (1, 0, 2, 3)) + \
            1.0*val091 + \
            1.0*np.transpose(val071, (0, 1, 3, 2)) + \
            1.0*np.tensordot(val124, ua_B, axes=(2,1)) + \
            -1.0*np.tensordot(val108, ua_B, axes=(2,1)) + \
            -1.0*val073 + \
            -1.0*np.transpose(val163, (0, 1, 3, 2)) + \
            -1.0*val078 + \
            -1.0*val081 + \
            -1.0*val165 + \
            1.0*np.transpose(val066, (0, 1, 3, 2)) + \
            1.0*val163 + \
            1.0*np.transpose(val069, (1, 0, 2, 3)) + \
            -1.0*val068 + \
            -1.0*val161 + \
            1.0*np.transpose(val072, (1, 0, 2, 3)) + \
            1.0*np.transpose(val081, (0, 1, 3, 2)) + \
            1.0*np.transpose(val093, (0, 1, 3, 2)) + \
            -1.0*np.transpose(val070, (0, 1, 3, 2)) + \
            1.0*np.transpose(val093, (1, 0, 2, 3)) + \
            -1.0*val090 + \
            -1.0*val072 + \
            -1.0*np.tensordot(val125, va_B, axes=(2,1)) + \
            1.0*val089
    return (np.asarray([rho_a, rho_b]), -kappa_ba_T), \
            (np.asarray([gamma0_a, gamma0_b, gamma0_ab]), \
            np.asarray([gamma2_a, gamma2_b]), gamma4)
