H0_H0 = 1.0*h_0
val001 = np.dot(u_A.T, h_A)
val002 = np.dot(u_B.T, h_B)
H0_H1 = 1.0*np.trace(np.dot(val001, u_A)) + \
        2.0*np.trace(np.dot(np.dot(u_A.T, D), v_B)) + \
        1.0*np.trace(np.dot(val002, u_B))
val003 = np.dot(u_A, u_A.T)
val004 = np.tensordot(val003, w_A, axes=((0, 1),(0, 1)))
val005 = np.tensordot(val003, w_A, axes=((0, 1),(0, 2)))
val006 = np.dot(u_B, u_B.T)
val007 = np.tensordot(val006, w_B, axes=((0, 1),(0, 1)))
val008 = np.tensordot(val006, w_B, axes=((0, 1),(0, 2)))
val009 = np.dot(u_A, v_B.T)
val010 = np.tensordot(val009, w_AB, axes=((0, 1),(0, 2)))
val011 = np.tensordot(val003, w_AB, axes=((0, 1),(0, 1)))
val012 = (-np.tensordot(val009, x, axes=((0, 1),(0, 2))))
H0_H2 = 1.0*np.trace(np.dot(val012, val009.T)) + \
        0.5*np.trace(np.dot(val004, val003)) + \
        1.0*np.trace(np.dot(val011, val006)) + \
        1.0*np.trace(np.dot(val010, val009.T)) + \
        0.5*np.trace(np.dot(val007, val006)) + \
        -0.5*np.trace(np.dot(val005, val003)) + \
        2.0*np.trace(np.dot(np.tensordot(val003, y_A, axes=((0, 1),(0, 3))), val009.T)) + \
        2.0*np.trace(np.dot(np.tensordot(val006, y_B, axes=((0, 1),(0, 3))), np.dot(v_A, u_B.T))) + \
        -0.5*np.trace(np.dot(val008, val006))
val013 = np.dot(v_A.T, h_A)
val014 = np.dot(v_A.T, D)
H1D_H1 = 1.0*np.dot(val013, u_A) + \
        1.0*np.dot(np.dot(u_B.T, D.T), u_A) + \
        1.0*np.dot(val014, v_B) + \
        -1.0*np.dot(val002, v_B)
val015 = np.dot(val014, u_B)
H1A_H1 = 1.0*val015.T + \
        -1.0*np.dot(val002, u_B) + \
        1.0*np.dot(val013, v_A) + \
        1.0*val015
val016 = np.dot(np.dot(v_B.T, D.T), u_A)
H1B_H1 = -1.0*np.dot(val001, u_A) + \
        -1.0*val016 + \
        -1.0*val016.T + \
        1.0*np.dot(np.dot(v_B.T, h_B), v_B)
val017 = np.dot(v_A.T, val004)
val018 = np.dot(v_A.T, val005)
val019 = np.dot(u_B.T, val007)
val020 = np.dot(u_B.T, val008)
val021 = np.dot(v_A.T, val010)
val022 = np.tensordot(val006, w_AB, axes=((0, 1),(2, 3)))
val023 = np.dot(v_A.T, val022)
val024 = np.dot(u_B.T, val011)
H1D_H2 = 1.0*np.dot(val021, v_B) + \
        1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val006, y_B, axes=((0, 1),(0, 3)))).T), v_B) + \
        -1.0*np.dot(val018, u_A) + \
        -1.0*np.dot(val024, v_B) + \
        1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2))))), u_A) + \
        -1.0*np.dot(np.dot(u_B.T, (-np.tensordot(val003, y_A, axes=((0, 1),(0, 3)))).T), u_A) + \
        1.0*np.dot(np.dot(u_B.T, val010.T), u_A) + \
        1.0*np.dot(np.dot(u_B.T, (-np.tensordot(val006, y_B, axes=((0, 1),(0, 3))))), u_A) + \
        1.0*np.dot(np.dot(v_A.T, val012), v_B) + \
        1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2)))).T), u_A) + \
        -1.0*np.dot(np.dot(u_B.T, (-val012).T), u_A) + \
        -1.0*np.dot(np.dot(u_B.T, (-np.tensordot(np.dot(v_A, u_B.T), y_B, axes=((1, 0),(0, 2)))).T), v_B) + \
        1.0*np.dot(val020, v_B) + \
        1.0*np.dot(val023, u_A) + \
        -1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val003, y_A, axes=((0, 1),(0, 3))))), v_B) + \
        1.0*np.dot(val017, u_A) + \
        -1.0*np.dot(val019, v_B) + \
        -1.0*np.dot(np.dot(u_B.T, (-np.tensordot(np.dot(v_A, u_B.T), y_B, axes=((1, 0),(0, 2))))), v_B)
val025 = np.dot(val021, u_B)
H1A_H2 = 1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val006, y_B, axes=((0, 1),(0, 3)))).T), u_B).T + \
        1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2))))), v_A) + \
        -1.0*np.dot(np.dot(u_B.T, (-np.tensordot(np.dot(v_A, u_B.T), y_B, axes=((1, 0),(0, 2)))).T), u_B) + \
        -1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val003, y_A, axes=((0, 1),(0, 3))))), u_B) + \
        1.0*val025 + \
        1.0*np.dot(val023, v_A) + \
        -1.0*(-np.dot(np.dot(v_A.T, (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2))))), v_A).T) + \
        1.0*np.dot(val020, u_B) + \
        -1.0*np.dot(val024, u_B) + \
        1.0*np.dot(np.dot(v_A.T, val012), u_B).T + \
        1.0*val025.T + \
        1.0*np.dot(np.dot(v_A.T, (-np.tensordot(val006, y_B, axes=((0, 1),(0, 3)))).T), u_B) + \
        -1.0*np.dot(val019, u_B) + \
        1.0*np.dot(np.dot(v_A.T, val012), u_B) + \
        1.0*np.dot(val017, v_A) + \
        1.0*(-np.dot(np.dot(v_A.T, (-np.tensordot(val003, y_A, axes=((0, 1),(0, 3))))), u_B).T) + \
        -1.0*np.dot(val018, v_A) + \
        -1.0*np.dot(np.dot(u_B.T, (-np.tensordot(np.dot(v_A, u_B.T), y_B, axes=((1, 0),(0, 2)))).T), u_B).T
val026 = np.dot(np.dot(u_A.T, val010), v_B)
H1B_H2 = 1.0*np.dot(np.dot(v_B.T, (-np.tensordot(val003, y_A, axes=((0, 1),(0, 3)))).T), u_A).T + \
        -1.0*np.dot(np.dot(u_A.T, (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2)))).T), u_A).T + \
        1.0*np.dot(np.dot(v_B.T, (-np.tensordot(val003, y_A, axes=((0, 1),(0, 3)))).T), u_A) + \
        -1.0*(-np.dot(np.dot(v_B.T, (-np.tensordot(np.dot(v_A, u_B.T), y_B, axes=((1, 0),(0, 2))))), v_B).T) + \
        -1.0*np.dot(np.dot(u_A.T, val022), u_A) + \
        -1.0*np.dot(np.dot(u_A.T, val004), u_A) + \
        -1.0*np.dot(np.dot(v_B.T, val008), v_B) + \
        -1.0*np.dot(np.dot(v_B.T, val012.T), u_A).T + \
        -1.0*val026 + \
        1.0*np.dot(np.dot(v_B.T, (-np.tensordot(np.dot(v_A, u_B.T), y_B, axes=((1, 0),(0, 2))))), v_B) + \
        -1.0*np.dot(np.dot(v_B.T, (-np.tensordot(val006, y_B, axes=((0, 1),(0, 3))))), u_A) + \
        -1.0*np.dot(np.dot(u_A.T, (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2)))).T), u_A) + \
        1.0*(-np.dot(np.dot(v_B.T, (-np.tensordot(val006, y_B, axes=((0, 1),(0, 3))))), u_A).T) + \
        1.0*np.dot(np.dot(v_B.T, val011), v_B) + \
        -1.0*np.dot(np.dot(v_B.T, val012.T), u_A) + \
        -1.0*val026.T + \
        1.0*np.dot(np.dot(v_B.T, val007), v_B) + \
        1.0*np.dot(np.dot(u_A.T, val005), u_A)
val027 = np.tensordot(u_A, w_A, axes=(0,0))
val028 = np.tensordot(np.tensordot(np.tensordot(u_A, val027, axes=(0,2)), v_A, axes=(3,0)), u_A, axes=(2,0))
val029 = np.tensordot(v_B, w_B, axes=(0,0))
val030 = np.tensordot(np.tensordot(np.tensordot(v_B, val029, axes=(0,2)), u_B, axes=(3,0)), v_B, axes=(2,0))
val031 = np.tensordot(v_B, w_AB, axes=(0,2))
val032 = np.tensordot(u_A, val031, axes=(0,1))
val033 = 0.5*np.tensordot(np.tensordot(np.tensordot(v_B, np.tensordot(v_B, y_B, axes=(0,0)), axes=(0,1)), u_B, axes=(3,0)), u_A, axes=(2,0)) + \
        0.5*np.transpose(val030, (1, 0, 2, 3)) + \
        -1.0*np.transpose(np.tensordot((-np.tensordot(np.tensordot(u_A, np.tensordot(v_B, y_B, axes=(0,3)), axes=(0,3)), u_B, axes=(3,0))), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.transpose(val028, (1, 0, 2, 3)) + \
        -0.5*np.tensordot(np.tensordot(np.tensordot(u_A, (-np.tensordot(u_A, y_A, axes=(0,0))), axes=(0,1)), v_A, axes=(3,0)), v_B, axes=(2,0)) + \
        -1.0*np.transpose(np.tensordot(np.tensordot(val032, v_A, axes=(2,0)), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(v_B, np.tensordot(v_B, x, axes=(0,2)), axes=(0,3)), v_A, axes=(3,0)), u_A, axes=(2,0)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(u_A, np.tensordot(u_A, x, axes=(0,0)), axes=(0,1)), u_B, axes=(3,0)), v_B, axes=(2,0)) + \
        1.0*np.transpose(np.tensordot(np.tensordot(val032, u_B, axes=(3,0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
        -0.5*val030 + \
        1.0*np.transpose(np.tensordot((-np.tensordot(np.tensordot(u_A, np.tensordot(v_B, y_A, axes=(0,2)), axes=(0,3)), v_A, axes=(3,0))), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(u_A, (-np.tensordot(u_A, y_A, axes=(0,0))), axes=(0,1)), u_B, axes=(2,0)), u_A, axes=(2,0)) + \
        -0.5*np.tensordot(np.tensordot(np.tensordot(v_B, np.tensordot(v_B, y_B, axes=(0,0)), axes=(0,1)), v_A, axes=(2,0)), v_B, axes=(2,0)) + \
        -0.5*val028
H2yB_H2 = val033 - np.transpose(val033, (1,0,2,3))
val034 = np.tensordot(v_A, w_A, axes=(0,0))
val035 = np.tensordot(np.tensordot(v_A, val034, axes=(0,2)), u_A, axes=(3,0))
val036 = np.tensordot(val035, v_A, axes=(2,0))
val037 = np.tensordot(u_B, w_B, axes=(0,0))
val038 = np.tensordot(np.tensordot(u_B, val037, axes=(0,2)), v_B, axes=(3,0))
val039 = np.tensordot(val038, u_B, axes=(2,0))
val040 = np.tensordot(v_A, w_AB, axes=(0,0))
val041 = np.tensordot(u_B, val040, axes=(0,2))
val042 = np.tensordot(val041, u_A, axes=(2,0))
val043 = np.tensordot(v_A, y_A, axes=(0,0))
val044 = np.tensordot(v_A, y_B, axes=(0,2))
val045 = np.tensordot(v_A, x, axes=(0,0))
val046 = 0.5*np.tensordot(np.tensordot(np.tensordot(u_B, (-np.tensordot(u_B, y_B, axes=(0,0))), axes=(0,1)), u_A, axes=(2,0)), u_B, axes=(2,0)) + \
        -0.5*np.tensordot(np.tensordot(np.tensordot(v_A, val043, axes=(0,1)), v_B, axes=(2,0)), v_A, axes=(2,0)) + \
        -0.5*np.tensordot(np.tensordot(np.tensordot(u_B, (-np.tensordot(u_B, y_B, axes=(0,0))), axes=(0,1)), v_B, axes=(3,0)), v_A, axes=(2,0)) + \
        1.0*np.transpose(np.tensordot((-np.tensordot(np.tensordot(u_B, val044, axes=(0,3)), v_B, axes=(3,0))), u_B, axes=(2,0)), (1, 0, 2, 3)) + \
        -1.0*np.transpose(np.tensordot(np.tensordot(val041, v_B, axes=(3,0)), v_A, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.transpose(val039, (1, 0, 2, 3)) + \
        1.0*np.transpose(np.tensordot(val042, u_B, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(v_A, val043, axes=(0,1)), u_A, axes=(3,0)), u_B, axes=(2,0)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(v_A, val045, axes=(0,1)), v_B, axes=(3,0)), u_B, axes=(2,0)) + \
        0.5*np.transpose(val036, (1, 0, 2, 3)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(u_B, np.tensordot(u_B, x, axes=(0,2)), axes=(0,3)), u_A, axes=(3,0)), v_A, axes=(2,0)) + \
        -0.5*val036 + \
        -0.5*val039 + \
        -1.0*np.transpose(np.tensordot((-np.tensordot(np.tensordot(u_B, np.tensordot(v_A, y_A, axes=(0,3)), axes=(0,3)), u_A, axes=(3,0))), v_A, axes=(2,0)), (1, 0, 2, 3))
H2yA_H2 = val046 - np.transpose(val046, (1,0,2,3))
val047 = np.tensordot(np.tensordot((-np.tensordot(u_A, np.tensordot(v_B, y_A, axes=(0,2)), axes=(0,1))), u_A, axes=(3,0)), u_A, axes=(2,0))
val048 = np.tensordot(np.tensordot(np.tensordot(v_B, np.tensordot(v_B, y_B, axes=(0,0)), axes=(0,3)), v_B, axes=(2,0)), u_A, axes=(2,0))
val049 = np.tensordot(np.tensordot((-np.tensordot(u_A, np.tensordot(v_B, x, axes=(0,2)), axes=(0,1))), v_B, axes=(3,0)), u_A, axes=(2,0))
val050 = 0.5*np.transpose(val048, (0, 1, 3, 2)) + \
        0.5*np.transpose(val047, (1, 0, 2, 3)) + \
        -1.0*np.tensordot(np.tensordot(np.tensordot(v_B, val031, axes=(0,3)), u_A, axes=(3,0)), u_A, axes=(2,0)) + \
        0.5*np.transpose(val047, (0, 1, 3, 2)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(v_B, val029, axes=(0,1)), v_B, axes=(3,0)), v_B, axes=(2,0)) + \
        0.25*np.transpose(val049, (0, 1, 3, 2)) + \
        0.5*np.transpose(val048, (1, 0, 2, 3)) + \
        0.25*np.transpose(val049, (1, 0, 2, 3)) + \
        0.5*np.tensordot(np.tensordot(np.tensordot(u_A, val027, axes=(0,1)), u_A, axes=(3,0)), u_A, axes=(2,0))
H2wB_H2 = val050 + np.transpose(val050, (2,3,0,1))
val051 = np.tensordot(v_A, val034, axes=(0,1))
val052 = np.tensordot(u_B, val037, axes=(0,1))
val053 = np.tensordot(v_A, val040, axes=(0,1))
val054 = np.tensordot(np.tensordot(np.tensordot(v_A, val043, axes=(0,3)), v_A, axes=(2,0)), u_B, axes=(2,0))
val055 = np.tensordot(np.tensordot((-np.tensordot(u_B, val044, axes=(0,1))), u_B, axes=(3,0)), u_B, axes=(2,0))
val056 = np.tensordot(np.tensordot((-np.tensordot(u_B, val045, axes=(0,2))), v_A, axes=(2,0)), u_B, axes=(2,0))
val057 = 0.5*np.tensordot(np.tensordot(val051, v_A, axes=(3,0)), v_A, axes=(2,0)) + \
        0.5*np.transpose(val055, (1, 0, 2, 3)) + \
        0.5*np.transpose(val054, (0, 1, 3, 2)) + \
        0.5*np.tensordot(np.tensordot(val052, u_B, axes=(3,0)), u_B, axes=(2,0)) + \
        -1.0*np.tensordot(np.tensordot(val053, u_B, axes=(3,0)), u_B, axes=(2,0)) + \
        0.25*np.transpose(val056, (0, 1, 3, 2)) + \
        0.5*np.transpose(val054, (1, 0, 2, 3)) + \
        0.5*np.transpose(val055, (0, 1, 3, 2)) + \
        0.25*np.transpose(val056, (1, 0, 2, 3))
H2wA_H2 = val057 + np.transpose(val057, (2,3,0,1))
val058 = np.tensordot(np.transpose(val035, (3, 0, 2, 1)), u_A, axes=(2,0))
val059 = np.tensordot(np.transpose(val038, (3, 0, 2, 1)), v_B, axes=(2,0))
val060 = np.tensordot(np.transpose(val042, (3, 1, 2, 0)), v_B, axes=(2,0))
val061 = 0.5*np.transpose(np.tensordot((-np.transpose(np.tensordot(np.tensordot(u_B, (-np.tensordot(u_B, y_B, axes=(0,0))), axes=(0,1)), u_A, axes=(2,0)), (3, 0, 2, 1))), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        -1.0*np.transpose(val060, (1, 0, 2, 3)) + \
        0.25*np.transpose(np.tensordot((-np.transpose(np.tensordot(np.tensordot(v_A, val045, axes=(0,1)), v_B, axes=(3,0)), (3, 0, 2, 1))), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.transpose(val059, (1, 0, 2, 3)) + \
        0.5*np.transpose(np.tensordot((-np.transpose(np.tensordot(np.tensordot(v_A, val043, axes=(0,1)), u_A, axes=(3,0)), (3, 0, 2, 1))), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.transpose(np.tensordot((-np.transpose((-np.tensordot(np.tensordot(u_B, val044, axes=(0,3)), v_B, axes=(3,0))), (3, 1, 2, 0))), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        0.5*np.transpose(val058, (1, 0, 2, 3)) + \
        0.5*np.transpose(np.tensordot(np.transpose((-np.tensordot(np.tensordot(u_B, np.tensordot(v_A, y_A, axes=(0,3)), axes=(0,3)), u_A, axes=(3,0))), (3, 1, 2, 0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
        0.25*np.transpose(np.tensordot((-np.transpose(np.tensordot(np.tensordot(u_B, np.tensordot(u_B, x, axes=(0,2)), axes=(0,3)), u_A, axes=(3,0)), (3, 0, 2, 1))), u_A, axes=(2,0)), (1, 0, 2, 3))
val062 = val061 - np.transpose(val061, (1,0,2,3))
H2x_H2 = val062 - np.transpose(val062, (0,1,3,2))
H2wAB_H2 = -1.0*np.transpose(val060, (1, 2, 0, 3)) + \
        -1.0*np.tensordot(np.tensordot(val052, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
        1.0*np.transpose(np.tensordot(np.tensordot(np.tensordot(u_B, val043, axes=(0,2)), u_A, axes=(3,0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
        -1.0*np.transpose(np.tensordot(np.tensordot((-np.tensordot(u_B, val045, axes=(0,2))), v_B, axes=(3,0)), u_A, axes=(2,0)), (0, 1, 3, 2)) + \
        -1.0*np.transpose(np.tensordot(np.tensordot((-np.tensordot(u_B, val045, axes=(0,2))), v_B, axes=(3,0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
        1.0*np.tensordot(np.tensordot(val053, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
        1.0*np.tensordot(np.tensordot(np.tensordot(u_B, np.tensordot(u_B, w_AB, axes=(0,2)), axes=(0,3)), u_A, axes=(3,0)), u_A, axes=(2,0)) + \
        -1.0*np.tensordot(np.tensordot(val051, u_A, axes=(3,0)), u_A, axes=(2,0)) + \
        -1.0*np.transpose(val060, (2, 1, 3, 0)) + \
        -1.0*(-np.transpose(np.tensordot(np.tensordot(np.tensordot(u_B, val043, axes=(0,2)), u_A, axes=(3,0)), u_A, axes=(2,0)), (0, 1, 3, 2))) + \
        1.0*(-np.transpose(np.tensordot(np.tensordot(np.tensordot(v_A, val043, axes=(0,3)), v_B, axes=(3,0)), u_A, axes=(2,0)), (0, 1, 3, 2))) + \
        1.0*np.transpose(val059, (1, 2, 0, 3)) + \
        1.0*np.transpose(np.tensordot(np.tensordot((-np.tensordot(u_B, np.tensordot(u_B, y_B, axes=(0,3)), axes=(0,1))), v_B, axes=(2,0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
        -1.0*np.transpose(np.tensordot(np.tensordot((-np.tensordot(u_B, val044, axes=(0,1))), v_B, axes=(2,0)), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
        -1.0*(-np.transpose(np.tensordot(np.tensordot((-np.tensordot(u_B, np.tensordot(u_B, y_B, axes=(0,3)), axes=(0,1))), v_B, axes=(2,0)), u_A, axes=(2,0)), (0, 1, 3, 2))) + \
        1.0*np.transpose(val058, (1, 2, 0, 3)) + \
        1.0*(-np.transpose(np.tensordot(np.tensordot((-np.tensordot(u_B, val044, axes=(0,1))), v_B, axes=(2,0)), v_B, axes=(2,0)), (0, 1, 3, 2))) + \
        -1.0*np.transpose(np.tensordot(np.tensordot(np.tensordot(v_A, val043, axes=(0,3)), v_B, axes=(3,0)), u_A, axes=(2,0)), (1, 0, 2, 3))
