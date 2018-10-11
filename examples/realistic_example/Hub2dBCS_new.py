import libdmet.utils.logger as log
import libdmet.dmet.HubbardBCS as dmet
import numpy as np
import numpy.linalg as la

log.verbose = "DEBUG0"

# System settings:
LatSize = (72, 72)
ImpSize = (4, 4)
U = 6.0 
Filling = 0.6 / 2.0 
Mu = U * Filling # initial guess of global Mu
last_dmu = -0.484630562247 
MaxIter = 50
maxM = 1200
LMO = True # LMO or site basis
save_vcor = True # save vcor or load vcor 
dump_file = './dmet.npy'

# DIIS settings:
DiisStart = 4 
TraceStart = 1 
DiisDim = 4 # ZHC NOTE may be larger?
dc = dmet.FDiisContext(DiisDim) # I don't know yet whether diis needs to be changed

# Construct system and Hamiltonian
nscsites = np.prod(ImpSize)
Lat = dmet.SquareLattice(*(LatSize + ImpSize))
Ham = dmet.Ham(Lat, U)
Lat.setHam(Ham)

#vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand = 0.001)
imp_size_small = (2, 2)
vcor = dmet.AFInitGuess(imp_size_small, U, Filling, rand = 0.0)

Mu, _, vcor_param, basis, \
        GRhoEmb, GRhoImp, EnergyImp, nelecImp = np.load('./dmet.npy')

vcor.update(vcor_param)
vcor = dmet.get_tiled_vcor(vcor, imp_size_small, ImpSize, rand = 0.001, U_Filling = (U, Filling))
if save_vcor:
    np.save('./vcor0_param.npy', vcor.param)
else:
    param0 = np.load('./vcor0_param.npy')
    vcor.update(param0)

# Solver
if LMO:
    ncas = nscsites * 2
    block = dmet.impurity_solver.StackBlock(nproc = 1, nthread = 28, nnode = 3, \
            bcs = True, tol = 1e-6, maxM = maxM, SharedDir = "./shdir", maxiter_initial = 36, maxiter_restart = 14)

    solver = dmet.impurity_solver.BCSDmrgCI(ncas = ncas, \
            cisolver = block, splitloc = True, algo = "energy",\
            doscf = True, mom_reorder = True) # ZHC NOTE check reorder
else:
    # ZHC TODO: add localization to site basis.
    solver = dmet.impurity_solver.StackBlock(nproc = 1, nthread = 28, nnode = 1, \
            bcs = True, reorder = True, tol = 1e-6, maxM = maxM)

# DMET main:
E_old = 0.0
conv = False
history = dmet.IterHistory()

for iter in range(MaxIter):
    log.section("\n----------------------------------------------------------")
    log.section("\nDMET Iteration %d\n", iter)
    log.section("----------------------------------------------------------\n")

    log.section ("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu, thrnelec = 1e-7)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H_energy, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu, localize_bath = False) 
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu) 
    log.section("\nsolving impurity problem\n")
    
    if LMO:
        if iter <= 2:
            solver.localized_cas = None
            solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis, "ci_args": {"restart": False}}
        elif iter <= 4:
            solver.localized_cas = None
            solver.cisolver.cisolver.optimized = False # not do restart among different dmet iters
            solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis}
        else:
            if dVcor_per_ele is not None:
                if dVcor_per_ele > 0.005:
                    solver.localized_cas = None
                    solver.cisolver.cisolver.optimized = False # not do restart among different dmet iters
            solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis}
    else:
        solver.cisolver.optimized = False # not do restart among different dmet iters
        solver_args = {}
    
    GRhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            delta = 0.004, step = 0.125, thrnelec = 2e-5,\
            solver_args = solver_args) 
            # ZHC NOTE be careful of delta and step
    dmet.SolveImpHam_with_fitting.save("./frecord")
    
    last_dmu += dmu
    log.result("last_dmu : %20.12f", last_dmu)
    
    GRhoImp, EnergyImp, nelecImp = \
            dmet.transformResults_new(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, H_energy, last_dmu, Mu)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
            MaxIter1 = max(len(vcor.param) * 5, 1000), MaxIter2 = 0,\
            triu = False, CG_check = True) # ZHC NOTE triu cost function and CG_check

    # ZHC NOTE add damping?
    if iter >= TraceStart:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        ddiagV = np.average(np.diagonal(\
                (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        vcor_new = dmet.addDiag(vcor_new, -ddiagV)

    dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
    dE = EnergyImp - E_old
    E_old = EnergyImp 

    # DIIS
    skipDiis = (iter < DiisStart) or (dVcor_per_ele > 0.01)
    pvcor, dpvcor, _ = dc.Apply( \
            vcor_new.param, \
            vcor_new.param - vcor.param, \
            Skip = skipDiis)
    
    dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
    vcor.update(pvcor)
    #print "trace of vcor: ", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2))
    
    history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
    history.write_table()
    dump_res_iter = np.array([Mu, last_dmu, vcor.param, GRhoEmb, basis], dtype = object)
    np.save('./dmet_iter.npy', dump_res_iter)


    # ZHC NOTE convergence criterion
    if dVcor_per_ele < 5.0e-5 and abs(dE) < 5.0e-4 and iter > 3 :
        conv = True
        break

if dump_file is not None:
    log.result("Dumping the results...")
    if dump_file[-4:] == '.npy':
        dump_res = np.array([Mu, last_dmu, vcor.param, basis, GRhoEmb, \
                GRhoImp, EnergyImp, nelecImp], dtype = object)
        np.save(dump_file, dump_res)
    elif dump_file[-5:] == '.hdf5':
        # ZHC TODO add hdf5 format
        raise NotImplementedError
    else:
        raise ValueError

# fix clean up
# ZHC TODO FIXME fix the cleanup function
#solver.cleanup()


if conv:
    log.result("DMET converged")
else:
    log.result("DMET cannot converge")