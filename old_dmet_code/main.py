import numpy as np
import numpy.linalg as la
from inputs import Input, dump_input
from geometry import BuildLatticeFromInput, Topology
from ham import Hamiltonian
from mean_field import init_mfd
from embedding import EmbSolver
from BCSdmet import BCSdmet, BCSDmetResult
from chempot_fit import Fit_ChemicalPotential_MF, Fit_ChemicalPotential_Emb, Fit_ChemicalPotential_Emb_special
from timer import *
from diis import FDiisContext
from utils import ToClass
import sys

def ChooseRoutine(inp_dmet, inp_ctrl, lattice, topo):
  if inp_dmet.CalcType == "BCS":
    return BCSdmet(inp_dmet, inp_ctrl, lattice, topo)
  else:
    raise Exception("DMET Calculation Type not define")

def main(InputDict):
  timer_all = Timer()
  timer_all.start()
  np.random.seed(1437)
  np.set_printoptions(precision = 12, suppress = True, threshold = 10000, linewidth = np.inf)

  # parse input dictionary
  Inp = Input(InputDict)
  Inp.gen_control_options()
  verbose = Inp.FORMAT.Verbose
  OrbType = Inp.DMET.OrbType
  if verbose > 0:
    print "\nJob Summary"
    print Inp
  sys.stdout.flush()
  
  # build lattice and hamitlonian
  Lattice = BuildLatticeFromInput(Inp.GEOMETRY, OrbType, verbose)
  Topo = Topology(Lattice)
  Ham = Hamiltonian(Inp.HAMILTONIAN, Inp.CTRL, Lattice)
  Lattice.set_Hamiltonian(Ham)

  # define computation type
  Dmet = ChooseRoutine(Inp.DMET, Inp.CTRL, Lattice, Topo)
  
  # set up mean field solver
  MfdSolver = init_mfd(Inp.MFD, Inp.CTRL, Lattice, Ham, Dmet.trans_lat_int2e, Inp.DMET.OrbType)
  
  # set up impurity solver
  ImpSolver = EmbSolver(Inp.IMPSOLVER, OrbType, \
      lambda basis, vcor, mu: Dmet.MakeEmbCoreHam(basis, vcor, mu, Lattice, MfdSolver), \
      lambda basis, format: Dmet.MakeEmbIntHam(basis, format, Lattice))
  
  # set up diis
  dc = FDiisContext(Inp.DMET.DiisDim)

  # get initial guess
  Vcor = Dmet.GuessVcor(Lattice, Interaction=Ham.get_Int2e())
  Mu = Inp.DMET.InitMu
  last_err = 0.
  EmbResult = None
  Conv = False

  if verbose > 1:
    print "Initial Guess"
    print "Vloc  ="
    print Vcor[0]
    print "Delta ="
    print Vcor[1]
    print "Mu    = %20.12f" % Mu
    print
  
  if Inp.FITTING.MFD_Mu_Stop != 0:
    func_mu = lambda mu: MfdSolver.run(Vcor, mu, verbose-2).n
    Mu += Fit_ChemicalPotential_MF(func_mu, Mu, Dmet.occ * Lattice.supercell.nsites, Inp.FITTING, verbose)
  
  IterationHistory = []
  IterationHistory.append("\n  Iter.        Energy               Nelec.                d[V]        DIIS")
  for iter in range(Inp.DMET.MaxIter):
    timer = Timers()
    timer.start("Iter")
    if verbose > 1:
      print "-" * 40
      print "DMET Iteration %2d" % iter
      print "-" * 40, "\n"
    
    timer.start("Mfd")
    MfdResult = MfdSolver.run(Vcor, Mu, verbose)
    # n, energy, rho, kappa, mu, gap
    if verbose > 2:
      print "Mean-field Density Matrix (Local)"
      print MfdResult.rho[0]
      print
      print "Mean-field Pairing Matrix (Local)"
      print MfdResult.kappa[0]
      print

    timer.end("Mfd")
    sys.stdout.flush()

    timer.start("MkBasis")
    EmbBasis = Dmet.MakeEmbBasis(MfdResult.rho, MfdResult.kappa, Lattice, verbose)
    # u, v
    timer.end("MkBasis")
    sys.stdout.flush()
    timer.start("Localization")
    EmbBasis = Dmet.Localize(EmbBasis, Lattice, verbose)
    timer.end("Localization")
    sys.stdout.flush()

    timer.start("EmbCalc")

    target_n = Dmet.occ * Lattice.supercell.nsites

    if EmbResult is None or abs(float(EmbResult.n)/target_n-1.) < 1e-3:
      # nelec already very close, in this case, probably we don't have to improve mu
      if verbose > 0:
        print "\nChemical Potential = %20.12f" % Mu
      EmbResult = ImpSolver.run(EmbBasis, Vcor, Mu, Mu, verbose)
      
      if abs(float(EmbResult.n)/target_n-1.) > 5e-4 and (Inp.FITTING.EMB_Mu_Stop  < 0 or iter < Inp.FITTING.EMB_Mu_Stop): # see if it works
        dmu = Fit_ChemicalPotential_Emb_special(lambda mu: ImpSolver.run(EmbBasis, Vcor, mu, Mu, verbose, True).n, Mu, \
            target_n, EmbResult.n, Inp.FITTING, verbose)
        Mu += dmu
        Vcor[0] += dmu * np.eye(Vcor[0].shape[0])
        
        if verbose > 0:
          print "Chemical Potential = %20.12f" % Mu
        EmbResult = ImpSolver.run(EmbBasis, Vcor, Mu, Mu, verbose)
    
    else:
      if iter < Inp.FITTING.EMB_Mu_Stop or Inp.FITTING.EMB_Mu_Stop  < 0:
        dmu = Fit_ChemicalPotential_Emb(lambda mu: ImpSolver.run(EmbBasis, Vcor, mu, Mu, verbose, True).n, Mu, \
            target_n, Inp.FITTING, verbose)
        Mu += dmu
        Vcor[0] += dmu * np.eye(Vcor[0].shape[0])
      
      if verbose > 0:
        print "Chemical Potential = %20.12f" % Mu
      EmbResult = ImpSolver.run(EmbBasis, Vcor, Mu, Mu, verbose)
    
    if verbose > 2:
      print "Embedded Result: Density Matrix (Local)"
      print EmbResult.rho_frag
      print
      print "Embedded Result: Pairing Matrix (Local)"
      print EmbResult.kappa_frag
      print
    timer.end("EmbCalc")
    sys.stdout.flush()

    timer.start("FitPotential")
    dVcor, dmu, err = Dmet.FitCorrPotential(EmbResult, EmbBasis, Vcor, Mu, Inp.FITTING, Lattice, MfdSolver, iter, verbose)
    dVmax = max([la.norm(dVcor[0], np.inf), la.norm(dVcor[1], np.inf), abs(dmu)])
    derr = err - last_err
    if verbose > 0:
      print "Rdm Error Change  = %20.12f" % derr
    last_err = err
    timer.end("FitPotential")
    
    IterationHistory.append(" %3d %20.12f %20.12f %20.12f  %2d %2d" % (iter, EmbResult.E, EmbResult.n, dVmax, dc.nDim, dc.iNext))

    if verbose > 0:
      print "\nFitting Progress"
      for line in IterationHistory:
        print line
      print 
    sys.stdout.flush()

    if iter >= Inp.DMET.MinIter and la.norm(dVcor[0], np.inf) < Inp.DMET.ConvThrVcor and \
        la.norm(dVcor[1], np.inf) < Inp.DMET.ConvThrVcor and abs(dmu) < Inp.DMET.ConvThrMu and abs(derr) < Inp.DMET.ConvThrRdm:
      Conv = True

    timer.start("DIIS")
    if not Conv:
      if iter >= Inp.FITTING.TraceStart:
        SkipDiis = iter < Inp.DMET.DiisStart and dVmax > Inp.DMET.DiisThr
        Vcor, Mu, dVcor, dmu, c0 = dc.ApplyBCS(Vcor, Mu, dVcor, dmu, Skip = SkipDiis)
        if not SkipDiis and verbose > 1:
          print "Vcor Extrapolation: DIIS %4d %4d %20.12f\n" % (dc.nDim, dc.iNext, c0)
      Vcor = [Vcor[0] + dVcor[0], Vcor[1] + dVcor[1]]
      Mu += dmu
    timer.end("DIIS")
    timer.end("Iter")

    if verbose > 1:
      print "Time of Iteration            %6.2f s" % timer("Iter")
      print "Mean Field Calculations      %6.2f s" % timer("Mfd")
      print "Make Embedding Basis         %6.2f s" % timer("MkBasis")
      print "Embedding Basis Localization %6.2f s" % timer("Localization")
      print "Impurity Solver              %6.2f s" % timer("EmbCalc")
      print "Fit Local Potentials         %6.2f s" % timer("FitPotential")
      print "DIIS Extrapolation           %6.2f s" % timer("DIIS")
      print
      print "Total Elapsed Time           %6.2f s" % timer_all.get_time()
      print
    sys.stdout.flush()

    if (Inp.FORMAT.Walltime is not None and Inp.FORMAT.Walltime - timer_all.get_time() < timer("Iter")):
      print "time remained before walltime is %d seconds,\nprobably not enough for another cycle, which takes about %d seconds" \
          % (Inp.FORMAT.Walltime - timer_all.get_time(), timer("Iter"))
      break

    if Conv:
      break

  if Conv:
    print "-------- DMET converged --------\n"
  else:
    print "DMET program will terminate, please restart and continue"
    print
    print "----- DMET Restart Information -----"
    print
    MfdSolver.info(Mu, Vcor)
    print "DMRG restart information stored to %s" % ImpSolver.prepare_restart_info()
    print
    print "------ DMET NOT converged ------\n"
  
  if OrbType == "UHFB":
    print "Final V_loc (electric)"
    print (Vcor[0][::2, ::2] + Vcor[0][1::2, 1::2]) / 2
    print "Final V_loc (spin)"      
    print (Vcor[0][::2, ::2] - Vcor[0][1::2, 1::2]) / 2
    print "Final Delta_loc"
    print Vcor[1]
    print
    print "Final Fragment RDM (electric)"
    print (EmbResult.rho_frag[::2, ::2] + EmbResult.rho_frag[1::2, 1::2]) / 2
    print "Final Fragment RDM (spin)"
    print (EmbResult.rho_frag[::2, ::2] - EmbResult.rho_frag[1::2, 1::2]) / 2
  else:
    print "Final V_loc"
    print Vcor[0]
    print "Final Delta_loc"
    print Vcor[1]
    print
    print "Final Fragment RDM"
    print EmbResult.rho_frag
  print "Final Fragment Pairing Matrix"
  print EmbResult.kappa_frag
  if Inp.IMPSOLVER.DoubleOcc:
    print "Double Occupancy"
    print EmbResult.docc
  sys.stdout.flush()

  ImpSolver.CleanUp()

  return Inp, BCSDmetResult(EmbResult, MfdResult, Vcor, Mu, dVcor, dmu, Lattice, Conv, iter)
