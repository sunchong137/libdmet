import numpy as np
import numpy.linalg as la
import itertools as it
from copy import deepcopy

from utils import ToClass, mdot, ToSpinOrb, ToSpatOrb
from localize import PinLocalize, TriLocalize
from vcor_fit import *
from chempot_fit import Fit_ChemicalPotential_MF
from BCS_transform import *

class BCSDmetResult(object):
  def __init__(self, EmbResult = None, MfdResult = None, Vcor = None, Mu = None, dVcor = None, dmu = None, Lattice = None, Conv = None, Iter = None, dict = None):
    if dict is None:
      nsites = Lattice.supercell.nsites
      # fitting results
      self.Vcor = Vcor
      self.Mu = Mu
      # embedding results
      self.Energy = EmbResult.E/nsites
      self.Nelec = EmbResult.n/nsites * 2 # for both spins
      self.Rho = EmbResult.rho_frag
      self.Kappa = EmbResult.kappa_frag
      self.docc = EmbResult.docc
      # mfd results
      self.MfdEnergy = MfdResult.energy/nsites
      self.MfdNelec = MfdResult.n/nsites
      self.Gap = MfdResult.gap
      # fitting quality
      self.dVcor = max([la.norm(dVcor[0], np.inf), la.norm(dVcor[1], np.inf), abs(dmu)])  # dVmax
      if EmbResult.rho_emb.shape == EmbResult.kappa_emb.shape: # restricted
        self.EmbRdmErr = np.sqrt(np.sum(EmbResult.rho_emb**2)*2 + np.sum(EmbResult.kappa_emb**2)*2)        
      else:
        self.EmbRdmErr = np.sqrt(np.sum(EmbResult.rho_emb**2) + np.sum(EmbResult.kappa_emb**2)*2)

      self.Conv = Conv
      self.Iter = Iter
    else:
      self.__dict__ = dict

  def __str__(self):
    str = ""
    str += "\n    FITTING RESULTS:\n"
    str += "        Vloc      = " + self.Vcor[0].__repr__().replace("\n", "\n\t"+"            ") + "\n"
    str += "        Delta     = " + self.Vcor[1].__repr__().replace("\n", "\n\t"+"            ") + "\n"
    str += "        Mu        = %-20.12f\n" % self.Mu
    str += "\n    EMBEDDING RESULTS:\n"
    str += "        Rho       = " + self.Rho.__repr__().replace("\n", "\n\t"+"            ") + "\n"
    str += "        Kappa     = " + self.Kappa.__repr__().replace("\n", "\n\t"+"            ") + "\n"
    str += "        Energy    = %-20.12f\n" % self.Energy
    str += "        Nelec     = %-20.12f\n" % self.Nelec
    str += "        DoubleOcc = %s\n" % self.docc
    str += "\n    MEAN-FIELD RESULTS:\n"
    str += "        Energy    = %-20.12f\n" % self.MfdEnergy
    str += "        Nelec     = %-20.12f\n" % self.MfdNelec
    str += "        Gap       = %-20.12f\n" % self.Gap
    str += "\n    FITTING QUALITY\n"
    str += "        dVcor     = %-20.12f\n" % self.dVcor
    str += "        RdmErr    = %-20.12f\n" % self.EmbRdmErr
    str += "        Fit Iter  = %-3d\n" % self.Iter
    str += "        Converged = %s\n" % self.Conv
    return str

class BCSdmet(object):
  def __init__(self, inp_dmet, inp_ctrl, lattice, topo):
    self.UHFB = (inp_dmet.OrbType == "UHFB")
    self.useDelta = inp_dmet.UseDelta

    if self.useDelta:
      self.occ = inp_dmet.Filling
    else:
      print "Warning: Nelec rounded to nearest integer"
      nelec = int(inp_dmet.Filling * lattice.nsites+0.5)
      self.occ = float(nelec) / lattice.nsites

    self.guess_type = inp_dmet.InitGuessType
    self.guess = inp_dmet.InitGuess
    self.useDelta = inp_dmet.UseDelta
    self.local_method = inp_dmet.Localize
    self.control = inp_ctrl
    self.topology = topo
    self.trans_lat_int2e = LatticeIntegral(lattice, lattice.Ham.Int2e, \
        lattice.Ham.Int2eShape, self.UHFB, thr_rdm = inp_dmet.TransformRdmThr, \
        thr_int = inp_dmet.TransformIntThr)

  def MakeEmbBasis(self, rho, kappa, lattice, verbose):
    if verbose > 2:
      print "****************** Making Embedding Basis ******************\n"
    if len(lattice.supercell.fragments) != 1:
      raise Exception("Currently only 1 fragment of the whole supercell is implemented")
    norbs = lattice.supercell.nsites
    nsorbs = norbs*2 if self.UHFB else norbs
    ncells = lattice.nscells

    assert(rho.shape == (ncells, nsorbs, nsorbs))
    assert(kappa.shape == (ncells, norbs, norbs))

    if self.UHFB:
      u_embs = [None, None]
      v_embs = [None, None]
      rho1 = [np.array([rho[c][::2,::2] for c in range(ncells)]), \
          np.array([rho[c][1::2,1::2] for c in range(ncells)])]
    else:
      u_embs = [None]
      v_embs = [None]
      rho1 = [rho]
    spin = len(u_embs)
    
    for s in range(spin):
      kappa_positive = (spin == s+1)
      prdm = np.zeros((norbs*2, norbs*2))
      prdm[:norbs, :norbs] = np.eye(norbs) - rho1[spin-1-s][0]
      prdm[norbs:, norbs:] = rho1[s][0]
      if kappa_positive:
        prdm[:norbs, norbs:] = kappa[0]
        prdm[norbs:, :norbs] = kappa[0].T.conj()
      else:
        prdm[:norbs, norbs:] = -kappa[0].T.conj()
        prdm[norbs:, :norbs] = -kappa[0]

      ew, ev = la.eigh(prdm)
      inv_A = np.dot(ev, np.diag(1/np.sqrt(ew)))
      # now the columns containing fragment sites
      prdm_emb = np.zeros((norbs*ncells*2, norbs*2))
      for i in range(ncells):
        # because rho is always hermite
        prdm_emb[i*norbs:(i+1)*norbs, :norbs] = -rho1[spin-1-s][i].T.conj()
        prdm_emb[(i+ncells)*norbs:(i+1+ncells)*norbs, norbs:] = rho1[s][i].T.conj()
        # but kappa isn't hermite in UHFB case
        if kappa_positive:
          prdm_emb[i*norbs:(i+1)*norbs, norbs:] = kappa[lattice.sc_pos2idx(-lattice.sc_idx2pos(i))]
          prdm_emb[(i+ncells)*norbs:(i+1+ncells)*norbs:, :norbs] = kappa[i].T.conj()
        else:
          prdm_emb[i*norbs:(i+1)*norbs, norbs:] = -kappa[i].T.conj()
          prdm_emb[(i+ncells)*norbs:(i+1+ncells)*norbs, :norbs] = -kappa[lattice.sc_pos2idx(-lattice.sc_idx2pos(i))]

      prdm_emb[:norbs, :norbs] += np.eye(norbs)
      uv_emb = np.dot(prdm_emb, inv_A)
      u_embs[spin-1-s] = uv_emb[:norbs*ncells]
      v_embs[s] = uv_emb[norbs*ncells:]
    
    if self.UHFB:
      return ToClass({"u": u_embs, "v": v_embs})
    else:
      return ToClass({"u": u_embs[0], "v": v_embs[0]})

  def Localize(self, EmbBasis, lattice, verbose):
    if self.local_method == None:
      return EmbBasis

    if len(lattice.supercell.fragments) != 1:
      raise Exception("Currently only 1 fragment of the whole supercell is implemented")
    #if lattice.supercell.nsites != lattice.supercell.ncells:
    #  raise Exception("Currently only 1 site per unit cell is implemented")

    # wraper for basis localization methods
    elif self.local_method == "Tri" or self.local_method == "Pin":
      Tri = lambda u,v: TriLocalize(u, v, lattice.supercell.nsites, verbose)
      if self.UHFB:
        EmbBasis.u[0], EmbBasis.v[1] = Tri(EmbBasis.u[0], EmbBasis.v[1])
        EmbBasis.u[1], EmbBasis.v[0] = Tri(EmbBasis.u[1], EmbBasis.v[0])
      else:
        EmbBasis.u, EmbBasis.v = Tri(EmbBasis.u, EmbBasis.v)
    
    if self.local_method == "Pin":
      Pin = lambda u,v: PinLocalize(u, v, self.topology, verbose)
      if self.UHFB:
        EmbBasis.u[0], EmbBasis.v[1] = Pin(EmbBasis.u[0], EmbBasis.v[1])
        EmbBasis.u[1], EmbBasis.v[0] = Pin(EmbBasis.u[1], EmbBasis.v[0])
      else:
        EmbBasis.u, EmbBasis.v = Pin(EmbBasis.u, EmbBasis.v)
    return EmbBasis

  def MakeEmbCoreHam(self, basis, vcor, mu, lattice, mfd):
    """
    The general procedure of generating core Hamiltonian
      1. transform Fock (H1, real or stored) from mfd
      2. add mu to the core Hamiltonian
      3. if BathVcor == True: add Vcor in environment
      4. if EffEnv == stored/computed: minus effective vcor in impurity
      5. if EffCore == computed: minus effective vcor in emb
    The fragment Hamiltonian is just bare H0
    """
    Fock = mfd.FockT
    # step 1
    if self.UHFB:
      Hemb, Demb = transform_trans_inv(basis, lattice, Fock[0], Fock[2], Fock[1])
    else:
      Hemb, Demb = transform_trans_inv(basis, lattice, Fock[0], Fock[2])
    
    # step 2 -mu*I
    Htemp, Dtemp = transform_scalar(basis, lattice, -mu, self.UHFB)
    Hemb += Htemp; Demb += Dtemp
    
    # step 3 +Vcor in environment(bath)
    if self.control.BathVcor:
      # first add Vcor on all replicas of the impurity
      if self.UHFB:
        Htemp, Dtemp = transform_local(basis, lattice, vcor[0][::2, ::2], \
            vcor[1], vcor[0][1::2, 1::2])
      else:
        Htemp, Dtemp = transform_local(basis, lattice, vcor[0], vcor[1])
      Hemb += Htemp; Demb += Dtemp
      # then substract the impurity part      
      if self.UHFB:
        Htemp, Dtemp = transform_impurity(basis, lattice, vcor[0][::2, ::2], \
            vcor[1], vcor[0][1::2, 1::2])
      else:
        Htemp, Dtemp = transform_impurity(basis, lattice, vcor[0], vcor[1])
      Hemb -= Htemp; Demb -= Dtemp
    
    # step 4 -impurity correlation if needed
    if self.control.EffEnv == "Store":
      # the stored number is always restricted
      ImpCorr = lattice.Ham.get_imp_corr()
      if self.UHFB:
        Htemp, Dtemp = transform_impurity(basis, lattice, ImpCorr, \
            np.zeros_like(ImpCorr), ImpCorr)
      else:
        Htemp, Dtemp = transform_impurity(basis, lattice, ImpCorr, \
            np.zeros_like(ImpCorr))
      Hemb -= Htemp; Demb -= Dtemp
    elif self.control.EffEnv == "Comp":
      # first compute impurity correlation
      ImpCorr = mfd.compute_imp_corr()
      if self.UHFB:
        Htemp, Dtemp = transform_impurity(basis, lattice, ImpCorr[0], \
            ImpCorr[2], ImpCorr[1])
      else:
        Htemp, Dtemp = transform_impurity(basis, lattice, ImpCorr[0], \
            ImpCorr[1])
      Hemb -= Htemp; Demb -= Dtemp

    # step 5 -embedded system correlation if needed
    if self.control.EffCore:
      # compute embedded system correlation
      VAeff, VBeff, Deff = self.trans_lat_int2e(basis)
      Htemp, Dtemp = transform_full_lattice(basis, lattice, VAeff, Deff, VBeff)
      Hemb -= Htemp; Demb -= Dtemp
 
    # Now compute Hfrag, Dfrag, E0
    H0 = lattice.get_h0(SpinOrb = False)
    Hfrag, Dfrag, E0 = transform_imp_env(basis, lattice, H0, self.UHFB)
    Hfrag = 0.5*(Hfrag + Hfrag.T)
    if not self.UHFB:
      Dfrag = 0.5*(Dfrag + Dfrag.T)

    return ToClass({"H":Hemb, "D":Demb}), ToClass({"H":Hfrag, "D":Dfrag, "E0": E0})

  def MakeEmbIntHam(self, basis, format, lattice):
    if format == "MPS":
      assert(lattice.Ham.type == "Hubbard" and not self.control.BathInt2e)
      return transform_EmbIntHam_MPS(basis, lattice, self.UHFB)
    elif format == "DUMP":
      return transform_EmbIntHam_Dump(basis, lattice, self.trans_lat_int2e, \
          self.control.BathInt2e, self.UHFB)
    else:
      raise Exception("Hamiltonian representation type not available")

  def GuessVcor(self, lattice, Interaction = None):
    norbs = lattice.supercell.nsites
    nsorbs = norbs*2 if self.UHFB else norbs
    U = np.average([Interaction[i, i, i, i] for i in range(norbs)])

    def gen_rand_guess():
      Vloc = np.random.rand(norbs, norbs) * 0.1
      Vloc += Vloc.T
      Vloc -= np.diag(np.diag(Vloc))
      Dloc = np.random.rand(norbs, norbs) * 0.4
      if self.UHFB:
        Vloc = np.zeros((norbs*2, norbs*2))
        Vloc[::2, ::2] = np.random.rand(norbs, norbs) * 0.2
        Vloc[1::2, 1::2] = Vloc[::2, ::2]
        Vloc += Vloc.T
        Vloc[::2, ::2] -= np.average(Vloc[::2, ::2])        
        Vloc[1::2, 1::2] -= np.average(Vloc[1::2, 1::2])        
      else:
        Vloc = np.random.rand(norbs, norbs) * 0.2
        Vloc += Vloc.T
        Dloc = (Dloc+Dloc.T) / 2
        Vloc -= np.average(Vloc)
      return Vloc, Dloc
    
    if self.guess_type == "ZERO":
      if self.UHFB:
        Vloc = np.zeros((norbs*2, norbs*2))
      else:
        Vloc = np.zeros((norbs, norbs))
      Dloc = np.zeros((norbs, norbs))
    if self.guess_type == "RAND":
      Vloc, Dloc = gen_rand_guess()
      if self.control.Fock != "Comp":
        Vloc += np.diag([U*self.occ] * nsorbs)
    if self.guess_type == "PM":
      Vloc, Dloc = gen_rand_guess()
      Dloc = (Dloc + Dloc.T) / 2
      if self.control.Fock != "Comp":
        Vloc += np.diag([U*self.occ] * nsorbs)
    elif self.guess_type == "MAN":
      Vloc = self.guess[0]
      Dloc = self.guess[1]
      assert(Vloc.shape == (nsorbs, nsorbs))
      assert(Dloc.shape == (norbs, norbs))
    elif self.guess_type == "AF":
      print "\tWarning: automatic AF initial guess may not be physical"
      assert(self.UHFB)
      # generate AF-type local potential based on nearest neighbor relation
      def divide_sites():
        sites = range(norbs)
        neighbors = lattice.get_NearNeighbor(sites, sites)[0]
        sitesA = [sites[0]]
        sitesB = []
        sites = sites[1:]

        while len(sites) != 0:
          for site in sites:
            for pair in neighbors:
              if site in pair:
                if pair[1-pair.index(site)] in sitesA:
                  sitesB.append(site)
                  break
                elif pair[1-pair.index(site)] in sitesB:
                  sitesA.append(site)
                  break
          sites = [s for s in sites if not s in sitesA + sitesB]
        return sitesA, sitesB
    
      sitesA, sitesB = divide_sites()
      
      Vloc, Dloc = gen_rand_guess()
      Vloc[[2*x for x in sitesA], [2*x for x in sitesA]] += U*self.occ*2
      Vloc[[2*x+1 for x in sitesB], [2*x+1 for x in sitesB]] += U*self.occ*2
      if self.control.Fock != "None":
        Vloc -= np.eye(2*norbs) * U*self.occ
    
    if not self.useDelta:
      Dloc *= 0.

    return [Vloc, Dloc]

  def BuildEmbBdG(self, basis, vcor, mu, lattice, mfd):
    # FIXME
    if self.control.FitLevel == "EmbH1":
      assert(np.allclose(mfd.Fock[0] - mfd.H0, 0.))
    
    Fock = mfd.FockT
    if self.UHFB:
      Hemb, Demb = transform_trans_inv(basis, lattice, Fock[0], Fock[2], Fock[1])
    else:
      Hemb, Demb = transform_trans_inv(basis, lattice, Fock[0], Fock[2])
    
    Htemp, Dtemp = transform_scalar(basis, lattice, -mu, self.UHFB)
    Hemb += Htemp; Demb += Dtemp
    
    if self.UHFB:
      Htemp, Dtemp = transform_local(basis, lattice, vcor[0][::2, ::2], \
          vcor[1], vcor[0][1::2, 1::2])
    else:
      Htemp, Dtemp = transform_local(basis, lattice, vcor[0], vcor[1])
    Hemb += Htemp; Demb += Dtemp

    nscsites = lattice.supercell.nsites
    EmbBdG = np.zeros((nscsites*4, nscsites*4))
    
    if self.UHFB:
      EmbBdG[:nscsites*2, :nscsites*2] = Hemb[::2, ::2]
      EmbBdG[nscsites*2:, nscsites*2:] = -Hemb[1::2, 1::2]
      # in the SCF case, Demb may not be exactly 0
      EmbBdG[:nscsites*2, nscsites*2:] = Demb 
      EmbBdG[nscsites*2:, :nscsites*2] = Demb.T.conj()
    else:
      EmbBdG[:nscsites*2, :nscsites*2] = Hemb
      EmbBdG[nscsites*2:, nscsites*2:] = -Hemb
      EmbBdG[:nscsites*2, nscsites*2:] = Demb
      EmbBdG[nscsites*2:, :nscsites*2] = Demb.T.conj()
    return EmbBdG

  def FitCorrPotential(self, HlResult, basis, Vcor, Mu, inp_fit, lattice, mfd, iter, verbose):

    if self.UHFB:
      rdm_error = np.sqrt(np.sum(HlResult.rho_emb**2) + np.sum(HlResult.kappa_emb**2)*2)
    else:
      rdm_error = np.sqrt(np.sum(HlResult.rho_emb**2)*2 + np.sum(HlResult.kappa_emb**2)*2)

    if verbose > 1:
      print "****************** Fit Local Potential *******************\n"
      print "Embbeding Rdm Error  = %20.12f" % rdm_error

    if self.control.FitLevel in ["EmbH1", "EmbFock"]:
      bdg = self.BuildEmbBdG(basis, Vcor, Mu, lattice, mfd)
      dVcor, err = FitVcorEmb(lattice.supercell.nsites, HlResult, basis, bdg, \
          inp_fit, self.UHFB, verbose)
      dVloc, dDelta = dVcor
    elif self.control.FitLevel == "EmbSCF":
      raise Exception("Not implemented yet")
    elif self.control.FitLevel == "LatSCF":
      dVcor, err = FitVcorLatSCF(lattice, HlResult, basis, mfd, Vcor, Mu, \
          inp_fit, self.UHFB, verbose)
      dVloc, dDelta = dVcor
    else:
      raise Exception("Unrecognized fitting method: %s" % self.control.FitLevel)
    
    if (iter+1 < inp_fit.MFD_Mu_Stop or inp_fit.MFD_Mu_Stop < 0):
      if iter >= inp_fit.TraceStart:
        shift = np.average(np.diag(dVloc))
        dVloc -= shift * np.eye(dVloc.shape[0])
      VcorNew = [Vcor[0] + dVcor[0], Vcor[1] + dVcor[1]]
      func_mu = lambda mu: mfd.run(VcorNew, mu, verbose-2).n
      dmu = Fit_ChemicalPotential_MF(func_mu, Mu, self.occ * lattice.supercell.nsites, inp_fit, verbose)
    else:
      dmu = 0.
    
    if verbose > 1:
      print "dVloc (max)  = %20.12f" % la.norm(dVloc, np.inf)
      print "dDelta (max) = %20.12f" % la.norm(dDelta, np.inf)
      print "dMu          = %20.12f" % dmu
      print

    return [dVloc, dDelta], dmu, rdm_error
