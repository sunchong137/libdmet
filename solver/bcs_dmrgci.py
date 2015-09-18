import numpy as np
import numpy.linalg as la
from libdmet.solver import block, scf
from libdmet.system import integral
import libdmet.utils.logger as log
from libdmet.utils.misc import mdot
from libdmet.routine.localizer import Localizer
from libdmet.utils.munkres import Munkres, make_cost_matrix
from libdmet.routine.bcs_helper import extractRdm, basisToCanonical, basisToSpin
from libdmet.integral.integral_emb_casci import transform
from libdmet.integral.integral_localize import transform as transform_local
from libdmet.solver.dmrgci import gaopt, momopt

def get_qps(casci, Ham, guess):
    # get quasiparticles by solving scf
    casci.scfsolver.set_system(None, 0, True, False)
    casci.scfsolver.set_integral(Ham)

    E_HFB, GRho_HFB = casci.scfsolver.HFB(Mu = 0, tol = 1e-5, \
            MaxIter = 30, InitGuess = guess)
    mo = casci.scfsolver.get_mo()
    mo_energy = casci.scfsolver.get_mo_energy()
    norb = mo_energy.size / 2
    ncas = casci.ncas
    log.info("ncore = %d ncas = %d", norb-ncas, ncas)
    log.info("core orbital energy cut-off = %20.12f", \
            max(mo_energy[norb-ncas-1], mo_energy[norb+ncas]) \
            if ncas < norb else float("Inf"))
    log.info("active orbital energy cut-off = %20.12f", \
            max(mo_energy[norb-ncas], mo_energy[norb+ncas-1]))

    core = np.empty((2, norb*2, norb - ncas))
    core[0] = mo[:, :norb - ncas]
    core[1, :norb] = mo[norb:, norb + ncas:]
    core[1, norb:] = mo[:norb, norb + ncas:]
    cas_temp = mo[:, norb - ncas: norb + ncas]
    cas_energy = mo_energy[norb - ncas: norb + ncas]
    cas = [{"o": [], "v": [], "p": []}, {"o": [], "v": [], "p": []}]
    cas_v = map(lambda i: la.norm(cas_temp[:norb,i]), range(ncas*2))
    order = np.argsort(cas_v)
    for idx in order[ncas:]: # alpha
        if cas_energy[idx] < -1e-4:
            cas[0]['o'].append(cas_temp[:, idx])
        elif cas_energy[idx] > 1e-4:
            cas[0]['v'].append(cas_temp[:, idx])
        else:
            cas[0]['p'].append(cas_temp[:, idx])
    for idx in order[:ncas]:
        if cas_energy[idx] < -1e-4:
            cas[1]['v'].append(cas_temp[range(norb, norb*2) + \
                    range(norb), idx])
        elif cas_energy[idx] > 1e-4:
            cas[1]['o'].append(cas_temp[range(norb, norb*2) + \
                    range(norb), idx])
        else:
            cas[1]['p'].append(cas_temp[range(norb, norb*2) + \
                    range(norb), idx])
    casinfo = map(lambda i: (len(cas[i]['o']), len(cas[i]['p']), \
            len(cas[i]['v'])), range(2))
    cas = np.asarray(map(lambda i: np.asarray(cas[i]['o'] + \
            cas[i]['p'] + cas[i]['v']).T, range(2)))

    for s in range(2):
        log.info("In CAS (spin %d):\n"
                "Occupied (e<mu): %d\n""Virtual  (e>mu): %d\n"
                "Partial Occupied: %d\n", s, casinfo[s][0], \
                casinfo[s][2], casinfo[s][1])
    return core, np.asarray(cas), casinfo

def buildCASHamiltonian(Ham, core, cas):
    norb = Ham.norb
    cVA, cVB, cUA, cUB = core[0, :norb], core[1, :norb], \
            core[1, norb:], core[0, norb:]
    cRhoA = np.dot(cVA, cVA.T)
    cRhoB = np.dot(cVB, cVB.T)
    cKappaBA = np.dot(cUB, cVA.T)
    # zero-energy
    _H0 = Ham.H0
    # core-core one-body
    _H0 += np.sum(cRhoA * Ham.H1["cd"][0] + cRhoB * Ham.H1["cd"][1] + \
            2 * cKappaBA.T * Ham.H1["cc"][0])
    # core-fock
    assert(Ham.H2["cccd"] is None or la.norm(Ham.H2["cccd"]) == 0)
    assert(Ham.H2["cccc"] is None or la.norm(Ham.H2["cccc"]) == 0)
    _eriA, _eriB, _eriAB = Ham.H2["ccdd"]

    vj00 = np.tensordot(cRhoA, _eriA, ((0,1), (0,1)))
    vj11 = np.tensordot(cRhoB, _eriB, ((0,1), (0,1)))
    vj10 = np.tensordot(cRhoA, _eriAB, ((0,1), (0,1)))
    vj01 = np.tensordot(_eriAB, cRhoB, ((2,3), (0,1)))
    vk00 = np.tensordot(cRhoA, _eriA, ((0,1), (0,3)))
    vk11 = np.tensordot(cRhoB, _eriB, ((0,1), (0,3)))
    vl10 = np.tensordot(cKappaBA, _eriAB, ((1,0), (0,2))) # wrt kappa_ba.T
    v = np.asarray([vj00+vj01-vk00, vj11+vj10-vk11, vl10])
    # core-core two-body
    _H0 += 0.5 * np.sum(cRhoA * v[0] + cRhoB * v[1] + 2 * cKappaBA.T * v[2])
    VA, VB, UA, UB = cas[0,:norb], cas[1,:norb], cas[1,norb:], cas[0,norb:]
    H0, CD, CC, CCDD, CCCD, CCCC = transform(VA, VB, UA, UB, _H0, \
            Ham.H1["cd"][0] + v[0], Ham.H1["cd"][1] + v[1], Ham.H1["cc"][0] + \
            v[2], Ham.H2["ccdd"][0], Ham.H2["ccdd"][1], Ham.H2["ccdd"][2])
    return integral.Integral(cas.shape[2], False, True, H0, {"cd": CD, "cc": CC}, \
            {"ccdd": CCDD, "cccd": CCCD, "cccc": CCCC})

def split_localize(orbs, info, Ham, basis = None):
    spin = 2
    norbs = Ham.H1["cd"].shape[1]
    localorbs = np.empty_like(orbs)
    rotmat = np.zeros_like(Ham.H1["cd"])
    for s in range(spin):
        occ, part, virt = info[s]
        if occ > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    :occ, :occ, :occ, :occ])
            log.info("Localization: Spin %d, occupied", s)
            localizer.optimize()
            occ_coefs = localizer.coefs.T
            localorbs[s, :, :occ] = np.dot(orbs[s,:,:occ], occ_coefs)
            rotmat[s, :occ, :occ] = occ_coefs
        if virt > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    -virt:, -virt:, -virt:, -virt:])
            log.info("Localization: Spin %d, virtual", s)
            localizer.optimize()
            virt_coefs = localizer.coefs.T
            localorbs[s, :, -virt:] = np.dot(orbs[s,:,-virt:], virt_coefs)
            rotmat[s, -virt:, -virt:] = virt_coefs
        if part > 0:
            localizer = Localizer(Han.H2["ccdd"][s, occ:norbs-virt, \
                occ:norbs-virt, occ:norbs-virt, occ:nobrs-virt])
            log.info("Localization: Spin %d, partially occupied:", s)
            localizer.optimize()
            part_coefs = localizer.ceofs.T
            localorbs[s, :, occ:norbs-virt] = \
                    np.dot(orbs[s,:,occ:norbs-virt], part_coefs)
            rotmat[s, occ:norbs-virt, occ:norbs-virt] = part_coefs
    if basis is not None:
        # match alpha, beta basis
        # localorbs contain v and u parts with respect to embedding quasiparticles
        localbasis = basisToSpin(np.tensordot(basisToCanonical(basis), \
                basisToCanonical(localorbs), (2, 0)))
        ovlp = np.tensordot(np.abs(localbasis[0]), np.abs(localbasis[1]), ((0,1), (0,1)))
        ovlp_sq = ovlp ** 2
        cost_matrix = make_cost_matrix(ovlp_sq, lambda cost: 1. - cost)
        m = Munkres()
        indices = m.compute(cost_matrix)
        indices = sorted(indices, key = lambda idx: idx[0])
        vals = map(lambda idx: ovlp_sq[idx], indices)
        log.debug(1, "Quasiparticle pairs and their overlap:")
        for i in range(norbs):
            log.debug(1, "(%2d, %2d) -> %12.6f", indices[i][0], indices[i][1], vals[i])
        log.info("Match localized quasiparticles: max %5.2f min %5.2f ave %5.2f", \
                np.max(vals), np.min(vals), np.average(vals))

        # update localorbs and rotmat
        orderb = map(lambda idx: idx[1], indices)
        localorbs[1] = localorbs[1][:, orderb]
        rotmat[1] = rotmat[1][:, orderb]

        localbasis[1] = localbasis[1][:,:,orderb]
        # make spin up and down basis have the same sign, i.e.
        # inner product larger than 1
        for i in range(norbs):
            if np.sum(localbasis[0,:,:,i] * localbasis[1,:,:,i]) < 0:
                localorbs[1,:,i] *= -1.
                rotmat[1,:,i] *= -1

    H0, CD, CC, CCDD, CCCD, CCCC = transform_local(rotmat[0], rotmat[1], Ham.H0, \
            Ham.H1["cd"][0], Ham.H1["cd"][1], Ham.H1["cc"][0], Ham.H2["ccdd"][0], \
            Ham.H2["ccdd"][1], Ham.H2["ccdd"][2], Ham.H2["cccd"][0], \
            Ham.H2["cccd"][1], Ham.H2["cccc"][0])
    HamLocal = integral.Integral(norbs, False, True, H0, {"cd": CD, "cc": CC}, \
            {"ccdd": CCDD, "cccd": CCCD, "cccc": CCCC})
    return HamLocal, localorbs, rotmat

def reorder(order, Ham, orbs, rot = None):
    # order 4 1 3 2 means 4 to 1, 1 to 2, 3 to 3, 2 to 4
    # reorder in place
    orbs = orbs[:, :, order]
    Ham.H1["cd"] = Ham.H1["cd"][:, order, :]
    Ham.H1["cd"] = Ham.H1["cd"][:, :, order]
    Ham.H1["cc"] = Ham.H1["cc"][:, order, :]
    Ham.H1["cc"] = Ham.H1["cc"][:, :, order]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, order, :, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, order, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, order, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, :, order]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, order, :, :, :]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, :, order, :, :]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, :, :, order, :]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, :, :, :, order]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, order, :, :, :]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, :, order, :, :]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, :, :, order, :]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, :, :, :, order]
    if rot is not None:
        rot = rot[:, :, order]
        return Ham, orbs, rot
    else:
        return Ham, orbs

class BCSDmrgCI(object):
    def __init__(self, ncas, splitloc = False, cisolver = None, \
            mom_reorder = True, tmpDir = "/tmp"):
        self.ncas = ncas
        self.splitloc = splitloc
        log.eassert(cisolver is not None, "No default ci solver is available" \
                " with CASCI, you have to use Block")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()

        # reorder scheme for restart block calculations
        if mom_reorder:
            if block.Block.reorder:
                log.warning("Using maximal overlap method (MOM) to reorder localized "\
                        "orbitals, turning off Block reorder option")
                block.Block.reoder = False

        self.mom_reorder = mom_reorder
        self.localized_cas = None
        self.tmpDir = tmpDir

    def run(self, Ham, ci_args = {}, guess = None, basis = None, similar = False):
        # ci_args is a list or dict for ci solver, or None

        # FIXME think about choosing number of electron/hole freely
        core, cas, casinfo = get_qps(self, Ham, guess)
        coreGRho = np.dot(core[0], core[0].T)
        casHam = buildCASHamiltonian(Ham, core, cas)

        if self.splitloc:
            casHam, cas, _ = \
                    split_localize(cas, casinfo, casHam, basis = basis)

        if self.mom_reorder:
            log.eassert(basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            if self.localized_cas is None:
                order = gaopt(casHam, tmp = self.tmpDir)
            else:
                # define cas_basis
                cas_basis = basisToSpin(np.tensordot(basisToCanonical(basis), \
                        basisToCanonical(cas), (2,0)))
                # cas_basis and self.localized_cas are both in
                # atomic representation now
                order, q = momopt(self.localized_cas, cas_basis)
                # if the quality of mom is too bad, we reorder the orbitals
                # using genetic algorithm
                # FIXME seems larger q is a better choice
                if q < 0.7:
                    order = gaopt(casHam, tmp = self.tmpDir)

            log.info("Orbital order: %s", order)
            # reorder casHam and cas
            casHam, cas = reorder(order, casHam, cas)
            # store cas in atomic basis
            self.localized_cas = basisToSpin(np.tensordot(basisToCanonical(basis), \
                        basisToCanonical(cas), (2,0)))

        casGRho, E = self.cisolver.run(casHam, **ci_args)
        cas1 = basisToCanonical(cas)
        GRho = mdot(cas1, casGRho, cas1.T) + coreGRho
        return GRho, E

    def cleanup(self):
        self.cisolver.cleanup()