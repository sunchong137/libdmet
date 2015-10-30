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

def get_BCS_mo(scfsolver, Ham, guess):
    scfsolver.set_system(None, 0, True, False)
    scfsolver.set_integral(Ham)

    E_HFB, GRho_HFB = scfsolver.HFB(Mu = 0, tol = 1e-7, \
            MaxIter = 50, InitGuess = guess)

    return scfsolver.get_mo(), scfsolver.get_mo_energy()

def get_qps(ncas, algo = "nelec", **kwargs):
    if algo == "nelec":
        log.eassert("nelec" in kwargs, \
            "number of electrons has to be specified")
        return lambda mo, mo_e: get_qps_nelec(ncas, kwargs["nelec"], mo, mo_e)
    elif algo == "energy":
        return lambda mo, mo_e: get_qps_energy(ncas, mo, mo_e)
    elif algo == "local":
        pass

def get_qps_nelec(ncas, nelec, mo, mo_energy):
    norb = mo_energy.size / 2

    log.info("ncore = %d ncas = %d", norb-ncas, ncas)
    # number of electrons per spin
    nelecas = nelec / 2
    mo_v = map(lambda i: np.sum(mo[:norb,i]**2), range(norb*2))
    mo_v_ord = np.argsort(mo_v)
    # if v > 0.5, classify as alpha modes, otherwise beta modes
    mo_a, mo_b = mo_v_ord[norb:], mo_v_ord[:norb]

    # ordered so that those close to the fermi surface come first 
    AO = sorted(filter(lambda i: i < norb, mo_a))[::-1]
    AV = sorted(filter(lambda i: i >= norb, mo_a))
    BO = sorted(filter(lambda i: i >= norb, mo_b))
    BV = sorted(filter(lambda i: i < norb, mo_b))[::-1]

    # divide into cas and core
    casA_idx = AO[:nelecas][::-1] + AV[:ncas - nelecas]
    casB_idx = BO[:nelecas][::-1] + BV[:ncas - nelecas]
    coreA_idx = AO[nelecas:] + BV[ncas-nelecas:]
    coreB_idx = BO[nelecas:] + AV[ncas-nelecas:]

    # now seriously classify casA and casB into occ, partial and virt
    casA_occ, casA_part, casA_virt = [], [], []
    # according to energy and particle character
    for idx in casA_idx:
        if mo_v[idx] > 0.7 and mo_energy[idx] < -1e-4:
            casA_occ.append(idx)
        elif mo_v[idx] > 0.7 and mo_energy[idx] > 1e-4:
            casA_virt.append(idx)
        else:
            casA_part.append(idx)

    casB_occ, casB_part, casB_virt = [], [], []
    for idx in casB_idx:
        if mo_v[idx] < 0.3 and mo_energy[idx] > 1e-4:
            casB_occ.append(idx)
        elif mo_v[idx] < 0.3 and mo_energy[idx] < -1e-4:
            casB_virt.append(idx)
        else:
            casB_part.append(idx)

    # extract cas from mo
    casA = mo[:, casA_occ+casA_part+casA_virt]
    casB = np.vstack((
        mo[norb:, casB_occ+casB_part+casB_virt],
        mo[:norb, casB_occ+casB_part+casB_virt]))
    casinfo = (
        (len(casA_occ), len(casA_part), len(casA_virt)),
        (len(casB_occ), len(casB_part), len(casB_virt))
    )
    # extract core
    coreA = mo[:, coreA_idx]
    coreB = np.vstack((
        mo[norb:, coreB_idx],
        mo[:norb, coreB_idx]))
    return np.asarray([coreA, coreB]), np.asarray([casA, casB]), \
            casinfo

def get_qps_energy(ncas, mo, mo_energy):
    norb = mo_energy.size / 2
    log.info("ncore = %d ncas = %d", norb-ncas, ncas)
    log.info("core orbital energy cut-off = %20.12f", \
            max(mo_energy[norb-ncas-1], mo_energy[norb+ncas]) \
            if ncas < norb else float("Inf"))
    log.info("active orbital energy cut-off = %20.12f", \
            max(mo_energy[norb-ncas], mo_energy[norb+ncas-1]))

    # generate core
    core = np.empty((2, norb*2, norb - ncas))
    # alpha : first norb-ncas modes
    core[0] = mo[:, :norb - ncas]
    # beta : last norb-ncas modes
    core[1, :norb] = mo[norb:, norb + ncas:]
    core[1, norb:] = mo[:norb, norb + ncas:]
    # cas 2(norb - ncas) modes
    cas_temp = mo[:, norb - ncas: norb + ncas]
    cas_energy = mo_energy[norb - ncas: norb + ncas]
    cas = [{"o": [], "v": [], "p": []}, {"o": [], "v": [], "p": []}]
    cas_v = map(lambda i: la.norm(cas_temp[:norb,i])**2, range(ncas*2))
    order = np.argsort(cas_v)
    for idx in order[ncas:]: # alpha
        if cas_energy[idx] < -1e-4:
            cas[0]['o'].append(cas_temp[:, idx])
        elif cas_energy[idx] > 1e-4:
            cas[0]['v'].append(cas_temp[:, idx])
        else:
            cas[0]['p'].append(cas_temp[:, idx])
    for idx in order[:ncas]: # beta
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
    # core-core energy
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
            {"ccdd": CCDD, "cccd": CCCD, "cccc": CCCC}), _H0

def rotateHam(rotmat, Ham):
    H0, CD, CC, CCDD, CCCD, CCCC = transform_local(rotmat[0], rotmat[1], Ham.H0, \
            Ham.H1["cd"][0], Ham.H1["cd"][1], Ham.H1["cc"][0], Ham.H2["ccdd"][0], \
            Ham.H2["ccdd"][1], Ham.H2["ccdd"][2], Ham.H2["cccd"][0], \
            Ham.H2["cccd"][1], Ham.H2["cccc"][0])
    return integral.Integral(Ham.norb, False, True, H0, {"cd": CD, "cc": CC}, \
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
            localizer = Localizer(Ham.H2["ccdd"][s, occ:norbs-virt, \
                occ:norbs-virt, occ:norbs-virt, occ:norbs-virt])
            log.info("Localization: Spin %d, partially occupied:", s)
            localizer.optimize()
            part_coefs = localizer.coefs.T
            localorbs[s, :, occ:norbs-virt] = \
                    np.dot(orbs[s,:,occ:norbs-virt], part_coefs)
            rotmat[s, occ:norbs-virt, occ:norbs-virt] = part_coefs
    if basis is not None:
        # match alpha, beta basis
        # localorbs contain v and u parts with respect to embedding quasiparticles
        localbasis = basisToSpin(np.tensordot(basisToCanonical(basis), \
                basisToCanonical(localorbs), (2, 0)))
        nscsites = basis.shape[2] / 2
        localbasis0 = np.sqrt(localbasis[0,:,:nscsites]**2+localbasis[0,:,nscsites:]**2)
        localbasis1 = np.sqrt(localbasis[1,:,:nscsites]**2+localbasis[1,:,nscsites:]**2)
        ovlp = np.tensordot(localbasis0, localbasis1, ((0,1), (0,1))) 
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

    HamLocal = rotateHam(rotmat, Ham)
    return HamLocal, localorbs, rotmat

def momopt(old_basis, new_basis):
    norb = old_basis.shape[2] / 2
    old_basis1 = np.sqrt(old_basis[:,:,:norb] ** 2 + old_basis[:,:,norb:] ** 2)
    new_basis1 = np.sqrt(new_basis[:,:,:norb] ** 2 + new_basis[:,:,norb:] ** 2)
    # use Hungarian algorithm to match the basis
    ovlp = 0.5 * np.tensordot(np.abs(old_basis1), np.abs(new_basis1), ((0,1,2), (0,1,2)))
    ovlp_sq = ovlp ** 2
    cost_matrix = make_cost_matrix(ovlp_sq, lambda cost: 1. - cost)

    m = Munkres()
    indices = m.compute(cost_matrix)
    indices = sorted(indices, key = lambda idx: idx[0])
    vals = map(lambda idx: ovlp_sq[idx], indices)
    log.info("MOM reorder quality: max %5.2f min %5.2f ave %5.2f", \
            np.max(vals), np.min(vals), np.average(vals))

    reorder = [idx[1] for idx in indices]
    return reorder, np.average(vals)

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
            mom_reorder = True, algo = "nelec", tmpDir = "/tmp", **kwargs):
        # algo can be nelec, energy and local
        self.ncas = ncas
        self.splitloc = splitloc
        log.eassert(cisolver is not None, "No default ci solver is available" \
                " with CASCI, you have to use Block")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()
        self.get_qps = get_qps(ncas, algo, **kwargs)

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

        mo, mo_energy = get_BCS_mo(self.scfsolver, Ham, guess)
        core, cas, casinfo = self.get_qps(mo, mo_energy)
        #core, cas, casinfo = get_qps(self, Ham, guess)
        coreGRho = np.dot(core[0], core[0].T)
        casHam, _ = buildCASHamiltonian(Ham, core, cas)

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
