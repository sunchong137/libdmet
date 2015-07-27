import numpy as np
import pyscf
from pyscf.mcscf import mc1step_uhf, casci_uhf
import libdmet.utils.logger as log
from libdmet.solver.scf import SCF, incore_transform, pyscflogger

class CASSCF(mc1step_uhf.CASSCF):
    def __init__(self, mf, ncas, nelecas, ncore = None, frozen = []):
        mc1step_uhf.CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)
        casci_uhf.CASCI.get_hcore = lambda *args: mf.h1e
        if log.Level[log.verbose] >= log.Level["RESULT"]:
            self.verbose = 4
        else:
            self.verbose = 2
        if log.Level[log.verbose] <= log.Level["INFO"]:
            pyscflogger.flush.addkey("macro iter")
            pyscflogger.flush.addkey("CASCI")

    def ao2mo(self, mo):
        nmo = mo[0].shape[1]
        ncore = self.ncore
        ncas = self.ncas
        nocc = (ncas + ncore[0], ncas + ncore[1])
        eriaa, eribb, eriab = incore_transform(self._scf._eri, (mo, mo, mo, mo))
        eris = lambda:None
        eris.jkcpp = np.einsum('iipq->ipq', eriaa[:ncore[0],:ncore[0],:,:]) \
                   - np.einsum('ipqi->ipq', eriaa[:ncore[0],:,:,:ncore[0]])
        eris.jkcPP = np.einsum('iipq->ipq', eribb[:ncore[1],:ncore[1],:,:]) \
                   - np.einsum('ipqi->ipq', eribb[:ncore[1],:,:,:ncore[1]])
        eris.jC_pp = np.einsum('pqii->pq', eriab[:,:,:ncore[1],:ncore[1]])
        eris.jc_PP = np.einsum('iipq->pq', eriab[:ncore[0],:ncore[0],:,:])
        eris.aapp = np.copy(eriaa[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
        eris.aaPP = np.copy(eriab[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
        eris.AApp = np.copy(eriab[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].transpose(2,3,0,1))
        eris.AAPP = np.copy(eribb[ncore[1]:nocc[1],ncore[1]:nocc[1],:,:])
        eris.appa = np.copy(eriaa[ncore[0]:nocc[0],:,:,ncore[0]:nocc[0]])
        eris.apPA = np.copy(eriab[ncore[0]:nocc[0],:,:,ncore[1]:nocc[1]])
        eris.APPA = np.copy(eribb[ncore[1]:nocc[1],:,:,ncore[1]:nocc[1]])

        eris.cvCV = np.copy(eriab[:ncore[0],ncore[0]:,:ncore[1],ncore[1]:])
        eris.Icvcv = eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:] * 2\
                   - eriaa[:ncore[0],:ncore[0],ncore[0]:,ncore[0]:].transpose(0,3,1,2) \
                   - eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:].transpose(0,3,2,1)
        eris.ICVCV = eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:] * 2\
                   - eribb[:ncore[1],:ncore[1],ncore[1]:,ncore[1]:].transpose(0,3,1,2) \
                   - eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:].transpose(0,3,2,1)

        eris.Iapcv = eriaa[ncore[0]:nocc[0],:,:ncore[0],ncore[0]:] * 2 \
                   - eriaa[:,ncore[0]:,:ncore[0],ncore[0]:nocc[0]].transpose(3,0,2,1) \
                   - eriaa[:,:ncore[0],ncore[0]:,ncore[0]:nocc[0]].transpose(3,0,1,2)
        eris.IAPCV = eribb[ncore[1]:nocc[1],:,:ncore[1],ncore[1]:] * 2 \
                   - eribb[:,ncore[1]:,:ncore[1],ncore[1]:nocc[1]].transpose(3,0,2,1) \
                   - eribb[:,:ncore[1],ncore[1]:,ncore[1]:nocc[1]].transpose(3,0,1,2)
        eris.apCV = np.copy(eriab[ncore[0]:nocc[0],:,:ncore[1],ncore[1]:])
        eris.APcv = np.copy(eriab[:ncore[0],ncore[0]:,ncore[1]:nocc[1],:].transpose(2,3,0,1))
        eris.vhf_c = (np.einsum('ipq->pq', eris.jkcpp) + eris.jC_pp,
                    np.einsum('ipq->pq', eris.jkcPP) + eris.jc_PP)

        return eris
 

if __name__ == "__main__":
    log.verbose = "INFO"
    Int1e = -np.eye(12, k = 1)
    Int1e[0, 11] = -1
    Int1e += Int1e.T
    Int1e = np.asarray([Int1e, Int1e])
    Int2e = np.zeros((3,12,12,12,12))

    for i in range(12):
        Int2e[0,i,i,i,i] = 1
        Int2e[1,i,i,i,i] = 1
        Int2e[2,i,i,i,i] = 1

    scf = SCF()
    scf.set_system(12, 0, False, False)
    scf.set_integral(12, 0, {"cd": Int1e}, {"ccdd": Int2e})
    ehf, rhoHF = scf.HF(MaxIter = 100, tol = 1e-3, \
        InitGuess = (np.diag([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]), \
        np.diag([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])))
    mc = CASSCF(scf.mf, 8, (4,4))
    emc = mc.mc1step()[0]
    print mc.make_rdm1s()
    print (ehf, emc, emc-ehf)
