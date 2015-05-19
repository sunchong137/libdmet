import numpy as np
import os
from tempfile import mkdtemp
import libdmet.utils.logger as log
from libdmet.system import integral
from libdmet.utils.misc import grep
from copy import deepcopy
import subprocess as sub

class Schedule(object):
    def __init__(self, maxiter = 30, sweeptol = 1e-7):
        self.initialized = False
        self.twodot_to_onedot = None
        self.maxiter = maxiter
        self.sweeptol = sweeptol

    def gen_initial(self, minM, maxM):
        defaultM = [100, 250, 400, 800, 1500, 2500]
        log.debug(1, "Generate default schedule with startM = %d maxM = %d, maxiter = %d", \
            minM, maxM, self.maxiter)

        self.arrayM = [minM] + [M for M in defaultM if M > minM and M < maxM] + [maxM]
        self.arraySweep = range(0, 6 * len(self.arrayM), 6)
        self.arrayTol = [self.sweeptol * 0.1 * 10.**i for i in range(len(self.arrayM))][::-1]
        self.arrayNoise = deepcopy(self.arrayTol)

        self.arrayM.append(maxM)
        self.arraySweep.append(self.arraySweep[-1] + 2)
        self.arrayTol.append(self.arrayTol[-1])
        self.arrayNoise.append(0)

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = self.arraySweep[-1] + 6

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        if self.twodot_to_onedot + 4 > self.maxiter:
            log.warning("only %d onedot iterations\nmodify maxiter to %d", \
                self.maxiter - self.twodot_to_onedot, self.twodot_to_onedot + 4)
            self.maxiter = self.twodot_to_onedot + 4
        self.initialized = True

    def gen_restart(self, M):
        log.debug(1, "Generate default schedule with restart calculation M = %d, maxiter = %d", M, self.maxiter)
        self.arrayM = [M, M]
        self.arraySweep = [0, 2]
        self.arrayTol = [self.sweeptol * 0.1] * 2
        self.arrayNoise = [self.sweeptol * 0.1, 0]

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = self.arraySweep[-1] + 6

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        if self.twodot_to_onedot + 4 > self.maxiter:
            log.warning("only %d onedot iterations\nmodify maxiter to %d", \
                self.maxiter - self.twodot_to_onedot, self.twodot_to_onedot + 4)
            self.maxiter = self.twodot_to_onedot + 4
        self.initialized = True

    def gen_extrapolate(self, M):
        log.debug(1, "Generate default schedule for truncation error extrapolation M = %d", M)
        self.arrayM = [M]
        self.arraySweep = [0]
        self.arrayTol = [self.sweeptol * 0.1]
        self.arrayNoise = [0]

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = 0
        self.maxiter = 2

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        self.initialized = True

    def gen_custom(self, arrayM, arraySweep, arrayTol, arrayNoise, twodot_to_onedot = None):
        log.debug(1, "Generate custom schedule")
        nstep = len(arrayM)
        log.eassert(len(arraySweep) == nstep and len(arrayTol) == nstep and len(arrayNoise) == nstep, \
            "The lengths of input arrays are not consistent.")

        self.arrayM, self.arraySweep, self.arrayTol, self.arrayNoise = \
            arrayM, arraySweep, arrayTol, arrayNoise

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        if twodot_to_onedot is None:
            self.twodot_to_onedot = self.arraySweep[-1] + 2
        else:
            self.twodot_to_onedot = twodot_to_onedot

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        if self.arraySweep[-1]+2 > self.maxiter:
            log.warning("maxiter smaller than scheduled number of sweeps\nmodify maxiter to %d", \
                self.arraySweep[-1]+2)
            self.maxiter = self.arraySweep[-1]+2
        self.initialized = True

    def get_schedule(self):
        log.eassert(self.initialized, "DMRG schedule has not been generated.")
        text = ["", "schedule"]
        nstep = len(self.arrayM)
        text += map(lambda n: "%d %d %.0e %.0e" % \
            (self.arraySweep[n], self.arrayM[n], self.arrayTol[n], self.arrayNoise[n]), range(nstep))
        text.append("end")
        text.append("")
        text.append("maxiter %d" % self.maxiter)
        if self.twodot_to_onedot <= 0:
            text.append("onedot")
        elif self.twodot_to_onedot >= self.maxiter:
            text.append("twodot")
        else:
            text.append("twodot_to_onedot %d" % self.twodot_to_onedot)
        text.append("sweep_tol %.0e" % self.sweeptol)
        text.append("")
        text = "\n".join(text)
        log.debug(2, "Generated schedule in configuration file")
        log.debug(1, text)

        return text

def readpdm(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    nsites = int(lines[0])
    pdm = np.zeros((nsites, nsites))

    for line in lines[1:]:
        tokens = line.split(" ")
        pdm[int(tokens[0]), int(tokens[1])] = float(tokens[2])

    return pdm

class Block(object):

    execPath = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../block"))
    nproc = 1
    nnode = 1
    intFormat = "FCIDUMP"
    basicFiles = ["dmrg.conf.*", "FCIDUMP"]
    restartFiles = ["RestartReorder.dat", "Rotation*", "StateInfo*", "statefile*", "wave*"]
    tempFiles = ["Spin*", "Overlap*", "dmrg.e", "spatial*", "onepdm.*", "twopdm.*", "pairmat.*", \
        "dmrg.out.*", "RI*"]
    env_slurm = "SLURM_JOBID" in os.environ
    mpipernode = ["mpirun", "-npernode", "1"]

    @classmethod
    def set_nproc(cls, nproc, nnode = 1):
        cls.nproc = nproc
        cls.nnode = nnode
        log.info("Block interface  running with %d nodes, %d processors per node", \
            cls.nnode, cls.nproc)
        log.info("Block running on nodes:\n%s", sub.check_output(Block.mpipernode + ["hostname"]).replace("\n", "\t"))

    def __init__(self):
        self.sys_initialized = False
        self.schedule_initialized = False
        self.integral_initialized = False
        self.optimized = False
        self.count = 0

        self.warmup_method = "local_2site"
        self.outputlevel = 0
        self.restart = False

        log.debug(0, "Using Block version %s", Block.execPath)

    def createTmp(self, tmp = "/tmp", shared = None):
        sub.check_call(["mkdir", "-p", tmp])
        self.tmpDir = mkdtemp(prefix = "BLOCK", dir = tmp)
        log.info("Block working dir %s", self.tmpDir)
        if Block.nnode > 1:
            log.eassert(shared is not None, "when running on multiple nodes, a shared tmporary folder is required")
            sub.check_call(["mkdir", "-p", shared])
            self.tmpShared = mkdtemp(prefix = "BLOCK", dir = shared)
            sub.check_call(Block.mpipernode + ["mkdir", "-p", self.tmpDir])
            log.info("Block shared dir %s", self.tmpShared)

    def set_system(self, nelec, spin, spinAdapted, bogoliubov, spinRestricted):
        self.nelec = nelec
        self.spin = spin
        log.fassert(not (spinAdapted and bogoliubov), \
            "Bogoliubov calculation with spin adaption is not implemented")
        self.spinAdapted = spinAdapted
        self.spinRestricted = spinRestricted
        self.bogoliubov = bogoliubov
        self.sys_initialized = True

    def set_integral(self, *args):
        log.eassert(self.sys_initialized, "set_integral() should be used after initializing set_system()")
        if len(args) == 1:
            # a single integral object
            self.integral = args[0]
        elif len(args) == 4:
            # norb, H0, H1, H2
            self.integral = integral.Integral(args[0], self.spinRestricted, self.bogoliubov, *args[1:])
        else:
            log.error("input either an integral object, or (norb, H0, H1, H2)")
        self.integral_initialized = True

    def set_schedule(self, schedule):
        self.schedule = schedule
        self.schedule_initialized = True

    def write_conf(self, f):
        f.write("nelec %d\n" % self.nelec)
        f.write("spin %d\n" % self.spin)
        f.write("hf_occ integral\n")
        f.write(self.schedule.get_schedule())
        f.write("orbitals %s\n" % os.path.join(self.tmpDir, "FCIDUMP"))
        f.write("warmup %s\n" % self.warmup_method)
        f.write("nroots 1\n")
        f.write("outputlevel %d\n" % self.outputlevel)
        f.write("prefix %s\n" % self.tmpDir)
        if self.restart or self.optimized:
            f.write("fullrestart\n")
        if self.bogoliubov:
            f.write("bogoliubov\n")
        if not self.spinAdapted:
            f.write("nonspinadapted\n")

    def copy_restartfile(self, src, cleanup = True):
        files = Block.restartFiles
        if Block.nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared
        for f in files:
            sub.check_call(Block.mpipernode + ["cp", os.path.join(src, f), startPath])
        if Cleanup:
            sub.check_call(["rm", "-rf", src])
        self.restart = True

    def save_restartfile(self, des, cleanup = True):
        # the des has to be created before calling this method
        # recommanded using mkdtemp(prefix = "BLOCK_RESTART", dir = path_to_storage)
        files = Block.restartFiles
        for f in files:
            sub.check_call(["cp", os.path.join(self.tmpDir, f), des])
        if cleanup:
            self.cleanup()

    def broadcast(self):
        files = Block.basicFiles
        if self.restart and not self.optimized:
            files += Block.restartFiles

        for f in files:
            sub.check_call(Block.mpipernode + ["cp", os.path.join(self.tmpShared, f), self.tmpDir])

    def callBlock(self):
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("BLOCK call No. %d", self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering = 1) as f:
            if Block.env_slurm:
                sub.check_call(" ".join(["srun", \
                    os.path.join(Block.execPath, "block.spin_adapted"), os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]), \
                    stdout = f, shell = True)
            else:
                sub.check_call(["mpirun", "-np", "%d" % (Block.nproc * Block.nnode), \
                    os.path.join(Block.execPath, "block.spin_adapted"), os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)], \
                    stdout = f)
        log.result("BLOCK sweep summary")
        log.result(grep("Sweep Energy", outputfile))
        self.count += 1

    def callOH(self):
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("OH call No. %d", self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering = 1) as f:
            if Block.env_slurm:            
                sub.check_call(" ".join(["srun", "-n", "1", \
                    os.path.join(Block.execPath, "OH"), os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]), \
                    stdout = f, shell = True)
            else:
                sub.check_call(["mpirun", "-np", "1", \
                    os.path.join(Block.execPath, "OH"), os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)], \
                    stdout = f)
        self.count += 1

    def extractE(self, text):
        results = []
        lines = map(lambda s: s.split(), text.split('\n')[-2:])
        keys = ["Weight"]
        for key in keys:
            place = map(lambda tokens: tokens.index(key), lines)
            results.append(np.average(map(lambda (tokens, idx): float(tokens[idx+2]), zip(lines, place))))

        lines = map(lambda s: s.split(), text.split('\n')[-1:])
        keys = ["Energy"]
        for key in keys:
            place = map(lambda tokens: tokens.index(key), lines)
            results.append(np.average(map(lambda (tokens, idx): float(tokens[idx+2]), zip(lines, place))))

        return tuple(results)

    def onepdm(self):
        if self.spinRestricted:
            rho = readpdm(os.path.join(self.tmpDir, "spatial_onepdm.0.0.txt")) / 2
            rho = rho.reshape((1, self.integral.norb, self.integral.norb))
        else:
            rho0 = readpdm(os.path.join(self.tmpDir, "onepdm.0.0.txt"))
            rho = np.empty((2, self.integral.norb, self.integral.norb))
            rho[0] = rho0[::2, ::2]
            rho[1] = rho0[1::2, 1::2]
        if self.bogoliubov:
            kappa = readpdm(os.path.join(self.tmpDir, "spatial_pairmat.0.0.txt"))
            if self.spinRestricted:
                kappa = (kappa + kappa.T) / 2
            return (rho, kappa)
        else:
            return rho

    def just_run(self, onepdm = True, dry_run = False):
        log.debug(0, "Run BLOCK")

        if Block.nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared

        configFile = os.path.join(startPath, "dmrg.conf.%03d" % self.count)
        with open(configFile, "w") as f:
            self.write_conf(f)
            if onepdm:
                f.write("onepdm\n")

        intFile = os.path.join(startPath, "FCIDUMP")
        integral.dump(intFile, self.integral, Block.intFormat)
        if Block.nnode > 1:
            self.broadcast()

        if not dry_run:
            self.callBlock()
            outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % (self.count-1))
            truncation, energy = self.extractE(grep("Sweep Energy", outputfile))

            if onepdm:
                return truncation, energy, self.onepdm()
            else:
                return truncation, energy, None
        else:
            return None, None, None

    def optimize(self, onepdm = True):
        log.eassert(self.sys_initialized and self.integral_initialized and self.schedule_initialized, \
            "components for optimization are not ready\nsys_init = %s\nint_init = %s\nschedule_init = %s", \
            self.sys_initialized, self.integral_initialized, self.schedule_initialized)

        if self.optimized:
            return self.restart_optimize(onepdm)

        log.info("Run BLOCK to optimize wavefunction")
        results = self.just_run(onepdm, dry_run = False)
        self.optimized = True
        return results

    def restart_optimize(self, onepdm = True, M = None):
        log.eassert(self.optimized, "No wavefunction available")

        if M is None:
            M = self.schedule.arrayM[-1]
        self.schedule.gen_restart(M = M)

        log.info("Run BLOCK to optimize wavefunction (restart)")
        return self.just_run(onepdm, dry_run = False)


    def extrapolate(self, Ms, onepdm = True):
        log.eassert(self.sys_initialized and self.integral_initialized, \
            "components for optimization are not ready\nsys_init = %s\nint_init = %s", \
            self.sys_initialized, self.integral_initialized)
        results = []
        if not self.optimized or self.restart:
            self.schedule = Schedule()
            self.schedule.gen_initial(Ms[0]/2, Ms[0])
            self.schedule_initialized = True
            results.append(self.optimize(onepdm = onepdm))
        else:
            results.append(self.restart_optimize(self, onepdm = onepdm, M = Ms[0]))
        for M in Ms[1:]:
            self.schedule.gen_extrapolate(M)
            results.append(self.just_run(onepdm = onepdm, dry_run = False))

    def evaluate(self, H0, H1, H2, op = "unknown operator"):
        log.eassert(self.optimized, "No wavefunction available")
        self.set_integral(self.integral.norb, H0, H1, H2)

        log.info("Run OH to evaluate expectation value of %s", op)

        if Block.nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared

        # just copy configure file
        sub.check_call(["cp", os.path.join(startPath, "dmrg.conf.%03d" % (self.count-1)), \
            os.path.join(startPath, "dmrg.conf.%03d" % (self.count))])
        with open(os.path.join(startPath, "dmrg.conf.%03d" % (self.count)), "a") as f:
            f.write("fullrestart\n")

        intFile = os.path.join(startPath, "FCIDUMP")
        integral.dump(intFile, self.integral, Block.intFormat)
        if Block.nnode > 1:
            self.broadcast()
        self.callOH()

        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % (self.count-1))
        h = float(grep("helement", outputfile).split()[-1])
        log.debug(1, "operator evaluated: %20.12f" % h)
        return h

    def cleanup(self, keep_restart = False):
        if keep_restart:
            for filename in Block.tempFiles:
                sub.check_call(Block.mpipernode + ["rm", "-rf", os.path.join(self.tmpDir, filename)])
        else:
            sub.check_call(Block.mpipernode + ["rm", "-rf", self.tmpDir])
            if Block.nnode > 1:
                sub.check_call(["rm", "-rf", self.tmpShared])
            self.optimized = False

