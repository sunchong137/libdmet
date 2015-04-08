import libdmet.utils.logger as log
import h5py
import numpy as np

class Integral(object):
    def __init__(self, norb, restricted, bogoliubov, H0, H1, H2):
        self.norb = norb
        self.restricted = restricted
        self.bogoliubov = bogoliubov
        self.H0 = H0
        self.H1 = H1
        self.H2 = H2

def dump(filename, integral):
    log.fassert(not (integral.bogoliubov and integral.restricted), \
        "Bogoliubov Hamiltonian with spin restriction is not implemented")

    pass

def read(filename, restricted, bogoliubov, norb):
    log.fassert(not (bogoliubov and restricted), "Bogoliubov Hamiltonian with spin restriction is not implemented")
    with open(filename, "r") as f:
        head = f.readline()
        log.eassert((bogoliubov and "&BCS" in head) or (not bogoliubov and "&FCI" in head), \
            "particle number conservation is not consistent")
        log.eassert(norb == int(head.split(',')[0].split('=')[1]), "orbital number is not consistent")
        IUHF = False
        line = f.readline()
        while not "&END" in line:
          IUHF = IUHF or "IUHF" in line
          line = f.readline()
        log.eassert(restricted != IUHF, "spin restriction is not consistent")
        if restricted and not bogoliubov:
            H0 = 0
            H1 = {"cd": np.zeros((norb, norb))}
            H2 = {"ccdd": np.zeros((norb, norb, norb, norb))}
            lines = f.readlines()
            for line in lines:
                tokens = line.split()
                val = float(tokens[0])
                i,j,k,l = [int(x) - 1 for x in tokens[1:]]
                if k >= 0 and l >= 0:
                    H2["ccdd"][i,j,k,l] = H2["ccdd"][j,i,k,l] = H2["ccdd"][i,j,l,k] = \
                        H2["ccdd"][j,i,l,k] = H2["ccdd"][k,l,i,j] = H2["ccdd"][k,l,j,i] = \
                        H2["ccdd"][l,k,i,j] = H2["ccdd"][l,k,j,i] = val
                elif i >= 0 and j >= 0:
                    H1["cd"][i,j] = H1["cd"][j,i] = val
                else:
                    H0 += val
            return Integral(norb, restricted, bogoliubov, H0, H1, H2)


read("/home/zhengbx/dev/libdmet/block/dmrg_tests/hubbard/FCIDUMP", True, False, 12)


def dump_bin(filename, restricted, bogoliubov, norb, H0, H1, H2):
    log.fassert(not (bogoliubov and restricted), "Bogoliubov Hamiltonian with spin restriction is not implemented")
    


def read_bin(filename, restricted, bogoliubov, norb):
    log.fassert(not (bogoliubov and restricted), "Bogoliubov Hamiltonian with spin restriction is not implemented")
    log.eassert(h5py.is_hdf5(filename), "File %s is not hdf5 file", filename)
    f = h5py.File(filename)
    log.eassert(f["restricted"] == restricted, "spin restriction is not consistent")
    log.eassert(f["bogoliubov"] == bogoliubov, "particle number conservation is not consistent")
    log.eassert(f["norb"] == norb, "orbital number is not consistent")
