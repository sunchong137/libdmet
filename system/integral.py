import libdmet.utils.logger as log
import h5py
import numpy as np
import itertools as it
import os

class Integral(object):
    def __init__(self, norb, restricted, bogoliubov, H0, H1, H2):
        self.norb = norb
        self.restricted = restricted
        self.bogoliubov = bogoliubov
        self.H0 = H0
        self.H1 = H1
        self.H2 = H2

    def pairNoSymm(self):
        return list(it.product(range(self.norb), repeat = 2))

    def pairSymm(self):
        return list(it.combinations_with_replacement(range(self.norb)[::-1], 2))[::-1]

    def pairAntiSymm(self):
        return list(it.combinations(range(self.norb)[::-1], 2))[::-1]

def dumpFCIDUMP(filename, integral, thr = 1e-8):
    header = []
    if integral.bogoliubov:
        header.append(" &BCS NORB= %d," % integral.norb)
    else:
        header.append(" &FCI NORB= %d,NELEC= %d,MS2= %d," % (integral.norb, integral.norb, 0))
    header.append("  ORBSYM=" + "1," * integral.norb)
    header.append("  ISYM=1,")
    if not integral.restricted:
        header.append("  IUHF=1,")
    header.append(" &END")

    def writeInt(fout, val, i, j, k = -1, l = -1):
        if abs(val) > thr:
            fout.write("%20.16f%4d%4d%4d%4d\n" % (val, i+1, j+1, k+1, l+1))

    def insert_ccdd(fout, matrix, symm_herm = True, symm_spin = True):
        if symm_herm:
            p = integral.pairSymm()
        else:
            p = integral.pairNoSymm()

        if symm_spin:
            for (i,j), (k,l) in list(it.combinations_with_replacement(p[::-1], 2))[::-1]:
                writeInt(fout, matrix[i,j,k,l], i, j, k, l)
        else:
            for (i,j), (k,l) in it.product(p, repeat = 2):
                writeInt(fout, matrix[i,j,k,l], i, j, k, l)

    def insert_cccd(fout, matrix):
        for (i,j), (k,l) in it.product(integral.pairAntiSymm(), integral.pairNoSymm()):
            writeInt(fout, matrix[i,j,k,l], i, j, k, l)

    def insert_cccc(fout, matrix, symm_spin = True):
        if symm_spin:
            for (i,j), (k,l) in list(it.combinations_with_replacement(integral.pairAntiSymm()[::-1], 2))[::-1]:
                writeInt(fout, matrix[i,j,k,l], i, j, k, l)

        else:
            for (i,j), (k,l) in it.product(integral.pairAntiSymm(), repeat = 2):
                writeInt(fout, matrix[i,j,k,l], i, j, k, l)

    def insert_2dArray(fout, matrix, symm_herm = True):
        if symm_herm:
            for i,j in integral.pairSymm():
                writeInt(fout, matrix[i,j], i, j)
        else:
            for i,j in integral.pairNoSymm():
                writeInt(fout, matrix[i,j], i, j)

    def insert_H0(fout, val = 0):
        fout.write("%20.16f%4d%4d%4d%4d\n" % (val, 0, 0, 0, 0)) # cannot be ignored even if smaller than thr

    if isinstance(filename, str):
        f = open(filename, "w", 1024*1024*128)
    elif isinstance(filename, file):
        f = filename

    f.write("\n".join(header) + "\n")
    if integral.restricted and not integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"])
        insert_2dArray(f, integral.H1["cd"])
        insert_H0(f, integral.H0)
    elif not integral.restricted and not integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccddA"])
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccddB"])
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccddAB"], symm_spin = False)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cdA"])
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cdB"])
        insert_H0(f, 0)
        insert_H0(f, integral.H0)
    elif integral.restricted and integral.bogoliubov:
        insert_ccdd(f, integral.H2["ccdd"])
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccd"])
        insert_H0(f, 0)
        insert_cccc(f, integral.H2["cccc"])
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cd"])
        insert_H0(f,0)
        insert_2dArray(f, integral.H1["cc"])
        insert_H0(f,0)
        insert_H0(f, integral.H0)
    else:
        insert_ccdd(f, integral.H2["ccddA"], symm_herm = False)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccddB"], symm_herm = False)
        insert_H0(f, 0)
        insert_ccdd(f, integral.H2["ccddAB"], symm_herm = False, symm_spin = False)
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccdA"])
        insert_H0(f, 0)
        insert_cccd(f, integral.H2["cccdB"])
        insert_H0(f, 0)
        insert_cccc(f, integral.H2["cccc"], symm_spin = False)
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cdA"])
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cdB"])
        insert_H0(f, 0)
        insert_2dArray(f, integral.H1["cc"], symm_herm = False)
        insert_H0(f, 0)
        insert_H0(f, integral.H0)

    if isinstance(filename, str):
        f.close()

def readFCIDUMP(filename, norb, restricted, bogoliubov):
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
        elif not restricted and not bogoliubov:
            H0 = 0
            H1 = {"cdA": np.zeros((norb, norb)), "cdB": np.zeros((norbs, norbs))}
            H2 = {
                "ccddA": np.zeros((norb, norb, norb, norb)),
                "ccddB": np.zeros((norb, norb, norb, norb)),
                "ccddAB": np.zeros((norb, norb, norb, norb))
            }
            lines = f.readlines()
            section = 0
            for line in lines:
                tokens = line.split()
                val = float(tokens[0])
                i,j,k,l = [int(x) - 1 for x in tokens[1:]]
                if i < 0 and j < 0 and k < 0 and l < 0:
                    section += 1
                    H0 += val
                elif section == 0 or section == 1:
                    key = "ccdd" + ["A", "B"][section]
                    H2[key][i,j,k,l] = H2[key][j,i,k,l] = H2[key][i,j,l,k] = \
                        H2[key][j,i,l,k] = H2[key][k,l,i,j] = H2[key][k,l,j,i] = \
                        H2[key][l,k,i,j] = H2[key][l,k,j,i] = val
                elif section == 2:
                    key = "ccddAB"
                    H2[key][i,j,k,l] = H2[key][j,i,k,l] = H2[key][i,j,l,k] = \
                        H2[key][j,i,l,k] = val # cannot swap ij <-> kl
                elif section == 3 or section == 4:
                    log.eassert(k == -1 and l == -1, "Integral Syntax unrecognized")
                    key = "cd" + ["A", "B"][section-3]
                    H1[key][i,j] = H1[key][j,i] = val
        elif restricted and bogoliubov:
            H0 = 0
            H1 = {"cd": np.zeros((norb, norb)), "cc": np.zeros((norb, norb))}
            H2 = {
                "ccdd": np.zeros((norb, norb, norb, norb)),
                "cccd": np.zeros((norb, norb, norb, norb)),
                "cccc": np.zeros((norb, norb, norb, norb))
            }
            lines = f.readlines()
            section = 0
            for line in lines:
              tokens = line.split()
              val = float(tokens[0])
              i,j,k,l = [int(x) - 1 for x in tokens[1:]]
              if i < 0 and j < 0 and k < 0 and l < 0:
                  section += 1
                  H0 += val
              elif section == 0:
                  H2["ccdd"][i,j,k,l] = H2["ccdd"][j,i,k,l] = H2["ccdd"][i,j,l,k] = \
                    H2["ccdd"][j,i,l,k] = H2["ccdd"][k,l,i,j] = H2["ccdd"][k,l,j,i] = \
                    H2["ccdd"][l,k,i,j] = H2["ccdd"][l,k,j,i] = val
              elif section == 1:
                  H2["cccd"][i,j,k,l] = val
                  H2["cccd"][j,i,k,l] = -val
              elif section == 2:
                  H2["cccc"][i,j,k,l] = H2["cccc"][j,i,l,k] = \
                      H2["cccc"][k,l,i,j] = H2["cccc"][l,k,j,i] = val
                  H2["cccc"][j,i,k,l] = H2["cccc"][i,j,l,k] = \
                      H2["cccc"][k,l,j,i] = H2["cccc"][l,k,i,j] = -val
        else: # bogoliubov, not restricted
            H0 = 0
            H1 = {
                "cdA": np.zeros((norb, norb)),
                "cdB": np.zeros((norb, norb)),
                "cc": np.zeros((norb, norb))
            }
            H2 = {
                "ccddA": np.zeros((norb, norb, norb, norb)),
                "ccddB": np.zeros((norb, norb, norb, norb)),
                "ccddAB": np.zeros((norb, norb, norb, norb)),
                "cccdA": np.zeros((norb, norb, norb, norb)),
                "cccdB": np.zeros((norb, norb, norb, norb)),
                "cccc": np.zeros((norb, norb, norb, norb))
            }
            lines = f.readlines()
            section = 0
            for line in lines:
                tokens = line.split()
                val = float(tokens[0])
                i,j,k,l = [int(x) - 1 for x in tokens[1:]]
                if i < 0 and j < 0 and k < 0 and l < 0:
                    section += 1
                    H0 += val
                if section == 0 or section == 1:
                    key = "ccdd" + ["A", "B"][section]
                    H2[key][i,j,k,l] = H2[key][k,l,i,j] = val
                elif section == 2:
                    H2["ccddAB"][i,j,k,l] = val
                elif section == 3 or section == 4: # cccdA/cccdB
                    key = "cccd" + ["A", "B"][section-3]
                    H2[key][i,j,k,l] = val
                    H2[key][j,i,k,l] = -val
                elif section == 5:
                    key = "cccc"
                    H2[key][i,j,k,l] = H2[j,i,l,k] = val
                    H2[key][j,i,k,l] = H2[i,j,l,k] = -val
                elif section == 6 or section == 7:
                    log.eassert(k == -1 and l == -1, "Integral Syntax unrecognized")
                    key = "cd" + ["A", "B"][section-6]
                    H1[key][i,j] = H1[key][j,i] = val
                elif section == 8:
                    log.eassert(k == -1 and l == -1, "Integral Syntax unrecognized")
                    H1["cc"][i,j] = val
    return Integral(norb, restricted, bogoliubov, H0, H1, H2)

def dumpHDF5(filename, integral):
    log.error("function not implemented: dump_bin")
    raise Exception


def readHDF5(filename, norb, restricted, bogoliubov):
    log.eassert(h5py.is_hdf5(filename), "File %s is not hdf5 file", filename)
    f = h5py.File(filename)
    log.eassert(f["restricted"] == restricted, "spin restriction is not consistent")
    log.eassert(f["bogoliubov"] == bogoliubov, "particle number conservation is not consistent")
    log.eassert(f["norb"] == norb, "orbital number is not consistent")
    log.error("function not implemented: read_bin")
    raise Exception

def dumpMMAP(filename, integral):
    log.eassert(os.path.isdir(filename), "unable to dump memory map files")

    def mmap_write(itype, data):
        temp = np.memmap(os.path.join(filename, itype + ".mmap"), dtype = "float", mode = 'w+', shape = data.shape)
        temp[:] = data[:]
        del temp

    for key, data in integral.H1.items():
        mmap_write(key, data)
    for key, data in integral.H2.items():
        mmap_write(key, data)

    temp = np.memmap(os.path.join(filename, "H0.mmap"), dtype = "float", mode = 'w+', shape = (1,))
    temp[0] = integral.H0
    del temp

def readMMAP(filename, norb, restricted, bogoliubov, copy = False):
    log.eassert(os.path.isdir(filename), "unable to read memory map files")

    def bind(itype, shape):
        if copy:
            return np.array(np.memmap(os.path.join(filename, itype+".mmap"), dtype = "float", mode = 'r', shape = shape))
        else:
            return np.memmap(os.path.join(filename, itype + ".mmap"), dtype = "float", mode = 'r+', shape = shape)


    if restricted and not bogoliubov:
        H1 = {"cd": bind("cd", (norb, norb))}
        H2 = {"ccdd": bind("ccdd", (norb, norb, norb, norb))}
    elif not restricted and not bogoliubov:
        H1 = {"cdA": bind("cdA", (norb, norb)), "cdB": bind("cdB", (norb, norb))}
        H2 = {
            "ccddA": bind("ccddA", (norb, norb, norb, norb)),
            "ccddB": bind("ccddB", (norb, norb, norb, norb)),
            "ccddAB": bind("ccddAB", (norb, norb, norb, norb)),
        }
    elif restricted and bogoliubov:
        H1 = {"cd": bind("cd", (norb, norb)), "cc": bind("cc", (norb, norb))}
        H2 = {
            "ccdd": bind("ccdd", (norb, norb, norb, norb)),
            "cccd": bind("cccd", (norb, norb, norb, norb)),
            "cccc": bind("cccc", (norb, norb, norb, norb)),
        }
    else:
        H1 = {
            "cdA": bind("cdA", (norb, norb)),
            "cdB": bind("cdB", (norb, norb)),
            "cc": bind("cc", (norb, norb)),
        }
        H2 = {
            "ccddA": bind("ccddA", (norb, norb, norb, norb)),
            "ccddB": bind("ccddB", (norb, norb, norb, norb)),
            "ccddAB": bind("ccddAB", (norb, norb, norb, norb)),
            "cccdA": bind("cccdA", (norb, norb, norb, norb)),
            "cccdB": bind("cccdB", (norb, norb, norb, norb)),
            "cccc": bind("cccc", (norb, norb, norb, norb)),
        }
    H0 = bind("H0", (1,))[0]

    return Integral(norb, restricted, bogoliubov, H0, H1, H2)

def read(filename, norb, restricted, bogoliubov, fmt, **kwargs):
    if fmt == "FCIDUMP":
        return readFCIDUMP(filename, norb, restricted, bogoliubov, **kwargs)
    elif fmt == "HDF5":
        return readHDF5(filename, norb, restricted, bogoliubov, **kwargs)
    elif fmt == "MMAP":
        return readMMAP(filename, norb, restricted, bogoliubov, **kwargs)
    else:
        raise Exception("Unrecognized formt %s" % fmt)

def dump(filename, Ham, fmt, **kwargs):
    if fmt == "FCIDUMP":
        return dumpFCIDUMP(filename, Ham, **kwargs)
    elif fmt == "HDF5":
        return dumpHDF5(filename, Ham, **kwargs)
    elif fmt == "MMAP":
        return dumpMMAP(filename, Ham, **kwargs)
    else:
        raise Exception("Unrecognized formt %s" % fmt)

def test():
    from subprocess import call
    from tempfile import mkdtemp
    import numpy.linalg as la

    log.result("Testing Bogoliubov unrestricted integrals ...")
    input = "../block/dmrg_tests/bcs/DMETDUMP"
    Ham = read(input, 8, False, True, "FCIDUMP")
    output = "../block/dmrg_tests/bcs/DMETDUMPtest"
    dump(output, Ham, "FCIDUMP")
    s = call(["diff", input, output])

    if s == 0:
        log.result("... Successful")
        call(["rm", output])
    else:
        log.result("... Failed")
        log.result("Manually check input vs. output file: %s %s", input, output)
    log.result("")

    log.result("Testing Hubbard restricted integrals ...")
    input = "../block/dmrg_tests/hubbard/FCIDUMP"
    Ham = read(input, 12, True, False, "FCIDUMP")
    output = "../block/dmrg_tests/hubbard/FCIDUMPtest"
    dump(output, Ham, "FCIDUMP")
    s = call(["diff", input, output])

    if s == 0:
        log.result("... Successful")
        call(["rm", output])
    else:
        log.result("... Failed")
        log.result("Manually check input vs. output file: %s %s", input, output)
    log.result("")

    log.result("Testing memory map integrals ...")
    output = mkdtemp(prefix = "MMAP_INT", dir = "../examples")
    log.result("Stored to %s", output)
    dump(output, Ham, "MMAP")
    Ham1 = read(output, 12, True, False, "MMAP", copy = False)
    log.result("H0: %f %f", Ham1.H0, Ham.H0)
    log.result("H1: %s", np.allclose(Ham1.H1["cd"], Ham.H1["cd"]))
    log.result("H2: %s", np.allclose(Ham1.H2["ccdd"], Ham.H2["ccdd"]))
    if Ham1.H0 == Ham.H0 and np.allclose(Ham1.H1["cd"], Ham.H1["cd"]) and np.allclose(Ham1.H2["ccdd"], Ham.H2["ccdd"]):
        log.result("...Successful")
        call(["rm", "-rf", output])
    else:
        log.result("... Failed")

if __name__ == "__main__":
    test()
