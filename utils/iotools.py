import libdmet.system.lattice as lat
import libdmet.utils.logger as log
import numpy as np
import sys

def build_unitcell(cellsize, atoms, frac = False, center = None):
    sites = map(lambda t: t[0], atoms)
    if frac:
        sites = map(lambda s: lat.Frac2Real(cellsize, s), sites)
        if center is not None:
            center = lat.Frac2Real(cellsize, center)

    if center is not None:
        T = 0.5 * np.sum(cellsize, axis = 0) - center
        sites = map(lambda s: s + T, sites)

    sites_names = map(lambda (s, a): (s, a[1]), zip(sites, atoms))
    return lat.UnitCell(cellsize, sites_names)

def __stat(_list):
    _dict = {}
    for idx, val in enumerate(_list):
        if val in _dict.keys():
            _dict[val].append(idx)
        else:
            _dict[val] = [idx]
    return _dict

def sc2POSCAR(sc, fout, elements = None):
    log.eassert(sc.dim == 3, "dimension error")
    sites = sc.sites
    name_dict = __stat(sc.names)

    if elements is not None:
        log.eassert(set(elements) == set(name_dict.keys()), "specified elements different from unitcell data")
    else:
        elements = name_dict.keys()

    fout.write(" ".join(elements))
    fout.write("  generated using libdmet.utils.iotools\n")
    fout.write("%10.6f\n" % 1)
    for d in range(sc.dim):
        fout.write("%20.12f%20.12f%20.12f\n" % tuple(sc.size[d]))
    for key in elements:
        fout.write("%3d " % len(name_dict[key]))
    fout.write("\n")
    fout.write("Cartesian\n")
    for key in elements:
        for s in name_dict[key]:
            fout.write("%20.12f%20.12f%20.12f\n" % tuple(sites[s]))

def sc2XYZ(sc, fout, elements = None):
    log.eassert(sc.dim == 3, "dimension error")
    sites = sc.sites
    name_dict = __stat(sc.names)

    if elements is not None:
        log.eassert(set(elements) == set(name_dict.keys()), "specified elements different from unitcell data")
    else:
        elements = name_dict.keys()

    fout.write("%d\n" % sc.nsites)
    fout.write("generated using libdmet.utils.iotools\n")

    for key in elements:
        for s in name_dict[key]:
            fout.write("%2s%20.12f%20.12f%20.12f\n" % ((key,) + tuple(sites[s])))

def struct_dump(cellsize, scsize, atoms, fmt, frac = False, center = None, \
        filename = None, elements = None):
    uc = build_unitcell(cellsize, atoms, frac, center)
    sc = lat.SuperCell(uc, scsize)
    if filename is not None:
        fout = open(filename, "w")
    else:
        fout = sys.stdout

    if fmt.upper() == "POSCAR":
        sc2POSCAR(sc, fout, elements = elements)
    elif fmt.upper() == "XYZ":
        sc2XYZ(sc, fout, elements = elements)
    else:
        log.error("Invalid dump format %s", fmt.upper())
    
    if filename is not None:
        fout.close()
