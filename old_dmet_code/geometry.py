#
# File: geometry.py
# Author: Bo-Xiao Zheng <boxiao@princeton.edu>
#

import numpy as np
import numpy.linalg as la
import itertools as it
from cmath import sqrt

from utils import ToSpinOrb, ToSpatOrb

class Topology2D(object):
  def __init__(self, lattice):
    scsites = lattice.supercell.sites
    nscsites = lattice.supercell.nsites
    scshape = lattice.supercell.size

    def compute_frac_coord(scshape, scsites):
      p = la.inv(scshape)
      frac = map(lambda x: np.dot(x, p), scsites)
      com1 = reduce(np.ndarray.__add__, frac) / len(frac)
      com2 = np.array([0.5, 0.5])
      return map(lambda x: x + com2 - com1, frac)

    frac_coord = compute_frac_coord(scshape, scsites)
      
    def get_phase(frac):
      if frac[0] + frac[1] < 1.:
        if frac[0] > frac[1]:
          return 1
        else:
          return 2
      else:
        if frac[0] > frac[1]:
          return 4
        else:
          return 3

    for frac in frac_coord:
      assert(np.all(frac > 0.) and np.all(frac < 1.))
    phases = [get_phase(frac) for frac in frac_coord]

    def map_sc(sc_pos):
      temp = []
      if sc_pos[0] <= 0.5*lattice.scsize[0] and sc_pos[1] <= 0.5*lattice.scsize[1]:
        temp.append(sc_pos)
      if sc_pos[0] >= 0.5*lattice.scsize[0] and sc_pos[1] <= 0.5*lattice.scsize[1]:
        temp.append(sc_pos-np.array([lattice.scsize[0], 0]))
      if sc_pos[0] <= 0.5*lattice.scsize[0] and sc_pos[1] >= 0.5*lattice.scsize[1]:
        temp.append(sc_pos-np.array([0, lattice.scsize[1]]))
      if sc_pos[0] >= 0.5*lattice.scsize[0] and sc_pos[1] >= 0.5*lattice.scsize[1]:
        temp.append(sc_pos-lattice.scsize)
      return tuple(temp)

    # now arrange the cells
    self.sclist = map(map_sc, lattice.supercell_list)
    sqsize = min(lattice.scsize/2)
    self.corners = {(0, 0): [np.array([-0.5, -0.5]), np.array([-0.5, 0.5]), \
        np.array([0.5, -0.5]), np.array([0.5, 0.5])]}
    def perimeter(x):
      return [(-x, -x)] + [(i,-x) for i in range(-x+1, x+1)] + \
          [(x,i) for i in range(-x+1, x+1)] + [(i,x) for i in range(x-1, -x-1, -1)] + \
          [(-x,i) for i in range(x-1, -x, -1)]
    
    ol = 0.5
    damp = 12
    for x in range(1, sqsize+1):
      il = ol
      ol += 1./float(x+damp)**0.5
      assert(ol/(2*x+1) < ol-il)
      for i,j in perimeter(x):
        order = range(4)
        s1 = s2 = 1
        if abs(i) == x and abs(j) == x:
          if i < 0:
            s1 = -1
            order[:2], order[2:] = order[2:], order[:2]
          if j < 0:
            s2 = -1
            order[0], order[1] = order[1], order[0]
            order[2], order[3] = order[3], order[2]
          temp = [np.array([s1*il, s2*il]), np.array([s1*ol*(1-1./(x+0.5)), s2*ol]), \
              np.array([s1*ol, s2*ol*(1-1./(x+0.5))]), np.array([s1*ol, s2*ol])]
          self.corners[i,j] = [temp[order[m]] for m in range(4)]
        elif abs(i) == x:
          if i < 0:
            s1 = -1
            order[:2], order[2:] = order[2:], order[:2]
          temp = [np.array([s1*il, il*(j-0.5)/(x-0.5)]), np.array([s1*il, il*(j+0.5)/(x-0.5)]), \
              np.array([s1*ol, ol*(j-0.5)/(x+0.5)]), np.array([s1*ol, ol*(j+0.5)/(x+0.5)])]
          self.corners[i,j] = [temp[order[m]] for m in range(4)]
        elif abs(j) == x:
          if j < 0:
            s2 = -1
            order[0], order[1] = order[1], order[0]
            order[2], order[3] = order[3], order[2]
          temp = [np.array([il*(i-0.5)/(x-0.5), s2*il]), np.array([ol*(i-0.5)/(x+0.5), s2*ol]), \
              np.array([il*(i+0.5)/(x-0.5), s2*il]), np.array([ol*(i+0.5)/(x+0.5), s2*ol])]
          self.corners[i, j] = [temp[order[m]] for m in range(4)]
        else:
          raise Exception()
     
    recsize = max(lattice.scsize/2)
    il0 = ol
    for x in range(sqsize+1, recsize+1):
      il = ol
      ol += 1./float(x+damp)**0.5
      for j in range(-sqsize, sqsize+1):
        if lattice.scsize[0] > lattice.scsize[1]:
          self.corners[x, j] = [np.array([il, il0*(j-0.5)/(sqsize+0.5)]), np.array([il, il0*(j+0.5)/(sqsize+0.5)]), \
              np.array([ol, il0*(j-0.5)/(sqsize+0.5)]), np.array([ol, il0*(j+0.5)/(sqsize+0.5)])]
          self.corners[-x, j] = [np.array([-ol, il0*(j-0.5)/(sqsize+0.5)]), np.array([-ol, il0*(j+0.5)/(sqsize+0.5)]), \
              np.array([-il, il0*(j-0.5)/(sqsize+0.5)]), np.array([-il, il0*(j+0.5)/(sqsize+0.5)])]
        else:
          self.corners[j, x] = [np.array([il0*(j-0.5)/(sqsize+0.5), il]), np.array([il0*(j-0.5)/(sqsize+0.5), ol]), \
              np.array([il0*(j+0.5)/(sqsize+0.5), il]), np.array([il0*(j+0.5)/(sqsize+0.5), ol])]
          self.corners[j, -x] = [np.array([il0*(j-0.5)/(sqsize+0.5), -ol]), np.array([il0*(j-0.5)/(sqsize+0.5), -il]), \
              np.array([il0*(j+0.5)/(sqsize+0.5), -ol]), np.array([il0*(j+0.5)/(sqsize+0.5), -il])]
        
    def get_size(x):
      if x == 1:
        return 0.5
      elif x % 2 == 1:
        return 0.5 + sum([1./float(i+damp)**0.5 for i in range(1, x/2+1)])
      else:
        return 0.5 + sum([1./float(i+damp)**0.5 for i in range(1, x/2)]) + 0.5/float(x/2+damp)**0.5

    self.lims = [get_size(x) for x in lattice.scsize]

    # now put in the atoms
    def get_pos(image, site):
      cor = self.corners[tuple(image)]
      com = np.average(cor, axis = 0)
      if phases[site] == 1:
        x = np.array([[0,0,1],[1,0,1],[0.5,0.5,1]])
        y = np.array([cor[0], cor[2], com])
      elif phases[site] == 2:
        x = np.array([[0,0,1],[0,1,1],[0.5,0.5,1]])
        y = np.array([cor[0], cor[1], com])
      elif phases[site] == 3:
        x = np.array([[0,1,1],[1,1,1],[0.5,0.5,1]])
        y = np.array([cor[1], cor[3], com])
      else:
        x = np.array([[1,0,1],[1,1,1],[0.5,0.5,1]])
        y = np.array([cor[2], cor[3], com])
      coefs = np.dot(la.inv(x), y)
      A = coefs[:2]
      b = coefs[2]
      return np.dot(frac_coord[site], A) + b
      

    self.frac_pos = []
    for cell_images in self.sclist:
      if len(cell_images) == 1:
        image = cell_images[0]        
        for i in range(nscsites):
          self.frac_pos.append(get_pos(image, i))
      elif len(cell_images) == 2:
        if cell_images[0][0] == cell_images[1][0]:        
          for i in range(nscsites):
            if frac_coord[i][1] < 0.5:
              self.frac_pos.append(get_pos(cell_images[0], i))
            else:
              self.frac_pos.append(get_pos(cell_images[1], i))
        else:
          for i in range(nscsites):
            if frac_coord[i][0] < 0.5:
              self.frac_pos.append(get_pos(cell_images[0], i))
            else:
              self.frac_pos.append(get_pos(cell_images[1], i))
      elif len(cell_images) == 4:
        for i in range(nscsites):
          if frac_coord[i][0] < 0.5 and frac_coord[i][1] < 0.5:
            self.frac_pos.append(get_pos(cell_images[0], i))
          elif frac_coord[i][0] >= 0.5 and frac_coord[i][1] < 0.5:
            self.frac_pos.append(get_pos(cell_images[1], i))
          elif frac_coord[i][0] < 0.5 and frac_coord[i][1] >= 0.5:
            self.frac_pos.append(get_pos(cell_images[2], i))
          elif frac_coord[i][0] >= 0.5 and frac_coord[i][1] >= 0.5:
            self.frac_pos.append(get_pos(cell_images[3], i))
          else:
            raise Exception()
      else:
        raise Exception()

    self.pos = [np.dot(f, scshape) for f in self.frac_pos]
    self.distance = np.zeros((nscsites, nscsites*len(self.sclist))) # only measure between those within the impurity
    
    translate = [np.dot(t*np.array(self.lims)*2, scshape) for t in it.product([-1, 0, 1], repeat = 2)]
    for i in range(nscsites):
      for j in range(nscsites*len(self.sclist)):
        self.distance[i, j] = min([la.norm(self.pos[i] - self.pos[j] + t) for t in translate])
        if i != j and self.distance[i,j] < 1e-5:
          self.distance[i,j] = 0.7
    
    #self.test_plot_frac()
    #self.test_plot_real(scshape)

  def test_plot_frac(self):
    import matplotlib.pyplot as plt
    for idx, c in self.corners.items():
      x = [point[0] for point in c]
      y = [point[1] for point in c]
      x[2], x[3] = x[3], x[2]
      y[2], y[3] = y[3], y[2]
      x.append(x[0])
      y.append(y[0])
      plt.plot(x, y, "k-")
      #plt.plot(np.average(x[:4]), np.average(y[:4]), "ro")
      #plt.text(np.average(x[:4]), np.average(y[:4]), "(%d,%d)" % idx, ha = 'center', va = 'center')
    for pos in self.frac_pos:
      plt.plot(pos[0], pos[1], "ro")
    plt.plot([-self.lims[0], -self.lims[0], self.lims[0], self.lims[0], -self.lims[0]], \
        [-self.lims[1], self.lims[1], self.lims[1], -self.lims[1], -self.lims[1]], "k--")
    plt.show()

  def test_plot_real(self, scshape):
    import matplotlib.pyplot as plt
    
    corner = [np.dot(f, scshape) for f in [np.array([-self.lims[0], -self.lims[1]]), \
        np.array([-self.lims[0], self.lims[1]]), np.array([self.lims[0], self.lims[1]]), \
        np.array([self.lims[0], -self.lims[1]]), np.array([-self.lims[0], -self.lims[1]])]]
    px = [p[0] for p in corner]
    py = [p[1] for p in corner]
    
    fig = plt.figure(figsize = (corner[2][0]+0.1,corner[2][1]+0.1))
    ax = plt.axes(frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    #ax.set_frame_on(False)
    for idx, c in self.corners.items():
      c1 = [np.dot(f, scshape) for f in c]
      x = [point[0] for point in c1]
      y = [point[1] for point in c1]
      x[2], x[3] = x[3], x[2]
      y[2], y[3] = y[3], y[2]
      x.append(x[0])
      y.append(y[0])
      plt.plot(x, y, "k-")
      #plt.plot(np.average(x[:4]), np.average(y[:4]), "ro")
      #plt.text(np.average(x[:4]), np.average(y[:4]), "(%d,%d)" % idx, ha = 'center', va = 'center')
    for i, pos in enumerate(self.pos):
      plt.plot(pos[0], pos[1], "ro")
    plt.plot(px, py, "k--")
    plt.xlim(corner[0][0]-0.1, corner[2][0]+0.1)
    plt.ylim(corner[0][1]-0.1, corner[2][1]+0.1)
    plt.savefig("Topology.png", dpi = 150, bbox_inches = "tight")
    #plt.show()
  

class FUnitCell(object):
  def __init__(self, size, sites): # sites is a list of tuples
    self.size = np.array(size) # unit cell shape
    assert(self.size.shape[0] == self.size.shape[1])
    self.dim = self.size.shape[0]
    self.sites = []
    self.names = []
    for site in sites:
      self.sites.append(np.array(site[0])) # coordination
      self.names.append(site[1])
    self.nsites = len(self.sites)
    self.site_dict = dict(zip([tuple(x) for x in self.sites], range(self.nsites)))

  def __str__(self):
    r = "UnitCell Shape\n%s\nSites:\n" % self.size
    for i in range(len(self.sites)):
      r += "%-10s%-10s\t" % (self.names[i], self.sites[i])
      if (i+1)%6 == 0:
        r+= "\n"
    r += "\n\n"
    return r

class FSuperCell(object):
  def __init__(self, unitcell, size):
    self.unitcell = unitcell
    self.dim = unitcell.dim
    self.csize = np.array(size)
    self.size = np.dot(np.diag(self.csize), unitcell.size)
    self.ncells = np.product(self.csize)
    self.nsites = unitcell.nsites * self.ncells

    self.sites = []
    self.names = []
    self.unitcell_list = []

    for p in it.product(*tuple([range(a) for a in self.csize])):
      self.unitcell_list.append(np.array(p))
      for i in range(len(unitcell.sites)):
        self.sites.append(np.dot(np.array(p), unitcell.size)  + unitcell.sites[i])
        self.names.append(unitcell.names[i])
    
    self.unitcell_dict = dict(zip([tuple(x) for x in self.unitcell_list], range(self.ncells)))
    self.site_dict = dict(zip([tuple(x) for x in self.sites], range(self.nsites)))
    self.fragments = None

  def __str__(self):
    r = self.unitcell.__str__()
    r += "SuperCell Shape\n"
    r += self.size.__str__()
    r += "\nNumber of Sites:%d\n" % self.nsites
    for i, f in enumerate(self.fragments):
      r += "Fragment %3d: %s\n" % (i, f)
    r += "\n"
    return r

  def set_fragments(self, frag):
    if frag is None:
      self.fragments = [range(self.nsites)]
    else:
      sites = []
      for f in frag:
        sites += f.tolist()

      if len(sites) != len(set(sites)): # check duplicate entries
        raise Exception("Fragments have overlaps")
      for i in range(len(sites)):
        if not sites[i] in range(self.nsites):
          raise Exception("SiteId %d is not a valid site" % sites[i])
      self.fragments = frag

class FLattice(object):
  def __init__(self, size, sc, bc, OrbType = "RHFB"):
    self.supercell = sc
    self.dim = sc.dim

    self.scsize = np.array(size)
    self.size = np.dot(np.diag(self.scsize), sc.size)
    self.nscells = np.product(self.scsize)
    self.nsites = sc.nsites * self.nscells

    if bc == "pbc":
      self.bc = 1
    else:
      raise Exception("Unsupported Boudary condition")

    self.sites = []
    self.names = []
    self.supercell_list = []
    for p in it.product(*tuple([range(a) for a in self.scsize])):
      self.supercell_list.append(np.array(p))
      for i in range(len(sc.sites)):
        self.sites.append(np.dot(np.array(p), sc.size)  + sc.sites[i])
        self.names.append(sc.names[i])
    self.supercell_dict = dict(zip([tuple(x) for x in self.supercell_list], range(self.nscells)))
    self.site_dict = dict(zip([tuple(x) for x in self.sites], range(self.nsites)))
    self.OrbType = OrbType
    self.h0 = None
    self.h0_kspace = None
    self.fock = None
    self.fock_kspace = None
    self.neighbor1 = None
    self.neighbor2 = None

  def __str__(self):
    r = self.supercell.__str__()
    r += "Lattice Shape\n%s\n" % self.scsize
    r += "Number of SuperCells: %4d\n" % self.nscells
    r += "Number of Sites:      %4d\n" % self.nsites
    r += "\n"
    return r

  def sc_idx2pos(self, i):
    return self.supercell_list[i % self.nscells]

  def sc_pos2idx(self, p):
    return self.supercell_dict[tuple(p % self.scsize)]

  def add(self, i, j):
    return self.sc_pos2idx(self.sc_idx2pos(i) + self.sc_idx2pos(j))

  def set_Hamiltonian(self, Ham):
    self.Ham = Ham
  
  def get_h0(self, kspace = False, SpinOrb = True):
    if kspace:
      if self.h0_kspace is None:
        self.h0_kspace = self.FFTtoK(self.get_h0(SpinOrb = False))
      if self.OrbType == "UHFB" and SpinOrb:
        return np.array([ToSpinOrb(self.h0_kspace[i]) for i in range(self.nscells)])
      else:
        return self.h0_kspace
    elif self.h0 is None:
        self.h0 = self.Ham.get_h0()
    if self.OrbType == "UHFB" and SpinOrb:
      return np.array([ToSpinOrb(self.h0[i]) for i in range(self.nscells)])
    else:
      return self.h0

  def get_fock(self, kspace = False, SpinOrb = True):
    if kspace:
      if self.fock_kspace is None:
        self.fock_kspace = self.FFTtoK(self.get_fock(SpinOrb = False))
      if self.OrbType == "UHFB" and SpinOrb:
        return np.array([ToSpinOrb(self.fock_kspace[i]) for i in range(self.nscells)])
      else:
        return self.fock_kspace
    elif self.fock is None:
        self.fock = self.Ham.get_fock()
    if self.OrbType == "UHFB" and SpinOrb:
      return np.array([ToSpinOrb(self.fock[i]) for i in range(self.nscells)])
    else:
      return self.fock

  def FFTtoK(self, A):
    # currently only for pbc
    assert(self.bc == 1)
    B = A.reshape(tuple(self.scsize) + A.shape[-2:])
    return np.fft.fftn(B, axes = range(self.dim)).reshape(A.shape)

  def FFTtoT(self, A):
    assert(self.bc == 1)
    B = A.reshape(tuple(self.scsize) + A.shape[-2:])
    C = np.fft.ifftn(B, axes = range(self.dim)).reshape(A.shape)
    if np.allclose(C.imag, 0.):
      return C.real
    else:
      return C

  def get_kpoints(self):
    kpoints = [np.fft.fftfreq(self.scsize[d], 1/(2*np.pi)) for d in range(self.dim)]
    return kpoints

  def expand(self, A, dense = False):
    # expand reduced matrices, eg. Hopping matrix
    assert(self.bc == 1)
    B = np.zeros((self.nsites, self.nsites))
    scnsites = self.supercell.nsites
    if dense:
      for i, j in it.product(range(self.nscells), repeat = 2):
        idx = self.sc_pos2idx(self.sc_idx2pos(i) + self.sc_idx2pos(j))
        B[i*scnsites:(i+1)*scnsites, idx*scnsites:(idx+1)*scnsites] = A[j]
    else:      
      nonzero = [i for i in range(A.shape[0]) if not np.allclose(A[i], 0.)]
      for i in range(self.nscells):
        for j in nonzero:
          idx = self.sc_pos2idx(self.sc_idx2pos(i) + self.sc_idx2pos(j))          
          B[i*scnsites:(i+1)*scnsites, idx*scnsites:(idx+1)*scnsites] = A[j]
    return B

  def transpose_reduced(self, A):
    assert(self.bc == 1)
    B = np.zeros_like(A)
    for n in range(self.nscells):
      B[n] = A[self.sc_pos2idx(-self.sc_idx2pos(n))].T
    return B

  def get_NearNeighbor(self, sites = None, sites2 = None):
    # return nearest neighbors
    # two lists are returned, first is within the lattice, second is along boundary
      
    if sites == None:
      sites = [s for s in range(self.nsites)]
    if sites2 == None:
      sites2 = [s for s in range(self.nsites)]

    neighbor1 = []
    neighbor2 = []
    shifts = [np.array(x) for x in it.product([-1, 0, 1], repeat = self.dim) if x != (0,) * self.dim]
    # first find supercell neighbors
    sc = self.supercell
    for s1 in sites:
      sc1 = s1 / sc.nsites
      sc2 = [sc1] + [self.sc_pos2idx(self.sc_idx2pos(sc1) + shift) for shift in shifts]
      for s2 in list(set(sites2) & set(it.chain.from_iterable([range(s*sc.nsites, (s+1)*sc.nsites)for s in sc2]))):
        if abs(la.norm(self.sites[s2] - self.sites[s1]) - 1.) < 1e-5:
            neighbor1.append((s1, s2))
        else:
          for shift in shifts:
            if abs(la.norm(self.sites[s2]-self.sites[s1] - np.dot(shift, self.size)) - 1.) < 1e-5:
              neighbor2.append((s1, s2))
              break
    return neighbor1, neighbor2

  def get_2ndNearNeighbor(self, sites = None, sites2 = None):
    if sites == None:
      sites = [s for s in range(self.nsites)]
    if sites2 == None:
      sites2 = [s for s in range(self.nsites)]
    neighbor1 = []
    neighbor2 = []
    shifts = [np.array(x) for x in it.product([-1, 0, 1], repeat = self.dim) if x != (0,) * self.dim]
    # first find supercell neighbors
    sc = self.supercell
    for s1 in sites:
      sc1 = s1 / sc.nsites
      sc2 = [sc1] + [self.sc_pos2idx(self.sc_idx2pos(sc1) + shift) for shift in shifts]
      for s2 in list(set(sites2) & set(it.chain.from_iterable([range(s*sc.nsites, (s+1)*sc.nsites)for s in sc2]))):
        if abs(la.norm(self.sites[s2] - self.sites[s1]) - sqrt(2.).real) < 1e-5:
          neighbor1.append((s1, s2))
        else:
          for shift in shifts:
            if abs(la.norm(self.sites[s2]-self.sites[s1] - np.dot(shift, self.size)) - sqrt(2.).real) < 1e-5:
              neighbor2.append((s1, s2))
              break
    return neighbor1, neighbor2

def BuildLatticeFromInput(inp_geom, OrbType = "RHFB", verbose = 5):
  unit = FUnitCell(inp_geom.UnitCell["Shape"],
                   inp_geom.UnitCell["Sites"])
  sc = FSuperCell(unit, np.array(inp_geom.ClusterSize))
  sc.set_fragments(inp_geom.Fragments)

  assert(np.allclose(np.array(inp_geom.LatticeSize) % np.array(inp_geom.ClusterSize), 0.))
  lattice = FLattice(np.array(inp_geom.LatticeSize)/np.array(inp_geom.ClusterSize),
                     sc, inp_geom.BoundaryCondition, OrbType)

  if verbose > 4:
    print "\nGeometry Summary"
    print lattice
  
  return lattice

def Topology(lattice):
  if lattice.dim == 2:
    return Topology2D(lattice)
  else:
    return None

if __name__ == "__main__":
  pass
