class IdxSymmetry(object):
    # only symmetric and antisymmetric implemented
    def __init__(self, symm, antisymm):
        self._symm = tuple(symm)
        self._antisymm = tuple(antisymm)

    def __eq__(self, other):
        return self.symm == other.symm and self.antisymm == other.antisymm
    
    def permute(self, order, indices):
        return tuple([indices[i] for i in order])

    def symm(self, indices):
        return map(lambda p: self.permute(p, indices), self._symm)

    def antisymm(self, indices):
        return map(lambda p: self.permute(p, indices), self._antisymm)

class IdxNoSymm(IdxSymmetry):
    def __init__(self, n):
        symm = [tuple(range(n))]
        asymm = []
        IdxSymmetry.__init__(self, symm, asymm)

class IdxSymm(IdxSymmetry):
    def __init__(self):
        symm = [(0,1), (1,0)]
        asymm = []
        IdxSymmetry.__init__(self, symm, asymm)

class IdxAntisymm(IdxSymmetry):
    def __init__(self):
        symm = [(0,1)]
        asymm = [(1,0)]
        IdxSymmetry.__init__(self, symm, asymm)

class Idx4FoldSymm(IdxSymmetry):
    def __init__(self):
        symm = [(0,1,2,3), (1,0,2,3), (0,1,3,2), (1,0,3,2)]
        asymm = []
        IdxSymmetry.__init__(self, symm, asymm)

class Idx8FoldSymm(IdxSymmetry):
    def __init__(self):
        symm = [(0,1,2,3), (1,0,2,3), (0,1,3,2), (1,0,3,2), \
          (2,3,0,1), (3,2,0,1), (2,3,1,0), (3,2,1,0)]
        asymm = []
        IdxSymmetry.__init__(self, symm, asymm)

class Idx8FoldAntisymm(IdxSymmetry): # eg. cccc vector
    def __init__(self):
        symm = [(0,1,2,3), (1,0,3,2), (3,2,1,0), (2,3,0,1)]
        asymm = [(1,0,2,3), (0,1,3,2), (3,2,0,1), (2,3,1,0)]
        IdxSymmetry.__init__(self, symm, asymm)

class Idx4FoldAntisymm(IdxSymmetry): # eg. cccc vector
    def __init__(self):
        symm = [(0,1,2,3), (1,0,3,2)]
        asymm = [(1,0,2,3), (0,1,3,2)]
        IdxSymmetry.__init__(self, symm, asymm)

class Idx2FoldAntisymm(IdxSymmetry): # eg. cccd vector
    def __init__(self):
        symm = [(0,1,2,3)]
        asymm = [(1,0,2,3)]
        IdxSymmetry.__init__(self, symm, asymm)        
