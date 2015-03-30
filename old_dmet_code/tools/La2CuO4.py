import numpy as np

shape = np.array([
      [7.6011851839 , 0            , 0,],
      [0.11342497103, 7.60033887244, 0,],
      [1.92865253873, 1.90008471811, 6.5586,],
    ])

sites = [
    (np.array([ 0.00000,  0.00000,   0.00000,]), "Cu"),
    (np.array([ 0.50000,  0.00000,   0.00000,]), "Cu"),
    (np.array([ 0.00000,  0.50000,   0.00000,]), "Cu"),
    (np.array([ 0.50000,  0.50000,   0.00000,]), "Cu"),
    (np.array([ 0.25415-0.2768/4,  0.25415-0.2768/4,   0.27680,]), "La"),
    (np.array([ 0.74585-0.2768/4,  0.24585-0.2768/4,   0.27680,]), "La"),
    (np.array([ 0.24585-0.2768/4,  0.74585-0.2768/4,   0.27680,]), "La"),
    (np.array([ 0.75415-0.2768/4,  0.75415-0.2768/4,   0.27680,]), "La"),
    (np.array([-0.25415+0.2768/4, -0.25415+0.2768/4,  -0.27680,]), "La"),
    (np.array([ 0.25415+0.2768/4, -0.24585+0.2768/4,  -0.27680,]), "La"),
    (np.array([-0.24585+0.2768/4,  0.25415+0.2768/4,  -0.27680,]), "La"),
    (np.array([ 0.24585+0.2768/4,  0.24585+0.2768/4,  -0.27680,]), "La"),
    (np.array([ 0.25000-0.01680/4,  0.00000-0.01680/4,   0.01680,]), "O" ),
    (np.array([ 0.00000-0.01680/4,  0.25000-0.01680/4,   0.01680,]), "O" ),
    (np.array([ 0.75000+0.01680/4,  0.00000+0.01680/4,  -0.01680,]), "O" ),
    (np.array([ 0.50000+0.01680/4,  0.25000+0.01680/4,  -0.01680,]), "O" ),
    (np.array([ 0.25000+0.01680/4,  0.50000+0.01680/4,  -0.01680,]), "O" ),
    (np.array([ 0.00000+0.01680/4,  0.75000+0.01680/4,  -0.01680,]), "O" ),
    (np.array([ 0.75000-0.01680/4,  0.50000-0.01680/4,   0.01680,]), "O" ),
    (np.array([ 0.50000-0.01680/4,  0.75000-0.01680/4,   0.01680,]), "O" ),
    (np.array([-0.02020-0.36740/4, -0.02020-0.36740/4,   0.36740,]), "O" ),
    (np.array([ 0.52020-0.36740/4,  0.02020-0.36740/4,   0.36740,]), "O" ),
    (np.array([ 0.02020-0.36740/4,  0.52020-0.36740/4,   0.36740,]), "O" ),
    (np.array([ 0.47980-0.36740/4,  0.47980-0.36740/4,   0.36740,]), "O" ),
    (np.array([ 0.02020+0.36740/4,  0.02020+0.36740/4,  -0.36740,]), "O" ),
    (np.array([ 0.47980+0.36740/4, -0.02020+0.36740/4,  -0.36740,]), "O" ),
    (np.array([-0.02020+0.36740/4,  0.47980+0.36740/4,  -0.36740,]), "O" ),
    (np.array([ 0.52020+0.36740/4,  0.52020+0.36740/4,  -0.36740,]), "O" ),
    ]

sites = [(np.dot(site[0], shape), site[1]) for site in sites]

Geometry = {
  "UnitCell": {
    "Shape": shape,
    "Sites": sites,
  },
  "ClusterSize": np.array([1,1,1]),
  "LatticeSize": np.array([1,1,1]),
}