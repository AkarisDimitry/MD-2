"""
Main Interaction module.
"""

import numpy as np
import itertools
import sklearn.neighbors
import ctypes as C
LJforces = C.CDLL('interaction/LennardJones.so')

class Interaction(object):
  """
  Base Interaction class.
  """
  def __init__(self, types):
    self.types = types

  def forces(self, x, v, t):
    """
    Main loop calculation.

    NOTE: This is highly experimental and slow.
    It is just meant to be a proof of concept for the main loop, but
    has to change *dramatically*, even in arity, when we want to add
    lists of neighbors, parallelization and so on.
    """
    return np.zeros_like(x), 0.0

class ShortRange(Interaction):
  """
  Base short-range class
  """
  def __init__(self, types, rcut, shift_style='None'):
    """
    Base short-range class

    Parameters
    ----------

    rcut : float
        The cut radius parameter

    shift_style: {'None', 'Displace', 'Splines'}
        Shift style when approaching rcut

    .. note:: 'Splines' not implemented yet
    """
    self.rcut = rcut
    self.shift_style = shift_style
    Interaction.__init__(self, types)


class LennardJones(ShortRange):
  """
  Lennard-Jones potential
  """
  def __init__(self, types, rcut, eps, sigma, shift_style='None'):
    self.eps = eps
    self.sigma = sigma
    ShortRange.__init__(self, types, rcut, shift_style)

  def Cforces(self, x, v, t):
    """
    Calculos de las fuerzas segun el potencial de LennardJones utilizando la libreia dinamica Lennajones.so 
    """
    in1 = x.astype( C.c_float)
    in2 = v.astype( C.c_float)
    out1 = np.zeros(x.shape, dtype=np.float32)
    out2 = np.zeros(1, dtype=np.float32)
    
    fltp = C.POINTER(C.c_float)
    LJforces.forces( in1.ctypes.data_as(fltp),
               in2.ctypes.data_as(fltp),
               out1.ctypes.data_as(fltp),
               out2.ctypes.data_as(fltp),
               
               C.c_int(x.shape[0]),
               C.c_int(50)
                   
           )
    
    return out1, out2[0]

  def forces(self, x, v, t):
    """
    Calculate Lennard-Jones force
    


    """
    x1 = x[t == self.types[0]]
    x2 = x[t == self.types[1]]
    i1 = np.arange(len(x))[t == self.types[0]]
    i2 = np.arange(len(x))[t == self.types[1]]
    forces = np.zeros_like(x)
    energ = 0
    # I have to split it to avoid double-counting. Don't want to get
    # too fancy since it will change when creating neighbor lists
    if self.types[0] == self.types[1]:
      for i, s1 in enumerate(x1):
        for j, s2 in enumerate(x2[i+1:]):
          f = self.pair_force(s1, s2)
          ii = i1[i]
          #print s1 
          #if np.abs(np.sum(f)) > 0.001:
               #print i, j+i+1, f
          jj = i2[j+i+1]
          forces[ii] += f
          forces[jj] -= f 
          energ += self.pair_energ(s1, s2)
    else:
      for i, s1 in enumerate(x1):
        for j, s2 in enumerate(x2):
          f = self.pair_force(s1, s2)
          ii = i1[i]
          jj = i2[j]
          forces[ii] += f
          forces[jj] -= f
          energ += self.pair_energ(s1, s2)
    return forces, energ

  def forces_neiboarhood(self, x, v, t, nh, l, rcut):
      """
      Calculate Lennard-Jones force
      Neiboarhood list implementation

      """
      x1 = x[t == self.types[0]]
      x2 = x[t == self.types[1]]
      i1 = np.arange(len(x))[t == self.types[0]]
      i2 = np.arange(len(x))[t == self.types[1]]
      forces = np.zeros_like(x)
      energ = 0
      pairs = [] 

      if self.types[0] == self.types[1]:
          for i, s1 in enumerate(x):
              for n in itertools.product( [0,1], repeat=3):
                   
                    for m in nh[int(s1[0]/rcut)+n[0]][int(s1[1]/rcut)+n[1]][int(s1[2]/rcut)+n[2]]:
                        if i != m  and not [i, m] in pairs and not [m, i] in pairs :
                          s2 = x[m,:]
                          #print m, i                            
                          f = self.pair_force(s1, s2)
                          forces[i] += f
                          forces[m] -= f
                    
                          energ += self.pair_energ(s1, s2)
                          pairs.append([i, m])

      return forces, energ

  def force_tree(self, x, v, t, Mtree, rcut):
      """
      sklearn neighbors KDTree implementation
      Mtree : is a KDTree (from library SKlearn)
      """

      forces = np.zeros_like(x)
      energ = 0
      for i, s1 in enumerate(x):
          for m in Mtree.query_radius(s1.reshape(1,-1), r=rcut)[0]:
              if m != i and m>i:
                 s2 = x[m,:]
                 f = self.pair_force(s1, s2)
                 forces[i] += f
                 forces[m] -= f
                 energ += self.pair_energ(s1, s2)
      return forces, energ

  def pair_force(self, s1, s2):
    d = np.linalg.norm(s1-s2)
    if d > self.rcut:
      return np.zeros_like(s1)
    ljf = 24*self.eps*(2*self.sigma**12/d**14 - self.sigma**6/d**8)*(s1-s2)
    if self.shift_style == 'None':
      return ljf
    elif self.shift_style == 'Displace':
      return ljf

  def pair_energ(self, s1, s2):
    vcut = 4*self.eps*(self.sigma**12/self.rcut**12 - self.sigma**6/self.rcut**6)
    d = np.linalg.norm(s1-s2)
    if d >= self.rcut:
      return 0
    ljf = 4*self.eps*(self.sigma**12/d**12 - self.sigma**6/d**6)
    if self.shift_style == 'None':
      return ljf
    elif self.shift_style == 'Displace':
      return ljf - vcut



class Morse(ShortRange):
  """
  Lennard-Jones potential
  """
  def __init__(self, types, rcut, bener, blem, beta, shift_style='None'):
     self.bener = bener
     self.beta = beta
     self.blem = blem
     ShortRange.__init__(self, types, rcut, shift_style)

  def forces(self, x, v, t):
     """
     Calculate Lennard-Jones force
     """
     x1 = x[t == self.types[0]]
     x2 = x[t == self.types[1]]
     i1 = np.arange(len(x))[t == self.types[0]]
     i2 = np.arange(len(x))[t == self.types[1]]
     forces = np.zeros_like(x)
     energ = 0
            # I have to split it to avoid double-counting. Don't want to get
            # too fancy since it will change when creating neighbor lists
     if self.types[0] == self.types[1]:
          for i, s1 in enumerate(x1):
               for j, s2 in enumerate(x2[i+1:]):
                   f = self.pair_force(s1, s2)
                   ii = i1[i]
                   jj = i2[j+i+1]
                   forces[ii] += f
                   forces[jj] -= f
                   energ += self.pair_energ(s1, s2)
     else:
         for i, s1 in enumerate(x1):
              for j, s2 in enumerate(x2):
                  f = self.pair_force(s1, s2)
                  ii = i1[i]
                  jj = i2[j]
                  forces[ii] += f
                  forces[jj] -= f
                  energ += self.pair_energ(s1, s2)
     return forces, energ



  def pair_force(self, s1, s2):
      d = np.linalg.norm(s1-s2)
      if d > self.rcut:
         return np.zeros_like(s1)
      morf = 2.0*self.bener*(1.0-np.exp(-self.beta*(d-self.blem)))*np.exp(-self.beta*(d-self.blem))*self.beta*(s1-s2)/d
      if self.shift_style == 'None':
         return morf
      elif self.shift_style == 'Displace':
         return morf
  def pair_energ(self, s1, s2):
      vcut = -self.bener*(1.0-np.exp(-self.beta*(self.rcut-self.blem)))**2
      d = np.linalg.norm(s1-s2)
      if d >= self.rcut:
         return 0
      morf = -self.bener*(1.0-np.exp(-self.beta*(d-self.blem)))**2
      if self.shift_style == 'None':
         return morf
      elif self.shift_style == 'Displace':
         return morf - vcut

