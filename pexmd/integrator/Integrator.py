"""
Main Integrator module
"""

import numpy as np
import ctypes as C
cvelvarlet = C.CDLL('integrator/libmymath.so')

class Integrator(object):
  """
  Integrator base class
  """

  def __init__(self, dt):
    self.dt = dt

  def first_step(self, x, v, a):
    return x, v

  def last_step(self, x, v, a):
    return x, v

class NVE(Integrator):
  """
  NVE base class
  """

class VelVerlet(NVE):
  """
  Velocity Verlet Integrator
  """
  def first_step(self, x, v, a):
    x = x + v*self.dt + 0.5*a*self.dt**2
    v = v + 0.5*a*self.dt
    return x, v

  def last_step(self, x, v, a):
    v = v + 0.5*a*self.dt
    return x, v

class NVT(Integrator):
  """
  NVT base class
  """
  def __init__(self, dt, temperature):
    self.temperature = temperature
    Integrator.__init()

class Andersen(NVT):
      """
      Andersen Thermostat
      """

      def __init__(self, dt, temperature, freq):
         self.freq = freq
         NVT.__init__(dt, temperature)

      def first_step(self, x, v, a):
         x = x + v*self.dt + 0.5*a*self.dt**2
         v = v + 0.5*a*self.dt
         return x, v

      def last_step(self, x, v, a):
         v = v + 0.5*a*self.dt
         p_i = np.random.rand()
         nparticles = len(x)
         if p_i < self.freq * self.dt:
              i_ran = np.random.randint(0,nparticles)
              v [i_ran, :] = np.random.normal(0.,np.sqrt(self.temperature),size = 3)
         return x, v




