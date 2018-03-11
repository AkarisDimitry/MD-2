# -*- coding: utf-8 -*-
from particles.Particles import *
from interaction.Interaction import * 
from integrator.Integrator import *
from box.Box import *

import time
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(200, 3) * 10 + 50
x1 = np.copy(x)
x2 = np.copy(x)

x.astype( dtype = np.float32)
x1.astype( dtype = np.float32)
x2.astype( dtype = np.float32)

sistema00 = Base(200, x=x)
sistema01 = Base(200, x=x1)
sistema02 = Base(200, x=x2)
interaction00 = LennardJones( (0,0), 5.5, 1.0, 1.0, 'Displace')
integrado00 = VelVerlet( 0.01)

fac1, fac2 = 2, 4

integrado01 = VelVerlet( 0.01/fac1)
integrado02 = VelVerlet( 0.01/fac2)

box00 = Box( [0,0,0], [100,100,100], 'Periodic' )

data = np.zeros((1000,1))
var00 = -1
f = np.zeros_like(x)
f1 = np.zeros_like(x)
f2 = np.zeros_like(x)

while var00 < 999:
    var00 += 1; print var00
    t1 = time.clock()
    sistema00.x, sistema00.v = integrado00.first_step(sistema00.x, sistema00.v, f )
    print 'Primer integrador', time.clock()-t1
    
    t1 = time.clock()
    f,E = interaction00.forces(sistema00.x, sistema00.v, sistema00.t)
    print 'Forces', time.clock()-t1

    t1 = time.clock()
    sistema00.x, sistema00.v = integrado00.last_step(sistema00.x, sistema00.v, f)    
    print 'Integrador', time.clock()-t1
    t1 = time.clock()
    sistema00.x, sistema00.v = box00.wrap_boundary(sistema00.x, sistema00.v)
    print 'Box', time.clock()-t1

    #data[var00,0] = np.sum(E) + (np.sum(sistema00.v**2)/2)

data1 = np.zeros((100*fac1,1))
var00 = -1
while var00 < 999*fac1:
    var00 += 1; print var00

    sistema01.x, sistema01.v = integrado01.first_step(sistema01.x, sistema01.v, f1 )
    f1,E1 = interaction00.forces(sistema01.x, sistema01.v, sistema01.t)
    sistema01.x, sistema01.v = integrado01.last_step(sistema01.x, sistema01.v, f1)
    sistema01.x, sistema01.v = box00.wrap_boundary(sistema01.x, sistema01.v)

    data1[var00] = np.sum(E1) + np.sum(sistema01.v**2)/2

data2 = np.zeros((1000*fac2,1))
var00 = -1
while var00 < 999*fac2:
    var00 += 1; print var00
    sistema02.x, sistema02.v = integrado02.first_step(sistema02.x, sistema02.v, f2 )
    f2,E2 = interaction00.forces(sistema02.x, sistema02.v, sistema02.t)
    sistema02.x, sistema02.v = integrado02.last_step(sistema02.x, sistema02.v, f2)
    sistema02.x, sistema02.v = box00.wrap_boundary(sistema02.x, sistema02.v)
    
    data2[var00] =  np.sum(E2) + np.sum(sistema02.v**2)/2


plt.plot(np.linspace(0, 1000, 1000), data)
plt.plot(np.linspace(0, 1000, 1000*fac1), data1)
plt.plot(np.linspace(0, 1000, 1000*fac2), data2)
plt.show()




