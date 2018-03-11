# -*- coding: utf-8 -*-
from particles.Particles import *
from interaction.Interaction import * 
from integrator.Integrator import *
from box.Box import *
from neiboarhood.Neiboarhood import *

import sklearn.neighbors

import time, pygame  
import matplotlib.pyplot as plt
import numpy as np
import main_display

particle_number = 1000
np.random.seed( 25 )
x = np.random.rand(particle_number, 3) * 40 + 50
x1 = np.copy(x)
x2 = np.copy(x)

x.astype( dtype = np.float32)
x1.astype( dtype = np.float32)
x2.astype( dtype = np.float32)

sistema00 = Base(particle_number, x=x)
sistema01 = Base(particle_number, x=x1)
sistema02 = Base(particle_number, x=x2)

interaction00 = LennardJones( (0,0), 5.0, 1.0, 1.0, 'Displace')
integrado00 = VelVerlet( 0.01)

nh = Neiboarhood( [0,0,0], [100,100,100], 5.0, 5.0) 

fac1, fac2 = 2, 4

integrado01 = VelVerlet( 0.01/fac1)
integrado02 = VelVerlet( 0.01/fac2)

box00 = Box( [0,0,0], [100,100,100], 'Periodic' )

steps = 100
data = np.zeros((steps+1,3))
Fdata = np.zeros((steps+1,particle_number,3))

EPdata = np.zeros((steps+1,3))
ECdata = np.zeros((steps+1,3))
Xdata = np.zeros((3, particle_number, steps+1))

var00 = -1
f = np.zeros_like(x)
f1 = np.zeros_like(x)
f2 = np.zeros_like(x)

while var00 < steps:
    var00 += 1#; print var00

    sistema00.x, sistema00.v = integrado00.first_step(sistema00.x, sistema00.v, f )
    t1 = time.clock()
    sistema00.x, sistema00.v = box00.wrap_boundary(sistema00.x, sistema00.v)
    
    f,E = interaction00.Cforces(sistema00.x, sistema00.v, sistema00.t)

    #f,E = interaction00.forces_neiboarhood(sistema00.x, sistema00.v, sistema00.t, 
    #        nh.new_neiboarhood( sistema00.x ), 5.0, 5.0 )
    #print time.clock() - t1 
    sistema00.x, sistema00.v = integrado00.last_step(sistema00.x, sistema00.v, f)    
    sistema00.x, sistema00.v = box00.wrap_boundary(sistema00.x, sistema00.v)
    #print nh.l
    Xdata[:,:,var00] = sistema00.x.T
    data[var00,0] = np.sum(E) + (np.sum(sistema00.v**2)/2)
    Fdata[var00,:,0 ] = np.sum(f, 1)
    EPdata[var00,0] = np.sum(E)
    ECdata[var00,0] = (np.sum(sistema00.v**2)/2)
    print var00
    #print data[var00,0]
main_display.main(Xdata, data[:, 0])

var00 = -1
while var00 < steps:
        var00 += 1; print var00

        sistema01.x, sistema01.v = integrado00.first_step(sistema01.x, sistema01.v, f1 )
        t1 = time.clock()
        f1,E = interaction00.Cforces(sistema01.x, sistema01.v, sistema01.t)
        print time.clock() - t1
        sistema01.x, sistema01.v = integrado00.last_step(sistema01.x, sistema01.v, f1)
        #sistema01.x, sistema01.v = box00.wrap_boundary(sistema01.x, sistema01.v)

        data[var00,1] = np.sum(E) + (np.sum(sistema01.v**2)/2)
        Fdata[var00,:, 1] = np.sum(f1, 1)
        EPdata[var00,1] = np.sum(E)
        ECdata[var00,1] = (np.sum(sistema01.v**2)/2)


var00 = -1
while var00 < steps:
    var00 += 1; print var00
    sistema02.x, sistema02.v = integrado00.first_step(sistema02.x, sistema02.v, f2 )
    t1 = time.clock()

    f2,E = interaction00.force_tree(sistema02.x, sistema02.v, sistema02.t, sklearn.neighbors.KDTree(sistema02.x, leaf_size=3), 5.0)
    print time.clock() - t1
    sistema02.x, sistema02.v = integrado00.last_step(sistema02.x, sistema02.v, f2)
    #sistema02.x, sistema02.v = box00.wrap_boundary(sistema02.x, sistema02.v)

    data[var00,2] = np.sum(E) + (np.sum(sistema02.v**2)/2)
    Fdata[var00,:, 2] = np.sum(f2, 1)

plt.figure(10) 
plt.plot(EPdata)
plt.figure(11)
plt.plot(ECdata)

plt.figure(1)
plt.plot(data[:, (0,1)])

plt.figure(2)
plt.plot(Fdata[:,:,0].T)
plt.figure(3)
plt.plot(Fdata[:,:,1].T)

plt.show()

data1 = np.zeros((1000*fac1,1))
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




