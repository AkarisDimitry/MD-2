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

particle_number = 100
np.random.seed( 132 )
x = np.random.rand(particle_number, 3) * 20+30 
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
box00 = Box( [0,0,0], [100,100,100], 'Fixed' )

steps = 1000
var00 = -1
f = np.zeros_like(x)
f1 = np.zeros_like(x)
f2 = np.zeros_like(x)
data = np.zeros( (steps+1, 2) )

while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema00.x, sistema00.v = integrado00.first_step(sistema00.x, sistema00.v, f )
        sistema00.x, sistema00.v = box00.wrap_boundary(sistema00.x, sistema00.v)
    
        f,E = interaction00.force_tree(sistema00.x, sistema00.v, sistema00.t, sklearn.neighbors.KDTree(sistema00.x, leaf_size=3), 5.0)
        sistema00.x, sistema00.v = integrado00.last_step(sistema00.x, sistema00.v, f)    
        print time.clock()-t1 
        data[var00, 0] = E

var00 = -1
while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema01.x, sistema01.v = integrado00.first_step(sistema01.x, sistema01.v, f1 )
        sistema01.x, sistema01.v = box00.wrap_boundary(sistema01.x, sistema01.v)

        f1, E = interaction00.Cforces(sistema01.x, sistema01.v, sistema01.t)
        #print 'E', E
        #print 'f', f1
        sistema01.x, sistema01.v = integrado00.last_step(sistema01.x, sistema01.v, f1)
        print time.clock() - t1 
        data[var00,1] = E
    
plt.plot(data)
plt.show()

