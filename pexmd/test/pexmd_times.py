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

data = np.asarray([
        np.asarray([0.001079+ 0.000841+ 0.000715, 0.000766+ 0.000719+ 0.000704, 0.001688+ 0.001977+ 0.002271,  ]),
        np.asarray([0.001494+ 0.001406+ 0.001479, 0.002662+ 0.002718+ 0.003966, 0.002171+ 0.002394+ 0.002137,]),
        np.asarray([0.005682+ 0.004089+ 0.00404, 0.018876+ 0.018142+ 0.018944, 0.005581+ 0.00650+ 0.05541,   ]),
        np.asarray([0.011949+ 0.013144+ 0.012867, 0.06977+ 0.070662+ 0.072008, 0.031387+ 0.031422+ 0.030479,  ]),
        np.asarray([0.03406 +0.033602+ 0.033303, 0.285598+ 0.280661+ 0.282206, 0.277566+ 0.29598+ 0.271731,   ]),
        np.asarray([0.14891+ 0.236392+ 0.241095, 1.754241+ 1.855012+ 1.858517, 8.435416+ 6.901241+ 6.559585,  ]),
        np.asarray([0.512198+ 0.76701+ 0.772287, 7.745698+ 7.38803+ 7.727984, 126.57535+ 92.389624+ 83.014639,]),
        np.asarray([1.111542+18.072533+18.155011,16.158342+33.216667+33.059187,642.364227+404.066644+542.046148,])])

#plt.plot(data)
#plt.show()

data = np.zeros((10, 3))
for i, n in enumerate((10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 5000)):
    print n
    particle_number = n
    np.random.seed( 2132 )
    x = np.random.rand(particle_number, 3) * 20 + 50
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

    steps = 2

    var00 = -1
    f = np.zeros_like(x)
    f1 = np.zeros_like(x)
    f2 = np.zeros_like(x)
    while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema00.x, sistema00.v = integrado00.first_step(sistema00.x, sistema00.v, f )
        sistema00.x, sistema00.v = box00.wrap_boundary(sistema00.x, sistema00.v)
    
        f,E = interaction00.force_tree(sistema00.x, sistema00.v, sistema00.t, sklearn.neighbors.KDTree(sistema00.x, leaf_size=3), 5.0)
        sistema00.x, sistema00.v = integrado00.last_step(sistema00.x, sistema00.v, f)    
        print time.clock()-t1 
        data[i, 2] = data[i, 2] + time.clock()-t1

    var00 = -1
    while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema01.x, sistema01.v = integrado00.first_step(sistema01.x, sistema01.v, f1 )
        sistema01.x, sistema01.v = box00.wrap_boundary(sistema01.x, sistema01.v)

        f1,E = interaction00.Cforces(sistema01.x, sistema01.v, sistema01.t)
        sistema01.x, sistema01.v = integrado00.last_step(sistema01.x, sistema01.v, f1)
        print time.clock() - t1 
        data[i,1] = data[i, 1] + time.clock()-t1
    
    var00 = -1
    while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema02.x, sistema02.v = integrado00.first_step(sistema02.x, sistema02.v, f2 )
        sistema02.x, sistema02.v = box00.wrap_boundary(sistema02.x, sistema02.v)

        f2,E = interaction00.forces_neiboarhood(sistema02.x, sistema02.v, sistema02.t, 
                nh.new_neiboarhood(sistema02.x), 5.0, 5.0)
        sistema02.x, sistema02.v = integrado00.last_step(sistema02.x, sistema02.v, f2)

        print time.clock() - t1
        data[i,2] = data[i,2] + time.clock() - t1

plt.figure(1) 
plt.plot(data)
plt.show()

