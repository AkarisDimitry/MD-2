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
import matplotlib.lines as mlines
import main_display

particle_number = 50
np.random.seed( 135 )
x = np.random.rand(particle_number, 3) * 15+30 
x1 = np.copy(x)
x2 = np.copy(x)
x3 = np.copy(x)

x.astype( dtype = np.float32)
x1.astype( dtype = np.float32)
x2.astype( dtype = np.float32)
x3.astype( dtype = np.float32)

sistema00 = Base(particle_number, x=x)
sistema01 = Base(particle_number, x=x1)
sistema02 = Base(particle_number, x=x2)
sistema03 = Base(particle_number, x=x3)

interaction00 = LennardJones( (0,0), 5.0, 1.0, 1.0, 'Displace')
interaction01 = Morse( (0,0), 5.0, 1.0, 1.0, 1.0, 'Displace')

integrado00 = VelVerlet( 0.01 )
    
nh = Neiboarhood( [0,0,0], [100,100,100], 5.0, 5.0) 
box00 = Box( [0,0,0], [100,100,100], 'Fixed' )

steps = 200
var00 = -1

data = np.zeros((particle_number, 3, steps+1, 4 ))
Edata = np.zeros(( steps+1, 4 ))

f = np.zeros_like(x)
f1 = np.zeros_like(x)
f2 = np.zeros_like(x)
f3 = np.zeros_like(x)

while var00 < steps:

        var00 += 1
        t1 = time.clock()
        sistema00.x, sistema00.v = integrado00.first_step(sistema00.x, sistema00.v, f )
        sistema00.x, sistema00.v = box00.wrap_boundary(sistema00.x, sistema00.v)
    
        f,E = interaction01.forces(sistema00.x, sistema00.v, sistema00.t)
        sistema00.x, sistema00.v = integrado00.last_step(sistema00.x, sistema00.v, f)    
        data[:,:,var00, 0] = sistema00.x   
        Edata[var00,0] = E + np.sum(sistema00.v**2)

var00 = -1
while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema01.x, sistema01.v = integrado00.first_step(sistema01.x, sistema01.v, f1 )
        sistema01.x, sistema01.v = box00.wrap_boundary(sistema01.x, sistema01.v)

        f1, E = interaction00.Cforces(sistema01.x, sistema01.v, sistema01.t)
        sistema01.x, sistema01.v = integrado00.last_step(sistema01.x, sistema01.v, f1 )     
        data[:,:,var00, 1] = sistema01.x
        Edata[var00,1] = E + np.sum(sistema01.v**2)


var00 = -1
while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema02.x, sistema02.v = integrado00.first_step(sistema02.x, sistema02.v, f2 )
        sistema02.x, sistema02.v = box00.wrap_boundary(sistema02.x, sistema02.v)
        f2,E = interaction00.forces_neiboarhood(sistema02.x, sistema02.v, sistema02.t,
                                nh.new_neiboarhood(sistema02.x), 5.0, 5.0)
        sistema02.x, sistema02.v = integrado00.last_step(sistema02.x, sistema02.v, f2 )

        data[:,:,var00, 2] = sistema02.x
        Edata[var00,2] = E + np.sum(sistema02.v**2)


var00 = -1
while var00 < steps:
        var00 += 1
        t1 = time.clock()
        sistema03.x, sistema03.v = integrado00.first_step(sistema03.x, sistema03.v, f3 )
        sistema03.x, sistema03.v = box00.wrap_boundary(sistema03.x, sistema03.v)
        f3,E = interaction00.forces(sistema03.x, sistema03.v, sistema03.t )
        sistema03.x, sistema03.v = integrado00.last_step(sistema03.x, sistema03.v, f3 )

        data[:,:,var00, 3] = sistema03.x
        Edata[var00,3] = E + np.sum(sistema03.v**2)

Xdata = np.transpose(data[:,:,:,:], (1,0,2,3) )
main_display.main(Xdata, Edata[:,0])



names = ['Arboles', 'Cforces', 'Lista de vecinos', 'Forces']
color = ( 'r', 'g', 'b', 'k')

plt.figure(1)
plt.xlabel( 'Tiempo del sistema', fontsize=32 )
plt.ylabel( 'Energia total' , fontsize=32)
plt.title( 'Integracion' , fontsize=40)

H = []
for i, n in enumerate( [ '0.05' , '0.02' , '0.01'] ):
    H.append( mlines.Line2D( [], [], color=color[i], label=n ))
plt.legend(handles=H, loc='upper left', shadow=False, fontsize=22)

for n in range(2):
    plt.plot(np.linspace(0,1,78), Edata[0][:78,n,2], color[0])
    plt.plot(np.linspace(0,1,150), Edata[1][:150,n,2], color[1])
    plt.plot(np.linspace(0,1,398), Edata[2][:398,n,2], color[2])

plt.show()

H = []
for i, n in enumerate(names):
    H.append( mlines.Line2D( [], [], color=color[i], label=n ))
plt.legend(handles=H, loc='upper left', shadow=False, fontsize=22)
for i in range(data.shape[1]):  plt.plot(data[:,i] , color=color[i])

plt.figure(2)
plt.xlabel( 'Numero de Step', fontsize=32 )
plt.ylabel( 'Energia potencial' , fontsize=32)
plt.title( ' Shift ' , fontsize=40)
H = []
for i, n in enumerate(names):
   H.append( mlines.Line2D( [], [], color=color[i], label=n ))
   plt.legend(handles=H, loc='upper left', shadow=False, fontsize=22)
for i, n in enumerate(color): plt.plot(Edata[:,i,0], n)
    
plt.figure(3)
plt.xlabel( 'Numero de Step', fontsize=32 )
plt.ylabel( 'Energia cinetica ' , fontsize=32)
plt.title( 'Shift' , fontsize=40)
H = []
for i, n in enumerate(names):
   H.append( mlines.Line2D( [], [], color=color[i], label=n ))
   plt.legend(handles=H, loc='upper left', shadow=False, fontsize=22)
for i, n in enumerate(color): plt.plot(Edata[:,i,1], n)

plt.figure(4)
plt.xlabel( 'Numero de step', fontsize=32 )
plt.ylabel( 'Energia total' , fontsize=32)
plt.title( 'Shift' , fontsize=40)
H = []
for i, n in enumerate(names):
   H.append( mlines.Line2D( [], [], color=color[i], label=n ))
   plt.legend(handles=H, loc='upper left', shadow=False, fontsize=22)
for i, n in enumerate(color): plt.plot(Edata[:,i,2], n)

plt.show()

