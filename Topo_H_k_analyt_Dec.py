# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 09:53:28 2021

@author: crisl
"""
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg') ####to run in Oberon/other clusters
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = [8, 8]
from numpy.linalg import inv

pi=np.pi


def find_nearest(array, value):
    #####calculates the closest but not bigger value in a given array
    diff = array - value
    
    if (value>0):
        i = np.where(diff<=0)
        idx = (abs(diff[i])).argmin()
        return idx
    
    elif(value<0):
        i = np.where(diff>=0)
        idx = (abs(diff[i])).argmin()
        return i[idx][0]        


'''load parameters'''
#energy is given in eV
import json
with open("parameters.json", "r") as f:
    param = json.load(f)

lattice_param = param["lattice_param"]
k_F = param["k_F"]
K = param["K"]
J = param["J"]
theta = param["theta"]
phi = param["phi"]
DOS = param["DOS"]
s = param["s"]
delta = param["delta"]
dynes = param["Dynes"]
alpha = param["alpha"]
mass_eff = param["mass_eff"]

'''omega and k-vectors'''
#####convert to atomic units
d = 1.0#distance between sites MUST BE 1.0!!!!!!
a_interatomic = d*lattice_param/0.529
K = K/27.2116#%potential scatt
j = J/27.2116#coupling
delta =delta/27.2116 #SC gap
Dynes = dynes/27.2116
lamda = (alpha/(2*a_interatomic*0.529))/27.2116 

##########k vector
Nk = param["Nk"]
k = np.linspace(-np.pi, np.pi, Nk)/a_interatomic
np.savetxt('k_vector.txt', k)

i1 = find_nearest(k, -k_F)
i2 = find_nearest(k, k_F)     

#########omega vector
N_omega = param["N_omega"]
range_omega = param["range_omega"]
N_delta = range_omega
    
Romega = np.zeros([N_omega])
Romega=np.array(Romega, np.longdouble)
step_omega=N_delta*delta/(N_omega-1)

for i_omega in range(N_omega):
    Romega[i_omega] = (-N_delta/2.*delta+(i_omega)*step_omega)
         
Romega = np.array(Romega)
vv=Romega*27211.6#####meV
np.savetxt('vv.txt', vv)

'''G(k, omega) and H(k) calculation'''
import Free_Green_k as gk
import Self_Energy_k as sk

Gk_0 = np.zeros([4, 4], dtype=complex)
H_k = np.zeros([4, 4, Nk], dtype=complex) 
omega_0 = int(N_omega/2.0)

for i_k in range(Nk):
    for i_omega in range(N_omega):
            
    
        #####G0(k)        
        Go = gk.Free_Green(Romega[i_omega], Dynes, k_F, mass_eff, DOS, delta, a_interatomic, k[i_k])
        
        #####Self energy(k)
        Self = sk.Self_Energy(j, s, theta, phi, K, lamda, k[i_k], a_interatomic)        
        
        #T = inv(np.eye(4) - Go@Self)
        #GG = T@Go
        Go_inv = inv(Go)
        Gk = Go_inv - Self                
        Der = (Gk - Gk_0)/step_omega####G_k derivative
        Gk_0 = Gk#####new Gk_0                
                
        Der_inv = inv(Der)
        #G_k_omega[:,:,i_k,i_omega] = inv(Gk)#####Gk total
        
        ######H(k)
        #G_k_omega[:,:,i_k,i_omega] = GG
        if (i_omega == omega_0):
            
            H_k[:,:,i_k] = -Der_inv@Gk_0
            
            
'''H(k) diagonalization'''
from numpy import linalg as LA
diag = LA.eig        

E_total = np.zeros([4, Nk], dtype = complex)
psi_total = np.zeros([4, 4, Nk], dtype = complex)

for i_k in range(Nk):

    #Diagonalize                       
    H_m = np.matrix(H_k[:,:,i_k])    
    (E, psi) = diag(H_m)
    
    #E_order = np.sort(E)
    E_total[:, i_k] = E#*27211.6
    psi_total[:,:, i_k] = psi
    
    
###Plot all E_n versus k
plt.figure(1)
for n in range(4):    
    plt.plot(k, np.real(E_total[n,:]*27211.6), 'C2.', ms = 1.0, label = 'H(k)')
            
            
plt.xlabel(r'$k$')
plt.ylabel(r'$E$ (meV)')
plt.xlim([-np.pi/a_interatomic,np.pi/a_interatomic]) 
plt.ylim([-5.,5.])   
#plt.xlim([-k_F, k_F])   
#plt.title('kF=0.7, J=1.5eV, U=5.5eV alpha=3.0eVA')
#plt.savefig('bands.png', dpi = 260, bbox_inches='tight')


'''Winding number'''
d_x_old = 0.0###initialized
d_y_old = 0.0
step_k = k[1] - k[0] 
winding_int = np.zeros(Nk, dtype = float)

d_x_k = np.zeros(Nk, dtype = float)
d_y_k = np.zeros(Nk, dtype = float)

A_k = np.zeros([Nk], dtype = complex)
NW = 0.0###initialized


for i_k in range(Nk):   
    H_ki = H_k[:,:,i_k]
    
    AA = (H_ki[0,0] +  H_ki[0,2]) * (H_ki[1,1] +  H_ki[1,3]) \
    - (H_ki[0,1] +  H_ki[0,3]) * (H_ki[1,0] +  H_ki[1,2])
         
    A_k[i_k] = AA
         
    d_x = np.real(AA)
    d_y = -np.imag(AA)  
    
    ###normalize    
    d_m=np.sqrt(d_x**2+d_y**2)
    d_x = d_x/d_m
    d_y = d_y/d_m
    
    d_x_k[i_k] = d_x
    d_y_k[i_k] = d_y
        
    Der_x = (d_x-d_x_old)/step_k#dx derivative
    Der_y = (d_y-d_y_old)/step_k#dy derivative
    d_x_old = d_x
    d_y_old = d_y
        
    if (i_k == 0):
        NW = 0.5*step_k*(d_x*Der_y-d_y*Der_x)
        #pfa1 = d_x
        #print('pfa -pi/a', pfa1)
        
    elif (i_k < Nk):
        NW = NW+step_k*(d_x*Der_y-d_y*Der_x)
        
    #        if(i_k == int(Nk/2)):
    #            #pfa2 = d_x
    #            #print('pfa 0.0', pfa2)
            
    else:            
        NW = NW + 0.5*step_k*(d_x*Der_y-d_y*Der_x) 
        
        
    winding_int[i_k] = NW
        
    
        
if(k_F < k[-1]):
    winding_i = winding_int[i1:i2+1]
    min_W = min(winding_i)
    max_W = max(winding_i)
    WW = np.sign(NW)*(max_W - min_W)/(2*np.pi)
            
else:
    WW = 1.0/(2*pi)*NW
        
''''Q topological invariant'''

if (k_F < k[-1]):  
    I = np.where(k < -k_F + 0.05)
    II = np.where(k > -k_F)
    III = list(set(I[0]) & set(II[0]))
    max_dx = max(d_x_k[III])

    ####select pfa1 around kF
    pfa1 = max_dx            
    pfa2 = d_x_k[int(Nk/2)]      

else:
            
    pfa1 = d_x_k[0] 
    pfa2 = d_x_k[int(Nk/2)]              
            
Q = np.sign(pfa1*pfa2)
        

print('Winding number', WW)
print('Topo inv', Q)  

 
'''Plots'''
theta = np.linspace(2*np.pi, 0, Nk)
x1 = np.cos(theta)
x2 = np.sin(theta)

norm = plt.Normalize()
color = plt.cm.cool(np.linspace(0,1.0,len(d_x_k)))
plt.rcParams['image.cmap'] = 'cool'

#%%

plt.figure(2)
plt.title('Winding vector')
plt.plot(x1, x2, '--k')
plt.axhline(y=0.0, color='k', linestyle='--')
plt.axvline(x=0.0, color='k', linestyle='--')
plt.scatter(d_x_k, d_y_k, facecolors='none', edgecolors = color)
plt.plot(d_x_k[0], d_y_k[0], 'go', label = 'First')
plt.plot(d_x_k[-1], d_y_k[-1], 'ro', label = 'Last')

plt.xlabel('Real det(A)')
plt.ylabel('Imag det(A)')
plt.legend()
#plt.show()

fig= plt.figure(2)
cbaxes = fig.add_axes([0.9, 0.12, 0.015, 0.2] )
cbar = plt.colorbar(cax=cbaxes, cmap = 'cool')
cbar.set_ticks([0,0.5,1])
cbar.set_ticklabels(['-pi/a', 0.0, 'pi/a'])

#plt.savefig('winding_trayectory.png', dpi = 260, bbox_inches='tight')
#np.savetxt('d_x.txt', d_x_k)
#np.savetxt('d_y.txt', d_y_k)


plt.figure(3) 
plt.plot(k, winding_int/(2*pi), label = 'winding')
plt.plot(k, d_x_k, markersize=0.75, label = 'd_x')
plt.plot(k[i1-1], d_x_k[i1-1], 'r*', markersize=7.0)
plt.plot(k, d_y_k, markersize=0.75, label = 'd_y')
plt.plot(k[i1-1], d_y_k[i1-1], 'r*', markersize=7.0)
plt.grid(True, color='0.95')
plt.show()
plt.title('Winding number')
plt.xlabel('k')
#plt.ylabel('Winding number')
plt.legend()            
            



#####save data for plot
bands = np.real(E_total*27211.6)
np.savetxt('bands.txt', bands)
#np.savetxt('d_x_k.txt', d_x_k)
#np.savetxt('d_y_k.txt', d_y_k)
#np.savetxt('winding_vector.txt', winding_int)