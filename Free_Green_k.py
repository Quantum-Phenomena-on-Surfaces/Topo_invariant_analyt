#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:06:24 2021

@author: cristina
"""

import numpy as np
import cmath

# Functions
pi = np.pi
sqrt = np.sqrt
exp = np.exp
log = cmath.log
arctan = cmath.atan

#def log2(z):
#    Re = np.real(log(z))
#    Im = arctan(np.imag(z)/np.real(z))
#    t = Re + 1j*Im
#    return(t)

#################################

# def FF(x, a_interatomic):
    
#     t = log( 1 - exp(x*a_interatomic) )
#     return(t)
    
    
def FF1(k, k_F, xi, a_interatomic):
    
    x = -xi + 1j*(k_F + k)    
    t = log( 1 - exp(x*a_interatomic) )
        
    return(t)
    
def FF2(k, k_F, xi, a_interatomic):
    
    x = -xi + 1j*(k_F - k)
    t = log( 1 - exp(x*a_interatomic) )        
    return(t)
    
    
def FF3(k, k_F, xi, a_interatomic):
    
    x = -xi - 1j*(k_F + k)    
    t = log( 1 - exp(x*a_interatomic) )    
        
    return(t)
    
def FF4(k, k_F, xi, a_interatomic):
    
    x = -xi - 1j*(k_F - k)
    t = log( 1 - exp(x*a_interatomic) )

    return(t)
    
##############################
    
#def FF1(k, k_F, xi, a_interatomic):
#    
#    x = -xi + 1j*(k_F + k)
#    
#    if (k < - k_F):
#        t = log( 1 - exp(x*a_interatomic) ) - 1j*np.pi
#        
#    else:
#        t = log( 1 - exp(x*a_interatomic) )
#        
#    return(t)
#    
#def FF2(k, k_F, xi, a_interatomic):
#    
#    x = -xi + 1j*(k_F - k)
#    
#    if (k > k_F):
#        t = log( 1 - exp(x*a_interatomic) ) - 1j*np.pi
#        
#    else:
#        t = log(1 - exp(x*a_interatomic)    )
#    return(t)
#    
#    
#def FF3(k, k_F, xi, a_interatomic):
#    
#    x = -xi - 1j*(k_F + k)
#    
#    if (k < - k_F):
#        t = log( 1 - exp(x*a_interatomic) ) + 1j*np.pi
#        
#    else:
#        t =  log(1 - exp(x*a_interatomic)   )
#        
#    return(t)
#    
#def FF4(k, k_F, xi, a_interatomic):
#    
#    x = -xi - 1j*(k_F - k)
#    
#    if (k > k_F):
#        t = log( 1 - exp(x*a_interatomic) ) + 1j*np.pi
#        
#    else:
#        t = log(1 - exp(x*a_interatomic))
#
#    return(t)

################################3

def Free_Green(lomega, Damping, Fermi_k, mass_eff, DOS_o, Delta, a_interatomic, k):
    
    G_0_k = np.zeros([4,4], dtype = complex)
    
    omega = lomega + 1j * Damping
    
    ####factors
    SS = sqrt(Delta**2 - omega**2)
    xi = (mass_eff * SS)/Fermi_k
    factor = pi * DOS_o / (2 * Fermi_k * a_interatomic)
    factor_0 = -pi*DOS_o
    
    
    '''diagonal in Nambu space'''
    ####G11
    G_0_k[0,0] = factor * ( + FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic) \
                            + FF3(k, Fermi_k, xi, a_interatomic) + FF4(k, Fermi_k, xi, a_interatomic) \
                            
        + omega/(1j*SS) * ( + FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic) \
                            - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic))\
                             ) + factor_0*omega/SS
          
          
    ####G22
    G_0_k[1,1] = factor * ( + FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic) \
                            + FF3(k, Fermi_k, xi, a_interatomic) + FF4(k, Fermi_k, xi, a_interatomic) \
                            
        + omega/(1j*SS) * ( + FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic) \
                            - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic))\
                             ) + factor_0*omega/SS
          
          
    ####G33
    G_0_k[2,2] = factor * ( -(+ FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                              + FF3(k, Fermi_k, xi, a_interatomic) + FF4(k, Fermi_k, xi, a_interatomic))\
                 
        + omega/(1j*SS) * (   + FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                              - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic))
                             ) + factor_0*omega/SS
          
    ####G44
    G_0_k[3,3] = factor * ( -(+ FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                              + FF3(k, Fermi_k, xi, a_interatomic) + FF4(k, Fermi_k, xi, a_interatomic))\
        
        + omega/(1j*SS) * (  + FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                             - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic))
                             ) + factor_0*omega/SS
    
    '''counter-diagonal in Nambu space'''
    ###G14
    G_0_k[0,3] = - factor*Delta/(1j*SS) * ( 
                             FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                           - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic) )\
                           - factor_0*Delta/SS
    
    
    ###G23
    G_0_k[1,2] = + factor*Delta/(1j*SS) * ( 
                             FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                           - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic) )\
                           + factor_0*Delta/SS
    
    ###G32
    G_0_k[2,1] = + factor*Delta/(1j*SS) * ( 
                             FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                           - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic) )\
                           + factor_0*Delta/SS
    
    ###G41
    G_0_k[3,0] = - factor*Delta/(1j*SS) * ( 
                             FF1(k, Fermi_k, xi, a_interatomic) + FF2(k, Fermi_k, xi, a_interatomic)\
                           - FF3(k, Fermi_k, xi, a_interatomic) - FF4(k, Fermi_k, xi, a_interatomic) )\
                           - factor_0*Delta/SS
    
    
    
    return(G_0_k)
    
    
    
    
    
    
    
    
    
    