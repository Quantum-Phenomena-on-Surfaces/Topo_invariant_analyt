#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:59:44 2021

@author: cristina
"""

import numpy as np

pi = np.pi
sqrt = np.sqrt
exp = np.exp
cos = np.cos
sin = np.sin

def Self_Energy(J, S, theta, phi, U, lamda, k, a):
    
    Self = np.zeros([4, 4], dtype=complex)
    
    ###magnetic impurities
    Self [0, 0]= J*S*cos(theta) - U
    Self [1, 1]= - J*S*cos(theta) - U
    Self [2, 2]= - J*S*cos(theta) + U
    Self [3, 3]= J*S*cos(theta) + U
         
    Self [0, 1]= J*S*sin(theta)*exp(-1j*phi)
    Self [1, 0]= J*S*sin(theta)*exp(1j*phi)
    Self [2, 3]= - J*S*sin(theta)*exp(1j*phi)
    Self [3, 2]= - J*S*sin(theta)*exp(-1j*phi)
    
    
#    ####Rashba coupling
#    Self [0, 1] = Self [0,1] + lamda*np.exp(1j*k*a) - lamda*np.exp(-1j*k*a)
#    Self [1, 0] = Self [1,0] - lamda*np.exp(1j*k*a) + lamda*np.exp(-1j*k*a)
#    Self [2, 3] = Self [2,3] - lamda*np.exp(1j*k*a) + lamda*np.exp(-1j*k*a)
#    Self [3, 2] = Self [3,2] + lamda*np.exp(1j*k*a) - lamda*np.exp(-1j*k*a)   
    
    ####Rashba coupling
    Self [0,1] = Self [0,1] + 2.0*1j*lamda* sin(k*a)
    Self [1,0] = Self [1,0] - 2.0*1j*lamda* sin(k*a)
    Self [2,3] = Self [2,3] - 2.0*1j*lamda* sin(k*a)
    Self [3,2] = Self [3,2] + 2.0*1j*lamda* sin(k*a)
    
    return(Self)