# ==========
# Created by Ivan Gadjev
# 2020.02.01
#
# Library of custom functions that aid in projects using pyRadia, genesis, elegant, and varius other codes and analyses. 
#   
# 
# ==========

import sys
import os
import time

import scipy.constants as pc
import numpy as np
import pandas as pd

import scipy.signal as sps

import matplotlib.pyplot as plt
import matplotlib.cm as mcm

#
# ===== global defs =====
#

xhat = np.array([1,0,0])
yhat = np.array([0,1,0])
zhat = np.array([0,0,1])
origin = np.array([0,0,0])

colordict = {'tblue':[0.12156862745098039, 0.4666666666666667, 0.7058823529411765],
            'tred':[0.8392156862745098, 0.15294117647058825, 0.1568627450980392],
            'tgreen':[0.17254901960784313, 0.6274509803921569, 0.17254901960784313],
            'torange':[1.0, 0.4980392156862745, 0.054901960784313725] }

#
# ===== helper functions =====
#

def rot_mat(angle, dim=3, axis='z'):
    """ Create a rotation matrix for rotation around the specified axis.
    """
    # sine cosine
    c = np.cos(angle)
    s = np.sin(angle)

    if dim == 3:
        if axis == 'x':
            rr = np.array( [[1, 0, 0], 
                            [0, c, -s], 
                            [0, s, c]] )
        elif axis == 'y':
            rr = np.array( [[c, 0, s], 
                            [0, 1, 0], 
                            [-s, 0, c]] )
        elif axis == 'z':
            rr = np.array( [[c, -s, 0], 
                            [s, c, 0], 
                            [0, 0, 1]] )
    elif dim == 2:
        rr = np.array( [[c, -s], 
                        [s, c]] )
    else:
        print(' error: `dim` variable must be 2 or 3. Specifies the dimension of the vectors that the rotation is acting on.')
    return rr

def rgb_color(value, cmapname='viridis'):
    """ returns the RGB values of the color from the colormap specified with scaled value.
    value - between 0 and 1
    cmapname='viridis' name of the colormap
    """ 
    cmap = mcm.ScalarMappable(cmap=cmapname)
    return list(cmap.to_rgba(value, bytes=False, norm=False)[:-1])

#
# === Integrate

def numint(xvec, yvec):

    """ Calculates the integral of the function defined by f(xvec) = yvec
    xvec - ndarray 
    yvec - ndarray
    
    """
    # differential element vector
    delxvec = np.abs(xvec[1:] - xvec[:-1])
    delxvec = np.append(delxvec, delxvec[-1]) # make same length
    
    numeric_integral = np.sum( delxvec * yvec )
    return numeric_integral


#
# === beam dynamics

def gammabetarel(totengMeV, restmassMeV):
    """ returns the relativistic factors gamma and beta for a given particle energy (gmc2) and rest mass (mc2)

    gamma = totengMeV / restmassMeV
    beta = sqrt( 1 - 1/g^2)
    """
    gamma = totengMeV / restmassMeV
    beta = np.sqrt( 1 - 1 / gamma**2)

    return gamma, beta

def momrel(totengMeV, restmassMeV):
    """ returns the relativistic momentum (in units of MeV/c) of the particle with the specified total energy (gmc2) and rest mass (mc2) 

    pc = sqrt( E^2 - mc2^2 )

    NB: the output is in units of MeV/c. 
    """
    prel = np.sqrt( totengMeV**2 - restmassMeV**2)
    return prel

def momrigid(momMeV, charge):
    """ Returns the momentum rigidity for a given momentum in MeV and charge in units of elementary charge units, e. 
    
    B * rho = (1e6 / c) * (1 / charge) * pc 
    """
    br = (1e6 / pc.c) * (momMeV / charge)

    return br

def mquad(maglen, gradient, totengMeV, restmassMeV=None, charge=1, focus='focus'):
    """ the 2x2 1st order matrix for quadrupole focusing.

    restmassMeV - default is electron mass ~0.511MeV 
    1/f = Kquad * L
    f - focal length
    Kquad = gradient / momentumrigidity # this is the S.Y.Lee definition (note this is not squared!)
    L - magnetic length
    """
    
    if restmassMeV == None:
        mc2 = 1e-6 * (pc.m_e*pc.c**2)/pc.elementary_charge # MeV. electron mass
        restmassMeV = mc2
    
    mrig = momrigid(momrel(totengMeV,restmassMeV), charge)
    kquad = gradient / mrig # normalized quadrupole gradient
    if focus == 'focus':
        fsign = -1
    else:
        fsign = 1

    fl = fsign / (kquad * maglen)

    mat = np.array( [[ 1, 0 ],
                     [ 1/fl, 1 ]
                     ])
    return mat

def mdrift(dlen):
    """ return the 2x2 matrix for a drift
    """
    mat = np.array( [[ 1, dlen ],
                     [ 0, 1 ]
                     ])
    return mat

def makeMmat(mat):
    """ returns the 3x3 matrix that acts on [beta,alpha,gamma] from the 2x2 matrix for [x,x']
    mat - ndarray with shape [2,2]
    """
    m11 = mat[0,0]
    m12 = mat[0,1]
    m21 = mat[1,0]
    m22 = mat[1,1]

    mat3 = np.array( [[ m11**2, -2*m11*m12, m12**2 ],
                      [ -m11*m21, m11*m22 + m12*m21, -m12*m22 ],
                      [ m21**2, -2*m21*m22, m22**2 ]
                     ])
    return mat3


#
##
###  === Particle Swarm Optimization
##
#

class PSO(object):
    """ Class handles Particle Swarm Optimization (PSO) of the magnet dimensions with a target gradient, strenght, magnetic length, etc.

    Particle Swarm Optimization pseudo code:
    - define seach space, SS
    - init particles inside SS
    - init particle velocity
    - init global best solution, gbest
        - compute cgbest = cost(f(gbest))
    while (termination condition not met):
        - init particle best solution, pbest
            - compute cpbest = cost(f(pbest))
        for particle in particles:
            
            - compute cost
                cparticle = cost(f(particle))
                
            - update gbest and pbest:
                if cpbest > cparticle:
                    pbest = particle
                    cpbest = cost(f(pbest))
                    if cgbest > cpbest:
                        gbest = pbest
                        cgbest = cpbest
                
            - update velocity
                v = v + alpha1*rand(0,1)*(pbest - particle) + alpha2*rand(0,1)*(gbest - particle)
            - update position
                particle = particle + v

    """
    def __init__(self):

        self.phi1 = 2.05
        self.phi2 = 2.05

        self.maxiter = 0
        self.precision = 0

    def cost(self, current, target):
        """ Calculates the total square difference between the current parameter values and the target values.

        current - np.array(n,)
        target - np.array(n,)
        """

        value = np.sqrt( np.sum( (current - target)**2 ))
        
        return value

    def velocity(self, vin, xin, pbest, gbest):
        """

        Updates the input velocity.
        vin - np.array(nparticles, ) input velocity
        xin - np.array(nparticles, ) input position
        pbest - np.array(nparticles, ) the best position for the particle so far
        gbest - np.float the best position for any particle 
        phi1=2.05, phi2=2.05 - regulate the 'randomness' in the velocity update as described below
        
        Clerc and Kennedy (2002) noted that there can be many ways to implement the constriction coefficient. One of the simplest methods of incorporating it is the following :
        v_(i+1) = chi * ( v_i + U(0,phi1) * (p_i - x_i) + U(0,phi2) * (pg - x_i) )
        x_(i+1) = x_i + v_i
        where,
        phi = phi1 + phi2 > 4
        chi = 2 / ( phi - 2 + sqrt(phi^2 - 4*phi) )

        When Clerc's constriction method is used, phi is commonly set to 4.1, phi1=phi2 and the constant multiplier chi is approximately 0.7298. This results in the previous velocity being multiploied by 0.7298 and each of the two (p - x) terms being multiplied by a random number limited by 0.7398*2.05 = 1.49618.
        """
        phi = self.phi1 + self.phi2
        chi = 2 / ( phi - 2 + np.sqrt(phi**2 - 4 * phi) )

        vout = chi * ( vin + self.phi1 * np.random.random(xin.shape) * (pbest - xin) + self.phi2 * np.random.random(xin.shape) * (gbest - xin) )

        return vout

    def run_pso(self, function, searchspace, target, nparticles, maxiter, precision, domain, verbose=True):

        """ Performs a PSO for the given function in the searchspace, looking for the target, which is in the output space.
        
        function - the function to be optimized. Its domain must include the seachspace and its output must be in the space of target.
        searchspace - np.array((ssdim, 2)) 
        target - np.array((tdim, ))
        nparticles - number of particles to use in the optimization
        maxiter - maximum number of iterations to the optimization routine
        precision - how close to the target to attemp to get
        domain - absolute boundaries on the trial solutions/particles
        """
        # update attributes
        self.maxiter = maxiter
        self.precision = precision

        # search space dimensionality
        if searchspace.shape[1] != 2:
            print('WARNING! searchspace does not have dimenstions (N,2).')
        ssdim = searchspace.shape[0]
        
        # init particle positions and velocities
        xpart = np.random.random((nparticles, ssdim))
        for ii in range(ssdim):
            xpart[:,ii] = (searchspace[ii,1] - searchspace[ii,0]) * xpart[:,ii] + searchspace[ii,0] # scale the uniform radnom dist
        
        vpart = np.zeros(xpart.shape)

        # init particle best solution
        pbest = 1.0 * xpart
        cpbest = np.array([ self.cost(function(*xp), target) for xp in pbest ])
        # init global best solutions
        im = np.argmin(cpbest)
        gbest = pbest[im]
        cgbest = cpbest[im]

        if False:
            return xpart, vpart, pbest, cpbest, gbest, cgbest

        # intermediate arrays
        # multiply by 1.0 to make copies not bind references
        xarr = 1.0 * xpart[:,:,None]
        varr = 1.0 * vpart[:,:,None]
        parr = 1.0 * pbest[:,:,None]
        cparr = 1.0 * cpbest[:,None]
        garr = 1.0 * gbest[:,None]
        cgarr = 1.0 * np.array([cgbest])

        iternum = 0

        t1 = time.time()
        while (iternum <= maxiter) and ( cgbest > precision):
        # while (iternum <= maxiter):

            for pp in range(nparticles):
                
                
                # update velocity
                vpart[pp] = self.velocity(vpart[pp], xpart[pp], pbest[pp], gbest)
                # update position
                xpart[pp] = xpart[pp] + vpart[pp]
                
                # keeps particles inside the absolute boundaries given by `domain`
                xpart[pp] = np.maximum(xpart[pp], domain[:,0])
                xpart[pp] = np.minimum(xpart[pp], domain[:,1])

                # compute cost of new position
                cpp = self.cost(function(*xpart[pp]) , target )

                # update best position
                if cpp < cpbest[pp]:
                    pbest[pp] = xpart[pp]
                    cpbest[pp] = cpp
                if cpp < cgbest:
                    gbest = xpart[pp]
                    cgbest = cpp

            xarr = np.concatenate((xarr, xpart[:,:,None]),axis=2)
            varr = np.concatenate((varr, vpart[:,:,None]), axis=2)
            parr = np.concatenate((parr, pbest[:,:,None]), axis=2)
            cparr = np.concatenate((cparr, cpbest[:,None]), axis=1)
            garr = np.concatenate((garr, gbest[:,None]), axis=1)
            cgarr = np.append(cgarr, cgbest)

            iternum += 1

        t2 = time.time()
        if verbose:
            print('optimization took {:5.2f} seconds'.format(*[t2-t1]))

        return xarr, varr, parr, cparr, garr, cgarr


