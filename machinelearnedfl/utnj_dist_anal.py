import numpy as np
from numba import njit

#def definebin(jmax,sorted_array):
#   deltarend = np.zeros(jmax+1)
#   deltarbin = np.zeros(jmax)
#   nbin = len(sorted_array)/jmax
#   for i in range(1,jmax):
#      deltarend[i] = sorted_array[nbin*(i)]
#   deltarend[jmax] = 5.0
#   deltarbin = np.diff(deltarend)
#   return deltarend, deltarbin

@njit(target='cpu')
def definebin(jmax, positive):
   deltarend = [0.0]
   nbin = int(len(positive)/jmax)
   mask = (positive[nbin::nbin] < 10.0 + 1e-6) & (positive[nbin::nbin] > positive[nbin-1:-1:nbin])
   deltarend.extend(positive[nbin::nbin][mask])
   deltarend.append(10.0)
   jmaxnew = len(deltarend) - 1
   # print indices of where mask is True
   # mask = (np.where(mask)[0] + 1).tolist()
   # mask.append(jmaxnew)
   # print(mask)
   # del mask
   return deltarend, np.diff(np.array(deltarend)), jmaxnew


@njit(target='cpu')
def phiPWNONequalbin(b,xj,xltr,deltarbin,jmax,phi0):
   # for numba
   b = b.ravel()
   xltr = xltr.ravel()
   deltarbin = deltarbin.ravel()
   # calculate phi
   phi = 0.0
   if xj <= 0.0:
      phi = 0.0;
   elif xj > 0.0 + 1.0e-16 and xj <= xltr[jmax] + 1.0e-16:
      if xj <= xltr[1]:
         phi = b[0] * (xj - xltr[0]) + phi0
         if phi < 0.0:
            phi = 0.0
      else:
         for j in range(2,jmax+1):
            if xj <= xltr[j] and xj > xltr[j-2]:
               phi = (b[:j-2] * deltarbin[:j-2]).sum()
               phi = phi + b[j-2] * (xj - xltr[j-2]) + phi0
               if phi < 0.0:
                  phi = 0.0
   else:
      for j in range(jmax):
         phi = phi + b[j] * deltarbin[j] + phi0
         if phi < 0.0:
            phi = 0.0
   return phi


@njit(target='cpu')
def deltarfuncNONequalbin(rPM,jmax,deltarend,deltarbin): #deltar vector defined in the analytical form, which is also the derivative of o wrt b
   deltar = np.zeros(jmax)
   if 0.0 < rPM < deltarend[jmax] + 1.e-14:
      if rPM < deltarend[1] + 1.e-14:
         deltar[0] = rPM - deltarend[0]
      else:
         deltar[0] = deltarbin[0]
       
      for j in range(1,jmax):
         if deltarend[j+1] + 1.e-14 >= rPM > deltarend[j] + 1.e-14:
            deltar[j] = rPM - deltarend[j]   
            deltar[:j] = deltarbin[:j]
   elif rPM >= deltarend[jmax] + 1.e-14:
      deltar = deltarbin 
   return deltar


@njit(target='cpu')
def limiterJacobianPWNONequalbin(dt,dx,nu,u,u_): #(dt,dx,n,nu,limiter,idx2,u,u_):
   max_man = 0.6 
   # for numba
   _n12 = np.array([0.,-1.,1.,0.,0.])
   _n23 = np.array([0.,0.,-1.,1.,0.])
   _n34 = np.array([0.,0.,0.,-1.,1.])
   _n13 = np.array([0.,-1.,0.,1.,0.])
   _1n223 = np.array([0.,1.,-2.,1.,0.])
   _0n224 = np.array([1.,0.,-2.,0.,1.])
   _12 = np.array([0.,1.,1.,0.,0.])
   _23 = np.array([0.,0.,1.,1.,0.])
   _0n1n23 = np.array([1.,-1.,-1.,1.,0.])
   _1n2n34 = np.array([0.,1.,-1.,-1.,1.])
   # get rP, rM
   if np.sign(u[2]) > 0:
      rP = np.dot(_n12, u) / np.dot(_n23, u + 1e-16)
   else:
      rP = np.dot(_n34, u) / np.dot(_n23, u + 1e-16)
   if np.sign(u_[2]) > 0:
      rM = np.dot(_n12, u_) / np.dot(_n23, u_ + 1e-14)
   else:
      rM = np.dot(_n34, u_) / np.dot(_n23, u_ + 1e-14)

   # minimized division
   nu_dx = nu/dx
   max_dx2dt = max_man*(0.5*dx/dt)
   dt8dx = -0.125*dt/dx
   # get deltaF
   DeltaF_Part1 = 0.25 * ( np.dot(_n13, u**2.) - nu_dx * np.dot(_0n224, u) ) - max_dx2dt * np.dot(_1n223, u)

   DeltaF_Part2 = ( dt8dx * np.dot(_23, u) * (np.dot(_n23, u**2.) - (nu_dx) * np.dot(_1n2n34, u) ) + max_dx2dt * np.dot(_n23, u) )

   DeltaF_Part3 = (-1) * ( dt8dx * np.dot(_12, u) * ( np.dot(_n12, u**2.) - (nu_dx) * np.dot(_0n1n23, u) ) + max_dx2dt * np.dot(_n12, u) )

   #deltatnj = DeltaF_Part1 + 1.0 * DeltaF_Part2 + \
   #                          1.0 * DeltaF_Part3
   #print(deltatnj)
   return DeltaF_Part1, DeltaF_Part2, DeltaF_Part3, rP, rM
#
