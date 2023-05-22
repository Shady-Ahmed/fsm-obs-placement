# -*- coding: utf-8 -*-
"""
Codes to reproduce the results for the scalar linear test case in the paper:
    "On the dual advantage of placing observations through forward sensitivity analysis"
    Proceedings of The Royal Society A, 2023.
    Authors: S. E. Ahmed, O. San, S. Lakshmivarahan, and J. M. Lewis
    For questions and/or comments, please feel free to contact me at:
        shady.ahmed@okstate.edu or shady.ahmed@pnnl.gov

Last checked: May 21, 2023
"""

#%% Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
mpl.rc('font', **font)

#%% Define functions

# Right-hand side of model
def rhs(x,a):
    f = a*x**2
    return f
     
# Time Integrator
def Euler(x,a,dt): #1st order Euler Scheme
    f = rhs(x,a)
    xn = x + dt*f
    return xn

def RK4(x,a,dt): #4th order RK Scheme
    k1 = rhs(x,a)
    k2 = rhs(x+k1*dt/2,a)
    k3 = rhs(x+k2*dt/2,a)
    k4 = rhs(x+k3*dt,a)
    xn = x + (dt/6)*(k1+2*k2+2*k3+k4)
    return xn

# Jacobian of right-hand side of model
def Jrhs(x,a):
    df = np.zeros([1,2])
    df[0,0] = 2*a*x
    df[0,1] = x**2
    return df
    
# Compute model Jacobians
def JEuler(x,a,dt): #1st order Euler Scheme
    df = Jrhs(x,a)
    dM = np.eye(1,2) + dt*df
    dMx = dM[:,0]
    dMa = dM[:,1]
    return dMx, dMa
    
def JRK4(x,a,dt): #4th order RK Scheme
    k1 = rhs(x,a)
    k2 = rhs(x+k1*dt/2,a)
    k3 = rhs(x+k2*dt/2,a)
    
    dk1 = Jrhs(x,a)
    dk2 = Jrhs(x+k1*dt/2,a) @ (np.eye(2)+(dt/2)*np.vstack([dk1,np.zeros([1,2])]))
    dk3 = Jrhs(x+k2*dt/2,a) @ (np.eye(2)+(dt/2)*np.vstack([dk2,np.zeros([1,2])]))
    dk4 = Jrhs(x+k3*dt,a) @ (np.eye(2)+(dt)*np.vstack([dk3,np.zeros([1,2])]))

    dM = np.eye(1,2) + (dt/6)*(dk1 + 2*dk2 + 2*dk3 + dk4)
    dMx = dM[:,:1]
    dMa = dM[:,1:]
    return dMx, dMa

# Observational map
def h(x):
    z = x
    return z

# Jacobian of observational map
def Dh(X):
    D = np.eye(1)
    return D
        
#%% Main script

n = 1 #dimension of state x
p = 1 #dimension of parameters
m = 1 #dimension of measurement z
  
tm = 2 #final (maximum) time
dt = 0.01 #timestep
nt = int(tm/dt) #total number of timesteps
t = np.linspace(0,tm,nt+1) #time

#true initial condition and parameter
x0true = 2.0
atrue = -1.0

#observation noise
sig = 0.1 #standard deviation
R = sig**2 * np.eye(m) #covariance matrix
Ri = np.linalg.inv(R)  #inverse of covatiance matrix

#%% Solution based on true control
ctrue = np.array([x0true,atrue]) #true control (initial condition and parameter)

#Time integration
x0 = ctrue[0]
a = ctrue[1]
xtrue = np.zeros([1,nt+1])
xtrue[:,0] = x0
for k in range(nt):
    #time integration using 4th order Runge-Kutta scheme 
    #ther schemes may be useed, e.g., 1st order Euler
    xtrue[:,k+1] = RK4(xtrue[:,k],a,dt)

#%% Twin Experiment to generate measurements
#   you can change t1 and t2 below to see the effect of observation placement
t0 = 0.1 #time of first observation 
t1 = 0.5 #time of second observation
tind = np.array([int(t0/dt),int(t1/dt)]) #time indices at which observations are collected
z = xtrue[:,tind] + np.random.normal(0,sig,xtrue[:,tind].shape)

#%% compute cost function at different values of control 
nvalx = 1001
nvala = 101
x0val = np.linspace(-1,4,nvalx)
aval = np.linspace(-2,0.5,nvala)

Jval = np.zeros([nvalx,nvala])
for i,x0 in enumerate(x0val):
    for j,a in enumerate(aval):
        x1 = x0
        a1 = a
        k = 0
        for n in range(tind[-1]):
            #evolve the state x
            x1 = RK4(x1,a1,dt)
            if n+1 == tind[k]:
                ek = (z[:,k] - h(x1)).reshape(-1,1)
                Jval[i,j] += (z[:,k] - h(x1))**2
                k = k+1
Jval = Jval/(2*sig**2)

#%% Solution based on guessed control

x0guess = 1.75 #some initial guess for the initial condiiton
aguess = -0.75 #some initial guess for the model's  parameter
xguess = np.zeros([1,nt+1])
xguess[:,0] = x0guess
for k in range(nt):
    xguess[:,k+1] = RK4(xguess[:,k],aguess,dt)

#%% Compute model sensitivities based on true control

utrue = np.zeros([1,1,nt+1])
vtrue = np.zeros([1,1,nt+1])
G = np.zeros([2,2,nt+1])

y = np.zeros(nt+1)
w = np.zeros(nt+1)

utrue[0,0,0] = 1
vtrue[0,0,0] = 0
for k in range(nt):
    dMx , dMa = JRK4(xtrue[:,k],atrue,dt)
    utrue[:,:,k+1] = dMx @ utrue[:,:,k]
    vtrue[:,:,k+1] = dMx @ vtrue[:,:,k] + dMa
    
    Hk1 = Dh(xtrue[:,k+1]) @ utrue[:,:,k+1]
    Hk2 = Dh(xtrue[:,k+1]) @ vtrue[:,:,k+1]
    Hk = np.hstack([Hk1,Hk2])
    Gk = Hk.T @ Hk
    G[:,:,k+1] = Hk.T @ Hk
    
#%% Compute model sensitivities based on guessed control

uguess = np.zeros([1,1,nt+1])
vguess = np.zeros([1,1,nt+1])
G = np.zeros([2,2,nt+1])

uguess[0,0,0] = 1
vguess[0,0,0] = 0
for k in range(nt):
    dMx , dMa = JRK4(xguess[:,k],aguess,dt)
    uguess[:,:,k+1] = dMx @ uguess[:,:,k]
    vguess[:,:,k+1] = dMx @ vguess[:,:,k] + dMa
       
    Hk1 = Dh(xguess[:,k+1]) @ uguess[:,:,k+1]
    Hk2 = Dh(xguess[:,k+1]) @ vguess[:,:,k+1]
    Hk = np.hstack([Hk1,Hk2])
    Gk = Hk.T @ Hk
    G[:,:,k+1] = Hk.T @ Hk
    
#%% compute sensitivites with respect to measurement based on guessed control

y1 = np.zeros([nt+1,nt+1])
w1 = np.zeros([nt+1,nt+1])
y2 = np.zeros([nt+1,nt+1])
w2 = np.zeros([nt+1,nt+1])

for k1 in range(nt+1):
    H1 = np.vstack([ Dh(xguess[:,k1])@uguess[:,:,k1], Dh(xguess[:,k1])@vguess[:,:,k1]])
    Gk1 = G[:,:,k1]

    for k2 in range(nt+1):
        H2 = np.vstack([ Dh(xguess[:,k2])@uguess[:,:,k2], Dh(xguess[:,k2])@vguess[:,:,k2]])
        Gk2 = G[:,:,k2]
        Gt = Gk1+Gk2

        yw1 = np.linalg.lstsq(Gt, H1, rcond=None)[0]
        yw2 = np.linalg.lstsq(Gt, H2, rcond=None)[0]

        if np.linalg.det(Gt) <= 1e-10:
            yw1[:] = 100
            yw2[:] = 100
            
        y1[k1,k2] = yw1[0]
        w1[k1,k2] = yw1[1]
        y2[k1,k2] = yw2[0]
        w2[k1,k2] = yw2[1]

#%% Plotting
#%% x plot -- figure 8
plt.figure(figsize=(8,6))
plt.plot(t,xtrue[0],'-',linewidth=2.5,label='True')
plt.plot(t,xguess[0],'--',linewidth=2.5,label='Guess')

plt.xlim([0,2])
plt.ylim([0,2])
plt.xlabel(r'$t$',fontsize=18)
plt.ylabel(r'$x(t)$',fontsize=18)
plt.legend(loc="upper right", ncol=1, fontsize=16)

#%% U and V plots -- figure 9
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(16,6))
ax = ax.flat
ax[0].plot(t,utrue[0,0,:]**2,'-',linewidth=2.5,label='True')
ax[0].plot(t,uguess[0,0,:]**2,'--',linewidth=2.5,label='Guess')

ax[1].plot(t,vtrue[0,0,:]**2,'-',linewidth=2.5,label='True')
ax[1].plot(t,vguess[0,0,:]**2,'--',linewidth=2.5,label='Guess')

ax[0].set_ylabel(r'$u^2$',fontsize=18)
ax[1].set_ylabel(r'$v^2$',fontsize=18)

for i in range(2):
    ax[i].set_xlabel(r'$t$',fontsize=18)
    ax[i].set_xlim([0,2])
    ax[i].legend(loc="upper right", ncol=1, fontsize=16)
    
ax[0].set_ylim([0,1])
ax[1].set_ylim([0,0.5])

#%% cost function contours plot -- figure 10
plt.figure(figsize=(8,6))

[x0grid,agrid] = np.meshgrid(x0val,aval,indexing='ij')
cvalf = np.linspace(0,500,51)
cval = np.array([1,10,25,50,100,150,250,500])

cs = plt.contourf(x0grid,agrid,Jval,cvalf,cmap='YlOrBr',extend='both')
cs.set_clim([0, 500])
CS = plt.contour(x0grid,agrid,Jval,cval,colors='k',linewidths=3)
plt.clabel(CS, colors = 'k', fontsize=16,inline=1,rightside_up=True) #contour line labels
plt.xlabel(r'$x_0$',fontsize=18)
plt.ylabel(r'$a$',fontsize=18)

#%% plots of the sensitivity of the optimal estimates of the initial condition to the first observation -- figure 11

fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,5))
ax = ax.flat

[t1grid,t2grid] = np.meshgrid(t,t,indexing='ij')
cval = np.array([0.5,2.0,5.0,15.0,30.0,50])

cval = np.array([1.0,5.0,20.0,50.0,100,250])

cs = ax[0].contourf(t1grid,t2grid,y1**2,cval,cmap='YlOrBr',extend='both')
cs.set_clim([1, 250])

CS = ax[0].contour(t1grid,t2grid,y1**2,cval,colors='k',linewidths=3)
manual_locations = [(0.8, 0.15), (1.1, 0.35),
                    (1.3,0.4),   (1.4, 0.60), 
                    (1.65,0.85),
                    (1.8, 1.00), (0.2,1.2),
                    (0.5, 1.20), (0.65,1.4),
                    (1.10, 1.50)
                    ]
ax[0].clabel(CS,colors='k',fontsize=18,inline=1,rightside_up=True, manual=manual_locations) #contour line labels
ax[0].set_xlim([0.0,2])
ax[0].set_ylim([0.0,2])
ax[0].set_xlabel(r'$t_1$',fontsize=18)
ax[0].set_ylabel(r'$t_2$',fontsize=18)

ax[1].plot(t,(y1[np.where(t==t0)[0][0],:])**2,'C0',linewidth=3,label=r'$t_1='+str(t0)+'$')
ax[1].set_xlim([0.1,2])
ax[1].set_ylim([1,10])
ax[1].set_xlabel(r'$t_2$',fontsize=18)
ax[1].set_ylabel(r'$y_1^2$',fontsize=18)

ax[2].plot(t,(y1[:,np.where(t==t1)[0][0]])**2,'C0',linewidth=3,label=r'$t_2='+str(t1)+'$')
ax[2].set_xlim([0.05,1])
ax[2].set_ylim([0,250])
ax[2].set_xlabel(r'$t_1$',fontsize=18)
ax[2].set_ylabel(r'$y_1^2$',fontsize=18)

for i in range(1,3):
    ax[i].legend(loc="best", ncol=1, fontsize=16)

fig.subplots_adjust(wspace=0.2)

#%% plots of the sensitivity of the optimal estimates of the model's parameter to the first observation -- figure 12
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,5))
ax = ax.flat

[t1grid,t2grid] = np.meshgrid(t,t,indexing='ij')
cval = np.array([0.2,0.5,2.0,5.0,10.0,20,50.0])

cs = ax[0].contourf(t1grid,t2grid,w1**2,cval,cmap='YlOrBr',extend='both')
cs.set_clim([0, 50])

CS = ax[0].contour(t1grid,t2grid,w1**2,cval,colors='k',linewidths=3)
manual_locations = [(1.45, 0.2), (1.35, 0.45),
                    (1.25, 0.70), (1.15, 0.90),
                    (1.5, 1.5), (1.05,1.45),
                    (0.8, 1.3), (0.6,1.0),
                    (0.2, 0.75),(0.25,1.5),
                    (0.4,0.9) ]

ax[0].clabel(CS,colors='k',fontsize=18,inline=1,rightside_up=True, manual=manual_locations) #contour line labels
ax[0].set_xlim([0.0,2])
ax[0].set_ylim([0.0,2])
ax[0].set_xlabel(r'$t_1$',fontsize=18)
ax[0].set_ylabel(r'$t_2$',fontsize=18)

ax[1].plot(t,(w1[np.where(t==t0)[0][0],:])**2,'C0',linewidth=3,label=r'$t_1='+str(t0)+'$')
ax[1].set_xlim([0.1,2])
ax[1].set_ylim([0,10])
ax[1].set_xlabel(r'$t_2$',fontsize=18)
ax[1].set_ylabel(r'$w_1^2$',fontsize=18)

ax[2].plot(t,(w1[:,np.where(t==t1)[0][0]])**2,'C0',linewidth=3,label=r'$t_2='+str(t1)+'$')
ax[2].set_xlim([0.05,1])
ax[2].set_ylim([0,250])
ax[2].set_xlabel(r'$t_1$',fontsize=18)
ax[2].set_ylabel(r'$w_1^2$',fontsize=18)

for i in range(1,3):
    ax[i].legend(loc="best", ncol=1, fontsize=16)

fig.subplots_adjust(wspace=0.2)

#%% plots of the sensitivity of the optimal estimates of the initial condition to the second observation -- figure 13

fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,5))
ax = ax.flat

[t1grid,t2grid] = np.meshgrid(t,t,indexing='ij')
cval = np.array([0.5,2.0,5.0,15.0,50.0])


cs = ax[0].contourf(t1grid,t2grid,y2**2,cval,cmap='YlOrBr',extend='both')
cs.set_clim([0, 50])

CS = ax[0].contour(t1grid,t2grid,y2**2,cval,colors='k',linewidths=3)
manual_locations = [(0.2, 1.25), (0.5, 1.30), 
                    (0.9, 1.35), (1.0, 0.5),
                    (0.75,0.35), (0.5, 0.1)]

ax[0].clabel(CS,colors='k',fontsize=18,inline=1,rightside_up=True, manual=manual_locations) #contour line labels
ax[0].set_xlim([0.0,2])
ax[0].set_ylim([0.0,2])
ax[0].set_xlabel(r'$t_1$',fontsize=18)
ax[0].set_ylabel(r'$t_2$',fontsize=18)

ax[1].plot(t,(y2[np.where(t==t0)[0][0],:])**2,'C0',linewidth=3,label=r'$t_1='+str(t0)+'$')
ax[1].set_xlim([0.1,2])
ax[1].set_ylim([0,4])
ax[1].set_xlabel(r'$t_2$',fontsize=18)
ax[1].set_ylabel(r'$y_2^2$',fontsize=18)

ax[2].plot(t,(y2[:,np.where(t==t1)[0][0]])**2,'C0',linewidth=3,label=r'$t_2='+str(t1)+'$')
ax[2].set_xlim([0.05,1])
ax[2].set_ylim([0,250])
ax[2].set_xlabel(r'$t_1$',fontsize=18)
ax[2].set_ylabel(r'$y_2^2$',fontsize=18)

for i in range(1,3):
    ax[i].legend(loc="best", ncol=1, fontsize=16)

fig.subplots_adjust(wspace=0.2)

#%% plots of the sensitivity of the optimal estimates of the model's parameter to the second observation -- figure 14

fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,5))
ax = ax.flat

[t1grid,t2grid] = np.meshgrid(t,t,indexing='ij')
cval = np.array([0.5,2.0,5.0,15,50])

cs = ax[0].contourf(t1grid,t2grid,w2**2,cval,cmap='YlOrBr',extend='both')
cs.set_clim([0, 50])

CS = ax[0].contour(t1grid,t2grid,w2**2,cval,colors='k',linewidths=3)
manual_locations = [(0.2, 1.1), (0.5, 1.2),
                    (1.0, 1.1), (1.3,1.0),
                    (1.25,0.75), (1.0,0.45),
                    (0.8, 0.15) ]

ax[0].clabel(CS,colors='k',fontsize=18,inline=1,rightside_up=True, manual=manual_locations) #contour line labels
ax[0].set_xlim([0.0,2])
ax[0].set_ylim([0.0,2])
ax[0].set_xlabel(r'$t_1$',fontsize=18)
ax[0].set_ylabel(r'$t_2$',fontsize=18)

ax[1].plot(t,(w2[np.where(t==t0)[0][0],:])**2,'C0',linewidth=3,label=r'$t_1='+str(t0)+'$')
ax[1].set_xlim([0.1,2])
ax[1].set_ylim([0,20])
ax[1].set_xlabel(r'$t_2$',fontsize=18)
ax[1].set_ylabel(r'$w_2^2$',fontsize=18)

ax[2].plot(t,(w2[:,np.where(t==t1)[0][0]])**2,'C0',linewidth=3,label=r'$t_2='+str(t1)+'$')
ax[2].set_xlim([0.05,1])
ax[2].set_ylim([0,250])
ax[2].set_xlabel(r'$t_1$',fontsize=18)
ax[2].set_ylabel(r'$w_2^2$',fontsize=18)

for i in range(1,3):
    ax[i].legend(loc="best", ncol=1, fontsize=16)

fig.subplots_adjust(wspace=0.2)


   
