# -*- coding: utf-8 -*-
"""
Codes to reproduce the results for the 1D Burgers test case in the paper:
    "On the dual advantage of placing observations through forward sensitivity analysis"
    Proceedings of The Royal Society A, 2023.
    Authors: S. E. Ahmed, O. San, S. Lakshmivarahan, and J. M. Lewis
    For questions and/or comments, please feel free to contact me at:
        shady.ahmed@okstate.edu or shady.ahmed@pnnl.gov

Last checked: May 21, 2023
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, sqrtm

#%% Define functions

def Burgers1D(state,params):
    u = state
    nu = params[0]
    dx = params[1]
    nx = len(u)
    f = np.zeros(nx)
    f[1:nx-1] = (nu/(dx*dx))*(u[2:nx] - 2*u[1:nx-1] + u[0:nx-2]) \
             - (1.0/3.0)*(u[2:nx]+u[0:nx-2]+u[1:nx-1])*(u[2:nx+1]-u[0:nx-2])/(2.0*dx) 
    return f

def JBurgers1D(state,params): #Jacobian of right hand side
    u = state
    nu = params[0]
    dx = params[1]
    nx = len(u)
    df = np.zeros([nx,nx])
    for i in range(1, nx-1):
        df[i,i] = -2*nu/(dx*dx)  - (1.0/3.0)*(u[i+1]-u[i-1])/(2.0*dx) 
        df[i,i+1] = (nu/(dx*dx)) - (1.0/3.0)*(u[i+1]+u[i-1]+u[i])/(2.0*dx) \
                                 - (1.0/3.0)*(1)*(u[i+1]-u[i-1])/(2.0*dx) 
        df[i,i-1] = (nu/(dx*dx)) - (-1.0/3.0)*(u[i+1]+u[i-1]+u[i])/(2.0*dx) \
                                 - (1.0/3.0)*(1)*(u[i+1]-u[i-1])/(2.0*dx) 
    return df

# Forward sensitivites dynamics
def Urhs(state,U,params):
    df = JBurgers1D(state,params)
    ru = np.dot(df,U) #df @ U 
    return ru


# time integration using third-order Runge Kutta method
def FSM_RK3(state,U,params,dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    #-------------------------#
    #stage-1
    #--------
    r = Burgers1D(state,params)
    ru = Urhs(state,U,params)
    state0 = state + dt*r
    U0 = U + dt*ru
    #-------------------------#
    #stage-2
    #--------
    r = Burgers1D(state0,params)
    ru = Urhs(state0,U0,params)
    state0 = 0.75*state + 0.25*state0 + 0.25*dt*r
    U0 = 0.75*U + 0.25*U0 + 0.25*dt*ru
    #-------------------------#
    #stage-3
    #--------
    r = Burgers1D(state0,params)
    ru = Urhs(state0,U0,params)
    state = c1*state + c2*state0 + c2*dt*r
    U = c1*U + c2*U0 + c2*dt*ru
    return state, U

# time integration using third-order Runge Kutta method
def Burgers1D_RK3(state,params,dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    #-------------------------#
    #stage-1
    #--------
    r = Burgers1D(state,params)
    state0 = state + dt*r
    #-------------------------#
    #stage-2
    #--------
    r = Burgers1D(state0,params)
    state0 = 0.75*state + 0.25*state0 + 0.25*dt*r
    #-------------------------#
    #stage-3
    #--------
    r = Burgers1D(state0,params)
    state = c1*state + c2*state0 + c2*dt*r
    return state

# Observation operator
def obs_operator(state):
    n= state.shape[0]
    H = np.zeros((m,n))
    di = int(n/m) #distance between measurements
    for i in range(m):
        H[i,(i+1)*di-1] = 1
    measurement = H.dot(state)
    return measurement

# Jacobian of observational map
def obs_operator_jac(state):
    n = state.shape[0]
    H = np.zeros((m,n))    
    di = int(n/m) # distance between measurements 
    for i in range(m):
        H[i,(i+1)*di-1] = 1

    return H

# Loss function (z-h(x))^T * R^{-1} * (z-h(x))
def loss_fun(rhs,obs_operator,t,ind_m,state0,measurement,R,params):
    
    n = len(state0)
    #determine the assimilation window
    t = t[:ind_m[-1]+1] #cut the time till the last observation point 
    nt = len(t)-1
    dt = t[1] - t[0]
    state = np.zeros([n,nt+1]) #trajectory
    state[:,0] = state0

    Ri = np.linalg.inv(R)
    loss = 0 
    km = 0
    nt_m = len(ind_m)
    #forward model
    for k in range(nt):
        state[:,k+1] = Burgers1D_RK3(state[:,k],params,dt)
        
        if (km<nt_m) and (k+1==ind_m[km]):
            tmp = measurement[:,km] - obs_operator(state[:,k+1])
            tmp = tmp.reshape(-1,1)
            loss = loss + np.linalg.multi_dot(( tmp.T, Ri , tmp ))
            km = km + 1

    loss = loss[0,0]/2
    return loss


def fsm1st(rhs,Jrhs,obs_operator,obs_operator_jac,t,ind_m,state0b,measurement,R,params,rcond):
       
    n = len(state0b)
    #determine the assimilation window
    t = t[:ind_m[-1]+1] #cut the time till the last observation point 
    nt = len(t)-1
    dt = t[1] - t[0]
    stateb = np.zeros([n,nt+1]) #base trajectory   
    Ri = np.linalg.inv(R)

    stateb[:,0] = state0b
    U = np.eye(n,n) #Initialization of U
    Q = np.zeros((1,n))  #Dh*U
    ef = np.zeros((1,1)) #w-h(u)
    W = np.zeros((1,1))  #weighting matrix
    km = 0
    nt_m = len(ind_m)

    for k in range(nt):
        stateb[:,k+1], U = FSM_RK3(stateb[:,k],U,params,dt)

        if (km<nt_m) and (k+1==ind_m[km]):
            tmp = measurement[:,km] - obs_operator(stateb[:,k+1])
            ek = tmp.reshape(-1,1)
            ef = np.vstack((ef,ek))
            Qk = obs_operator_jac(stateb[:,k+1]).dot(U)
            Q = np.vstack((Q,Qk))
            W = block_diag(W,Ri)
            km = km + 1
            
    Q = np.delete(Q, (0), axis=0)
    ef = np.delete(ef, (0), axis=0)
    W = np.delete(W, (0), axis=0)
    W = np.delete(W, (0), axis=1)
    
    # solve weighted least-squares
    W1 = sqrtm(W) 
    du0 = np.linalg.lstsq(W1@Q, W1@ef, rcond=rcond)[0]
    
    return du0.ravel()


def GoldenAlpha(p,rhs,obs_operator,t,ind_m,state0,measurement,R,params):
    
    # p is the optimization direction
    a0=0
    b0=1
    r=(3-np.sqrt(5))/2
    
    uncert = 1e-2 # Specified uncertainty
    a1= a0 + r*(b0-a0)
    b1= b0 - r*(b0-a0)
    while (b0-a0) > uncert:
        la = loss_fun(rhs,obs_operator,t,ind_m,state0+a1*p,measurement,R,params)
        lb = loss_fun(rhs,obs_operator,t,ind_m,state0+b1*p,measurement,R,params)

        if  la <  lb:  
            b0=b1;
            b1=a1;
            a1= a0 + r*(b0-a0);
        else:
            a0=a1;
            a1=b1;
            b1= b0 - r*(b0-a0);
    alpha = (b0+a0)/2

    return alpha

def exact_solution(x,t,nu):
    """Exact solution"""
    t0 = np.exp(1/(8*nu))
    true_state = x/(t+1) / (1+np.sqrt((t+1)/t0)*np.exp((x**2)/(4*nu*(t+1))))
    return true_state

def apply_bc(state):
    state[0] = 0
    state[-1] = 0
    return state

#%% Application: Burgers 1D
lx = 1
nx = 128 #dimension of state
x = np.linspace(0,lx,nx)

re = 500 #Reynolds number

dt = 0.01
tm = 1
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

dx = lx/(nx-1)
nu = 1/re
params = [nu,dx]

#%% initial condition
state0true = exact_solution(x,0,nu)  #true initial condition

# guessed initial condition
state0b = np.sin(2*np.pi*x/lx)
state0b = apply_bc(state0b)

#%% Twin experiment
# Define observations
m = 128    
ind_m = np.array([25,35])
nt_m = len(ind_m)  #number of observation instants
t_m = t[ind_m]

sig_m= 0.2  #standard deviation for measurement noise
R = sig_m**2*np.eye(m) #covariance matrix for measurement noise

#time integration
statetrue = np.zeros([nx,nt+1])
statetrue[:,0] = state0true

km = 0
measurement = np.zeros([m,nt_m])
for k in range(nt):
    statetrue[:,k+1] = Burgers1D_RK3(statetrue[:,k],params,dt)

    if (km<nt_m) and (k+1==ind_m[km]):
        measurement[:,km] = obs_operator(statetrue[:,k+1]) + np.random.normal(0,sig_m,[m,])*np.mean(statetrue[:,k+1])
        km = km+1
     
#%%
########################### Data Assimilation #################################
rcond = 1e-1
state0a = state0b

J0 = loss_fun(Burgers1D_RK3,obs_operator,t,ind_m,state0a,measurement,R,params)
plt.figure()
for iter in range(200):
    
    #computing the correction vector
    du0 = fsm1st(Burgers1D,JBurgers1D,obs_operator,obs_operator_jac,t,ind_m,state0a,measurement,R,params,rcond)
    #minimization direction
    p = du0
    #Golden method for linesearch
    alpha = GoldenAlpha(p,Burgers1D,obs_operator,t,ind_m,state0a,measurement,R,params)   
    #update initial condition with gradient descent
    state0a = state0a + alpha*p
    state0a = apply_bc(state0a)
    J = loss_fun(Burgers1D,obs_operator,t,ind_m,state0a,measurement,R,params)
    
    if np.abs(J0-J < 1e-2):
        print('Convergence: loss function')
        break
    else:
        J0=J
        
    if np.linalg.norm(du0) < 1e-6:
        print('Convergence: correction vector')
        break
    
filename = './results/fsm_ic_t_m=' + str(t_m) +'_noise='+str(sig_m)+'.npy'
np.save(filename,state0a)
