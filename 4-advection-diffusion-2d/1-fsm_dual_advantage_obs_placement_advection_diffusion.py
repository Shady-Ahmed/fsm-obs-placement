"""
Codes to reproduce the results for the 2D advection-diffusion test case in the paper:
    "On the dual advantage of placing observations through forward sensitivity analysis"
    Proceedings of The Royal Society A, 2023.
    Authors: S. E. Ahmed, O. San, S. Lakshmivarahan, and J. M. Lewis
    For questions and/or comments, please feel free to contact me at:
        shady.ahmed@okstate.edu or shady.ahmed@pnnl.gov

Last checked: May 21, 2023
"""

#%% Import libraries
import numpy as np
from scipy.linalg import block_diag, sqrtm

import matplotlib.pyplot as plt

#%% Define functions

def rhs(state,params):
    
    u  = state    
    nu, cx, cy, dx, dy, nx, ny = params

    i = np.arange(1,nx-1) #internal points in x-direction
    j = np.arange(1,ny-1) #internal points in y-direction
    [_i,_j] = np.meshgrid(i,j) #to mitigate nested loops
    _i, _j = _i.flatten() , _j.flatten()
    _ind = _i + _j*nx
    _indE = (_i+1) +  _j*nx     #east
    _indW = (_i-1) +  _j*nx     #west
    _indN =  _i    + (_j+1)*nx  #north
    _indS =  _i    + (_j-1)*nx  #south

    f = np.zeros(nx*ny, dtype=np.double)
    f[_ind] = -cx*(u[_indE]-u[_indW])/(2*dx)\
              -cy*(u[_indN]-u[_indS])/(2*dy)\
              +nu*(u[_indE]-2*u[_ind]+u[_indW])/(dx**2)\
              +nu*(u[_indN]-2*u[_ind]+u[_indS])/(dy**2)

    return f

def Jrhs(state,params): #Jacobian of rhs
    
    #u = state
    nu, cx, cy, dx, dy, nx, ny = params

    i = np.arange(1,nx-1) #internal points in x-direction
    j = np.arange(1,ny-1) #internal points in y-direction
    [_i,_j] = np.meshgrid(i,j) #to mitigate nested loops
    _i, _j = _i.flatten() , _j.flatten()
    _ind = _i + _j*nx
    _indE = (_i+1) +  _j*nx     #east
    _indW = (_i-1) +  _j*nx     #west
    _indN =  _i    + (_j+1)*nx  #north
    _indS =  _i    + (_j-1)*nx  #south
    
    df = np.zeros([nx*ny,nx*ny], dtype=np.double)
    
    df[_ind,_ind] = nu*(-2)/(dx**2) + nu*(-2)/(dy**2)
    df[_ind,_indE] = -cx*(1)/(2*dx) + nu*(1)/(dx**2)
    df[_ind,_indW] = -cx*(-1)/(2*dx) + nu*(1)/(dx**2)
    df[_ind,_indN] = -cy*(1)/(2*dy) + nu*(1)/(dy**2)
    df[_ind,_indS] = -cy*(-1)/(2*dy) + nu*(1)/(dy**2)
        
    #fix boundaries
    _j = np.arange(ny)
    _i = 1
    _ind = _i+_j*nx
    _indW = (_i-1) +  _j*nx   #left boundary
    df[_ind,_indW] = 0
    
    _i = nx-2
    _ind = _i+_j*nx
    _indE = (_i+1) +  _j*nx #right boundary
    df[_ind,_indE] = 0
    
    _i = np.arange(nx)
    _j = 1
    _ind = _i+_j*nx
    _indS = _i + (_j-1)*nx   #bottom boundary
    df[_ind,_indS] = 0
    
    _j = ny-2
    _ind = _i+_j*nx
    _indN = _i + (_j+1)*nx  #top boundary
    df[_ind,_indN] = 0
                                      
    return df

# Forward sensitivites dynamics
def Urhs(state,U,params):
    df = Jrhs(state,params)
    ru = np.dot(df,U) #df @ U 
    return ru


# time integration using third-order Runge Kutta method
def FSM_RK3(state,U,params,dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    #-------------------------#
    #stage-1
    #--------
    r = rhs(state,params)
    ru = Urhs(state,U,params)
    state0 = state + dt*r
    U0 = U + dt*ru
    #-------------------------#
    #stage-2
    #--------
    r = rhs(state0,params)
    ru = Urhs(state0,U0,params)
    state0 = 0.75*state + 0.25*state0 + 0.25*dt*r
    U0 = 0.75*U + 0.25*U0 + 0.25*dt*ru
    #-------------------------#
    #stage-3
    #--------
    r = rhs(state0,params)
    ru = Urhs(state0,U0,params)
    state = c1*state + c2*state0 + c2*dt*r
    U = c1*U + c2*U0 + c2*dt*ru
    return state, U

# time integration using third-order Runge Kutta method
def RK3(state,params,dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    #-------------------------#
    #stage-1
    #--------
    r = rhs(state,params)
    state0 = state + dt*r
    #-------------------------#
    #stage-2
    #--------
    r = rhs(state0,params)
    state0 = 0.75*state + 0.25*state0 + 0.25*dt*r
    #-------------------------#
    #stage-3
    #--------
    r = rhs(state0,params)
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
        state[:,k+1] = RK3(state[:,k],params,dt)
        
        if (km<nt_m) and (k+1==ind_m[km]):
            tmp = measurement[:,km] - obs_operator(state[:,k+1])
            tmp = tmp.reshape(-1,1)
            loss = loss + np.linalg.multi_dot(( tmp.T, Ri , tmp ))
            km = km + 1

    loss = loss[0,0]/2
    return loss


def fsm1st(rhs,Jrhs,obs_operator,obs_operator_jac,t,ind_m,state0b,measurement,R,params):
       
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
    du0 = np.linalg.lstsq(W1@Q, W1@ef, rcond=1e-1)[0]
    
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


def exact_solution(x,y,t,params):
    """Exact solution"""

    x, y = np.meshgrid(x,y)
    x, y = x.reshape(-1,), y.reshape(-1,)

    nu, cx, cy = params[0],  params[1], params[2]
    true_state = 1/(1+4*t) * np.exp( -(x-0.25-cx*t)**2/(nu*(1+4*t)) -(y-0.25-cy*t)**2/(nu*(1+4*t)) )

    return true_state


def background_solution(x,y,t,params):
    """Exact solution"""

    x, y = np.meshgrid(x,y)
    x, y = x.reshape(-1,), y.reshape(-1,)

    nu, cx, cy = params[0],  params[1], params[2]
    state = 1/(1+4*t) * np.exp( -(x-0.5-cx*t)**2/(nu*(1+4*t)) -(y-0.5-cy*t)**2/(nu*(1+4*t)) )

    return state

def apply_bc(state,x,y,t):
    nu, cx, cy, dx, dy, nx, ny = params
    x, y = np.meshgrid(x,y)
    xx, yy = x.reshape(-1,), y.reshape(-1,)
    
    fw = lambda x,y: 1/(1+4*t) * np.exp( -((x-0.25-cx*t)**2+(y-0.25-cy*t)**2)/(nu*(1+4*t)) )

    for i in [0,nx-1]: #left and right boundaries (x=0 for i=0, and x=lx for i=nx-1)
        _ind = [i+j*nx for j in range(ny)]
        state[_ind] = fw(xx[_ind],yy[_ind])

    for j in [0,ny-1]: #top and bottom boundaries (y=0 for j=0, and y=ly for j=ny-1)
        _ind = [i+j*nx for i in range(nx)]
        state[_ind] = fw(xx[_ind],yy[_ind])

    return state


#%% Application: 2D Advection Diffusion

lx, ly = 1, 1
nx, ny = 50, 50 #dimension of state
cx, cy = 0.5, 0.5
nu = 0.01

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)
dx, dy = lx/(nx-1), ly/(ny-1)
xx, yy = np.meshgrid(x,y)

dt = 0.01
tm = 1
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

params = [nu,cx,cy,dx,dy,nx,ny]

#%% initial condition
state0true = exact_solution(x,y,0,params)  #true initial condition

state0b = background_solution(x,y,0,params) #guessed initial condition

#%% Twin experiment
np.random.seed(seed=1)

# Define observations
m = int(nx*ny/1)    
ind_m = np.array([10,20])
nt_m = len(ind_m)
t_m = t[ind_m]

sig_m= 0.1  #standard deviation for measurement noise
R = sig_m**2*np.eye(m) #covariance matrix for measurement noise

#time integration
statetrue = np.zeros([nx*ny,nt+1])
statetrue[:,0] = state0true

stateanalytic = np.zeros([nx*ny,nt+1])
stateanalytic[:,0] = state0true

km = 0
measurement = np.zeros([m,nt_m])
for k in range(nt):
    statetrue[:,k+1] = RK3(statetrue[:,k],params,dt)
    statetrue[:,k+1] = apply_bc(statetrue[:,k+1],x,y,t[k+1])
    if (km<nt_m) and (k+1==ind_m[km]):
        measurement[:,km] = obs_operator(statetrue[:,k+1]) + np.random.normal(0,sig_m,[m,])*np.mean(statetrue[:,k+1])
        km = km+1

#%%
########################### Data Assimilation #################################

state0a = state0b
J0 = loss_fun(RK3,obs_operator,t,ind_m,state0a,measurement,R,params)
plt.figure()
for iter in range(10):
    
    #computing the correction vector
    du0 = fsm1st(rhs,Jrhs,obs_operator,obs_operator_jac,t,ind_m,state0a,measurement,R,params)
    #minimization direction
    p = du0
    #Golden method for linesearch
    alpha = GoldenAlpha(p,rhs,obs_operator,t,ind_m,state0a,measurement,R,params)   
    #update initial condition with gradient descent
    state0a = state0a + alpha*p
    state0a = apply_bc(state0a,x,y,0)
    J = loss_fun(rhs,obs_operator,t,ind_m,state0a,measurement,R,params)
    
    if np.abs(J0-J) < 1e-2:
        print('Convergence: loss function')
        break
    else:
        J0=J
        
    if np.linalg.norm(du0) < 1e-6:
        print('Convergence: correction vector')
        break
    
filename = './results/fsm_ic_t_m=' + str(t_m) +'_noise='+str(sig_m)+'.npy'
np.save(filename,state0a)
