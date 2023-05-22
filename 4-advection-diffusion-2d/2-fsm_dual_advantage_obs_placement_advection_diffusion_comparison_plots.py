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
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)
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

#%%
########################### Reading data #################################
sig_m= 0.1  # standard deviation for measurement noise

t_m_list = [np.array([0.01,0.05]),np.array([0.1,0.2]),np.array([0.25,0.5]),np.array([0.5,1.0])]

state0alist = []
for t_m in t_m_list:
    
    filename = './results/fsm_ic_t_m=' + str(t_m) +'_noise='+str(sig_m)+'.npy'
    state0a = np.load(filename)
    state0alist.append(state0a)

#%%
##################### Time Integration [Comparison] ###########################

stateb = np.zeros([nx*ny,nt+1])
stateb[:,0] = state0b
statetrue = np.zeros([nx*ny,nt+1])
statetrue[:,0] = state0true

Ub = np.eye(nx*ny)
AA = Ub.T@Ub

ubb1 = np.zeros(nt+1)
ubb1[0] = np.trace(AA)

ubb2 = np.zeros(nt+1)
ubb2[0] = (1/2)*(np.trace(AA)**2 - np.trace(AA@AA))

for k in range(nt):

    statetrue[:,k+1] = RK3(statetrue[:,k],params,dt)
    
    stateb[:,k+1], Ub = FSM_RK3(stateb[:,k],Ub,params,dt)
    stateb[:,k+1] = apply_bc(stateb[:,k+1],x,y,t[k+1])
    
    AA = Ub.T@Ub
    ubb1[k+1] = np.trace(AA)
    ubb2[k+1] = (1/2)*(np.trace(AA)**2 - np.trace(AA@AA))

statealist = []
for i, t_m in enumerate(t_m_list):
    state0a = state0alist[i]
    
    statea = np.zeros([nx*ny,nt+1])
    statea[:,0] = state0a
    for k in range(nt):
        statea[:,k+1] = RK3(statea[:,k],params,dt)
        statea[:,k+1] = apply_bc(statea[:,k+1],x,y,t[k+1])
    statealist.append(statea)

#%% Plotting
#%% plot the first two invariants for the model forecast sensitivities with respect to initial conditions  -- figure 18

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

ax[0].plot(t[1:],ubb1[1:],linewidth = 2)
ax[0].set_ylabel(r'$I_1$',fontsize=20)

ax[1].semilogy(t[1:],ubb2[1:],linewidth = 2)
ax[1].set_ylabel(r'$I_2$',fontsize=20)

for i in range(2):
    ax[i].set_xlabel(r'$t$',fontsize=20)
    ax[i].set_xticks([0,0.25,0.50,0.75,1.0])

fig.subplots_adjust(wspace=0.3)

#%% plot the comparison between inverse problem solutions at different observation placement times -- figure 19

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(20,12))
ax = ax.flat
colormap = 'RdBu_r'
nlvl = 120
lvl0 = state0true.min()
lvl1 = state0true.max()
lvl = np.linspace(lvl0, lvl1, nlvl, endpoint=True)
ctick = np.linspace(lvl0, lvl1, 5, endpoint=True)

ct= ax[0].tricontourf(xx.flatten(),yy.flatten(),state0true,lvl,cmap=colormap,extend='both')
ax[0].set_title(r'\textbf{True}', pad=16)
for c in ct.collections:
    c.set_edgecolor("face")
    
ct= ax[1].tricontourf(xx.flatten(),yy.flatten(),state0b,lvl,cmap=colormap,extend='both')
ax[1].set_title(r'\textbf{Background}', pad=16)
for c in ct.collections:
    c.set_edgecolor("face")
    
for i,t_m in enumerate(t_m_list): 
    state0a = state0alist[i]
    ct= ax[i+2].tricontourf(xx.flatten(),yy.flatten(),state0a,lvl,cmap=colormap,extend='both')
    ax[i+2].set_title(r'\textbf{Analysis:  Obs. @} $t\in \{'+str(t_m[0])+','+str(t_m[1])+'\}$', pad=16)
    for c in ct.collections:
        c.set_edgecolor("face")
    
for i in range(6):
    ax[i].set_xlabel(r'$x$',fontsize=20)
    ax[i].set_ylabel(r'$y$',fontsize=20)
    
fig.subplots_adjust(wspace=0.4,hspace=0.4)

#%% plot the predicted solution and the relative error -- figure 20

lt = np.linalg.norm(statetrue,axis=0)
lb = np.linalg.norm(statetrue-stateb,axis=0)

fig, ax = plt.subplots(nrows=1,ncols=4,figsize=(20,4))
ax = ax.flat
colormap = 'RdBu_r'
nlvl = 120
lvl0 = statetrue[:,-1].min()
lvl1 = statetrue[:,-1].max()
lvl = np.linspace(lvl0, lvl1, nlvl, endpoint=True)
ctick = np.linspace(lvl0, lvl1, 5, endpoint=True)

ct= ax[0].tricontourf(xx.flatten(),yy.flatten(),statetrue[:,-1],lvl,cmap=colormap,extend='both')
ax[0].set_title(r'\textbf{True}', pad=10,fontsize=20)
for c in ct.collections:
    c.set_edgecolor("face")
            
ct= ax[1].tricontourf(xx.flatten(),yy.flatten(),stateb[:,-1],lvl,cmap=colormap,extend='both')
ax[1].set_title(r'\textbf{Background}', pad=10,fontsize=20)
for c in ct.collections:
    c.set_edgecolor("face")
    
statea = statealist[0]
la = np.linalg.norm(statetrue-statea,axis=0)

ct= ax[2].tricontourf(xx.flatten(),yy.flatten(),statea[:,-1],lvl,cmap=colormap,extend='both')
ax[2].set_title(r'\textbf{Analysis}', pad=10,fontsize=20)
for c in ct.collections:
    c.set_edgecolor("face")

for i in range(3):
    ax[i].set_xlabel(r'$x$',fontsize=20)
    ax[i].set_ylabel(r'$y$',fontsize=20)
    ax[i].set_xticks([0.0,0.25,0.5,0.75,1.0])
    ax[i].set_yticks([0.0,0.25,0.5,0.75,1.0])

ax[3].plot(t,lb/lt, ':', label=r'\bf{Background}', linewidth = 3, color='C1')
ax[3].plot(t,la/lt, '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
ax[3].set_xlabel(r'$t$',fontsize=22)
ax[3].set_ylabel(r'${\big\|q^{\bf{True}}  - q^{\bf{Pred}}\big\|_2}/{\big\|q^{\bf{True}}\big\|_2}$',fontsize=20)

ax[3].legend(loc="center left",fontsize=16)
fig.subplots_adjust(wspace=0.4,hspace=0.4)





