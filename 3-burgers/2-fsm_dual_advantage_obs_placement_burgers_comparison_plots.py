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

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)

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

def JBurgers1D(state,params): #Jacobian of Lorenz 96 model
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
state0true = exact_solution(x,0,nu)  # Initial state

# perturb IC
np.random.seed(seed=1)

sig_b = 1
state0b = np.sin(2*np.pi*x/lx)
state0b = apply_bc(state0b)

t_m_list = [ np.array([0.01,0.05]), np.array([0.01,0.35]), np.array([0.25,0.5]), np.array([0.5,1.0]) ]
sig_m_list = [0.01,0.05,0.1,0.2]
state0a_list_tm = []
for t_m in t_m_list:
    state0a_list_noise = []
    for sig_m in sig_m_list:
        filename = './results/fsm_ic_t_m=' + str(t_m) +'_noise='+str(sig_m)+'.npy'
        state0a = np.load(filename)
        state0a_list_noise.append(state0a)
    state0a_list_tm.append(state0a_list_noise)
    
#%%
##################### Time Integration [Comparison] ###########################

statetrue = np.zeros([nx,nt+1])
statetrue[:,0] = state0true
for k in range(nt):
    statetrue[:,k+1] = Burgers1D_RK3(statetrue[:,k],params,dt)
        
stateb = np.zeros([nx,nt+1])
stateb[:,0] = state0b
Ub = np.eye(nx)
AA = Ub.T@Ub

ubb1 = np.zeros(nt+1)
ubb1[0] = np.trace(AA)

ubb2 = np.zeros(nt+1)
ubb2[0] = (1/2)*(np.trace(AA)**2 - np.trace(AA@AA))
for k in range(nt):           
    stateb[:,k+1], Ub = FSM_RK3(stateb[:,k],Ub,params,dt)
    AA = Ub.T@Ub
    ubb1[k+1] = np.trace(AA)
    ubb2[k+1] = (1/2)*(np.trace(AA)**2 - np.trace(AA@AA))

#%%
statea_list_tm = []
for i, t_m in enumerate(t_m_list):
    statea_list_noise = []
    state0a_list_noise = state0a_list_tm[i]
    
    for j, sig_m in enumerate(sig_m_list):    
        state0a = state0a_list_noise[j]
        statea = np.zeros([nx,nt+1])
        statea[:,0] = state0a
        
        for k in range(nt):           
            statea[:,k+1] = Burgers1D_RK3(statea[:,k],params,dt)
        statea_list_noise.append(statea)
    statea_list_tm.append(statea_list_noise)


#%%Plotting 
#%% plot the first two invariants for the model forecast sensitivities with respect to the initial condition -- figure 15

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

ax[0].plot(t[:],ubb1[:],linewidth = 2)
ax[0].set_ylabel(r'$I_1$',fontsize=20)


ax[1].plot(t[1:],ubb2[1:],linewidth = 2)
ax[1].set_ylabel(r'$I_2$',fontsize=20)

for i in range(2):
    ax[i].set_xlabel(r'$t$',fontsize=20)
    ax[i].set_xticks([0,0.25,0.50,0.75,1.0])

fig.subplots_adjust(wspace=0.3)

#%% plot the comparison between inverse problem solutions at different observation placement times and varying levels of noise  -- figure 16

fig, ax = plt.subplots(nrows=4,ncols=4, figsize=(20,15))
for i, t_m in enumerate(t_m_list):
    state0a_list_noise = state0a_list_tm[i]
    for j, sig_m in enumerate(sig_m_list):    
        state0a = state0a_list_noise[j]
        ax[i][j].plot(x,state0true, label=r'\bf{True}', linewidth = 3, color='C0')
        ax[i][j].plot(x,state0b, ':', label=r'\bf{Background}', linewidth = 3, color='C1')
        ax[i][j].plot(x,state0a, '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
        ax[i][j].set_xlabel(r'$x$',fontsize=22)
        ax[i][j].set_ylabel(r'$q(x,t=0)$',fontsize=22)
        
        if i == 0:
            ax[i][j].set_title(r'\textbf{Obs. noise:} $'+str(int(sig_m*100))+'\%$',fontsize=22)

    ax[i][0].text(-0.9, 0.5, r'\textbf{Obs. times:\\ \\ } $ t\in \{'+str(t_m[0])+','+str(t_m[1])+'\}$',va='center',ha='center',fontsize=22, transform=ax[i][0].transAxes)

ax[3][1].legend(loc="center", bbox_to_anchor=(1.1,-0.6),ncol=3,fontsize=24)
fig.subplots_adjust(wspace=0.5,hspace=0.5)

#%% plot the predicted velocity field and  relative error -- figure 17

lt = np.linalg.norm(statetrue,axis=0)
lb = np.linalg.norm(statetrue-stateb,axis=0)
fig, ax = plt.subplots(nrows=4,ncols=3, figsize=(20,15))
for i, t_m in enumerate(t_m_list):
    statea_list_noise = statea_list_tm[i]
    statea = statea_list_noise[2]
    la = np.linalg.norm(statetrue-statea,axis=0)

    for j, nn in enumerate([50,100]):
        ax[i][j].plot(x,statetrue[:,nn], label=r'\bf{True}', linewidth = 3, color='C0')
        ax[i][j].plot(x,stateb[:,nn], ':', label=r'\bf{Background}', linewidth = 3, color='C1')
        ax[i][j].plot(x,statea[:,nn], '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
        ax[i][j].set_xlabel(r'$x$',fontsize=22)
        ax[i][j].set_ylabel(r'$q(x,t='+str(t[nn])+')$',fontsize=22)

    ax[i][0].text(-0.6, 0.5, r'\textbf{Obs. times:\\ \\ } $ t\in \{'+str(t_m[0])+','+str(t_m[1])+'\}$',va='center',ha='center',fontsize=22, transform=ax[i][0].transAxes)
    ax[i][2].plot(t,lb/lt, ':', label=r'\bf{Background}', linewidth = 3, color='C1')
    ax[i][2].plot(t,la/lt, '--', label=r'\bf{Analysis}', linewidth = 3, color='C3')
    ax[i][2].set_xlabel(r'$t$',fontsize=22)
    ax[i][2].set_ylabel(r'$\frac{\big\|q^{\bf{True}}  - q^{\bf{Pred}}\big\|_2}{\big\|q^{\bf{True}}\big\|_2}$',fontsize=28)

ax[3][1].legend(loc="center", bbox_to_anchor=(0.3,-0.6),ncol=3,fontsize=24)
fig.subplots_adjust(wspace=0.4,hspace=0.5)


