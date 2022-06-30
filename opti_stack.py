
import casadi as cs
import numpy as np
def discretize_dynamics_and_cost(t_horizon, n_points, m_steps_per_point, x, u, dynamics_f, cost_f, ind):
    """
    Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
    :param t_horizon: time horizon in seconds
    :param n_points: number of control input points until time horizon
    :param m_steps_per_point: number of integrations steps per control input
    :param x: 4-element list with symbolic vectors for position (3D), angle (4D), velocity (3D) and rate (3D)
    :param u: 4-element symbolic vector for control input
    :param dynamics_f: symbolic dynamics function written in CasADi symbolic syntax.
    :param cost_f: symbolic cost function written in CasADi symbolic syntax. If None, then cost 0 is returned.
    :param ind: Only used for trajectory tracking. Index of cost function to use.
    :return: a symbolic function that computes the dynamics integration and the cost function at n_control_inputs
    points until the time horizon given an initial state and
    """

    # Fixed step Runge-Kutta 4 integrator
    dt = t_horizon / n_points / m_steps_per_point
    x0 = x
    q = 0

    for j in range(m_steps_per_point):
        k1 = dynamics_f(x=x, u=u)['x_dot']
        k2 = dynamics_f(x=x + dt / 2 * k1, u=u)['x_dot']
        k3 = dynamics_f(x=x + dt / 2 * k2, u=u)['x_dot']
        k4 = dynamics_f(x=x + dt * k3, u=u)['x_dot']
        
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q = q + cost_f(x=x, u=u)['q']
        

    return cs.Function('F', [x0, u], [x, q], ['x0', 'p'], ['xf', 'qf'])

x_ref = [15.0,10,1,]
N = 5  # number of control intervals

sigma_max = 0.15
opti = cs.Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(6,N+1) # state trajectory
# pos   = X[0,:]
# speed = X[1,:]
U = opti.variable(3,N)   # control trajectory 
T = opti.variable()
# ---- objective          ---------
opti.minimize(T) # 

# ---- dynamic constraints --------
# f = lambda x,u: vertcat(x[1],u-x[1]) # dx/dt = f(x,u)
def f(x,u):
    # dx/dt = f(x,u)
    px = x[0]
    py = x[1]
    pz = x[2]
    vx = x[3]
    vy = x[4]
    vz = x[5]
    ax = u[0]
    ay = u[1]
    az = u[2]
    vx_dot = ax
    vy_dot = ay
    vz_dot = az
    return cs.vertcat(vx, vy, vz, vx_dot, vy_dot, vz_dot)
    
dt = T/N # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- path constraints -----------
opti.subject_to(opti.bounded(-18.0,U,+18.0)) # control is limited


# ---- boundary conditions --------
opti.subject_to(X[0,0]==0)
opti.subject_to(X[1,0]==0)
opti.subject_to(X[2,0]==0)
opti.subject_to(X[3,0]==0)
opti.subject_to(X[4,0]==0)
opti.subject_to(X[5,0]==0)




opti.subject_to(X[0,-1]==15)
opti.subject_to(X[1,-1]==12)
opti.subject_to(X[2,-1]==-3)
# opti.subject_to(X[3,-1]==0)
# opti.subject_to(X[4,-1]==0)
# opti.subject_to(X[5,-1]==0)
# ---- misc. constraints  ----------

# ---- solve NLP              ------
opts = {};
opts["ipopt"] = dict(max_iter=10000)
opti.solver("ipopt", opts) # set numerical backend
sol = opti.solve()   # actual solve

#%%
# ---- post-processing        ------
import matplotlib.pyplot as plt
# plot(sol.value(speed),label="speed")
# plot(sol.value(pos),label="pos")
# plot(limit(sol.value(pos)),'r--',label="speed limit")
t_opt = sol.value(T)
print("Time:", t_opt)
plt.figure(0)
plt.clf()
plt.step(range(N),sol.value(U).T)
optimal_control_input = sol.value(U).T
sigma = optimal_control_input[:,0]
theta = optimal_control_input[:,1]
phi = optimal_control_input[:,2]

sigma_x = sigma * np.cos(theta) * np.cos(phi)
sigma_y = sigma * np.cos(theta) * np.sin(phi)
sigma_z = sigma * np.sin(theta)
# legend(loc="upper left")

# figure()
# spy(sol.value(jacobian(opti.g,opti.x)))
# figure()
# spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))

# show()
plt.figure(1)
plt.clf()
plt.title("Positions")
plt.plot(sol.value(X[0,:]))
plt.plot(sol.value(X[1,:]))
plt.plot(sol.value(X[2,:]))

plt.figure(2)
plt.clf()
plt.title("Velocities")
plt.plot(sol.value(X[3,:]))
plt.plot(sol.value(X[4,:]))
plt.plot(sol.value(X[5,:]))

plt.figure(3)
plt.clf()
plt.title("sigma")
plt.plot(sigma_x)
plt.plot(sigma_y)
plt.plot(sigma_z)
plt.legend(["sigma_x", "sigma_y", "sigma_z"])

plt.show()