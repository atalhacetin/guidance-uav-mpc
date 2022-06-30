import casadi as cs
import numpy as np

# Model equations
def f(x,u):
    
    return cs.vertcat(x[3:6], u)

def discretize_dynamics_and_cost(t_horizon, n_points, m_steps_per_point, x, u, dynamics_f, cost_f):
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

class QuadGuidanceOptimizer():
    def __init__(self, T, N, m_steps_per_point, q_diag, r_diag, acc_limits=np.array([18.0, 18.0, 18.0])):
        
        self.T = T
        self.N = N
        self.m_steps_per_point = m_steps_per_point
        self.Q = cs.diag(q_diag)
        self.R = cs.diag(r_diag)
        self.min_u = (-acc_limits).tolist()
        self.max_u = acc_limits.tolist()

        self.x = cs.MX.sym('x', 6) 
        self.u = cs.MX.sym('u', 3)
        self.x_dot = f(self.x,self.u)
        self.L = (self.x).T @ self.Q @ (self.x) + self.u.T @ self.R @ self.u 
        self.dynamics_f = cs.Function('x_dot', [self.x, self.u], [self.x_dot], ['x', 'u'], ['x_dot'])
        self.cost_f = cs.Function('q', [self.x, self.u], [self.L], ['x', 'u'], ['q'])
        self.F = discretize_dynamics_and_cost(T, N, m_steps_per_point, self.x, self.u, self.dynamics_f, self.cost_f)
    
    def solve(self, initial_state, target_pos, target_vel):
        initial_u = [0.0, 0.0, 0.0]
        initial_state = initial_state.tolist()
        w=[]
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g=[]
        lbg = []
        ubg = []
        
        for i in range(len(target_pos)):
            if (abs(initial_state[i] - target_pos[i])<0.01):
                initial_state[i] = target_pos[i]
        
        # "Lift" initial conditions
        Xk = cs.MX.sym('X_0', len(initial_state))
        w += [Xk]
        lbw += initial_state
        ubw += initial_state
        w0 += initial_state
        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = cs.MX.sym('U_' + str(k), len(initial_u))
            w   += [Uk]
            lbw += self.min_u
            ubw += self.max_u
            w0  += initial_u
        
            # Integrate till the end of the interval
            Fk = self.F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J=J+Fk['qf']*k**3
        
            # New NLP variable for state at end of interval
            Xk = cs.MX.sym('X_' + str(k+1), len(initial_state))
            w   += [Xk]
            lbw += [-cs.inf]*6
            ubw += [cs.inf]*6
            # lbw += [-50]*6
            # ubw += [50]*6
            w0  += [0]*len(initial_state)
        
            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0]*len(initial_state)
            ubg += [0]*len(initial_state)
            if k == self.N-1:
                g += [cs.cross(Xk_end[3:6] , target_vel)]
                lbg += [0]*3
                ubg += [0]*3
                
            if k == self.N-1:
                g += [Xk_end[0:3]]
                # lbg += [target_pos[0], target_pos[1], target_pos[2]]
                # ubg += [target_pos[0], target_pos[1], target_pos[2]]
                gain = 1.0
                lbg += [target_pos[0]+target_vel[0]*gain*self.T, target_pos[1]+target_vel[1]*gain*self.T, target_pos[2]+target_vel[2]*gain*self.T]
                ubg += [target_pos[0]+target_vel[0]*gain*self.T, target_pos[1]+target_vel[1]*gain*self.T, target_pos[2]+target_vel[2]*gain*self.T]
                

        
        # Create an NLP solver
        prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
        opts = {'ipopt.print_level':0}
        
        solver = cs.nlpsol('solver', 'ipopt', prob, opts)
        
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        a1_opt = w_opt[6::9]
        a2_opt = w_opt[7::9]
        a3_opt = w_opt[8::9]
        return w_opt



if __name__=="__main__":


    target_pos = np.array([5,5,0])
    target_vel = np.array([2,0,-2])
    x_ref = cs.vertcat(target_pos, target_vel)
    
    
    initial_state = np.array([0.0, 0.0, 0.0, 0,0,0])
    initial_u = [0,0,0.0]
    min_u = 3*[-18.0]
    max_u = 3*[18.0]
    m_steps_per_point = 4
    T = 1.0 # Time horizon
    N = 10  # number of control intervals
    q_diag = [0]*3 + [0,0,0]
    r_diag = [1]*3

    quad_optimizer = QuadGuidanceOptimizer(T, N, 4, q_diag, r_diag, np.array(max_u))
    
    w_opt = quad_optimizer.solve(initial_state=initial_state, target_pos=target_pos, target_vel=target_vel)

        

    # Plot the solution
    x1_opt = w_opt[0::9]
    x2_opt = w_opt[1::9]
    x3_opt = w_opt[2::9]
    
    v1_opt = w_opt[3::9]
    v2_opt = w_opt[4::9]
    v3_opt = w_opt[5::9]

    a1_opt = w_opt[6::9]
    a2_opt = w_opt[7::9]
    a3_opt = w_opt[8::9]

    tgrid = [T/N*k for k in range(N+1)]
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    plt.figure(1)
    plt.clf()
    plt.plot(x1_opt)
    plt.plot(x2_opt, '-')
    plt.plot(x3_opt, '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2','x3'])
    plt.grid()
    
    plt.figure(2)
    plt.clf()
    plt.plot(v1_opt)
    plt.plot(v2_opt, '-')
    plt.plot(v3_opt, '-.')
    plt.xlabel('t')
    plt.legend(['v1','v2','v3'])
    plt.grid()

    plt.figure(3)
    plt.clf()
    plt.plot(a1_opt)
    plt.plot(a2_opt, '-')
    plt.plot(a3_opt, '-.')
    plt.xlabel('t')
    plt.legend(['a1','a2','a3'])
    plt.grid()

    
    plt.figure(4)
    ax = plt.axes(projection='3d')
    ax.plot3D(x1_opt,x2_opt,x3_opt, color="blue")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    
    fig , axs= plt.subplots(1,1)
    fig = plt.figure(5)
    axs.plot(x1_opt, x2_opt)
    axs.set_aspect('equal', 'box')
    plt.show()
    