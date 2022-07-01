import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import os
import sys
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

class QuadGuidanceOptimizerAcados():
    def __init__(self, t_horizon, number_of_nodes, m_steps_per_point, q_diag, r_diag, acc_limits=np.array([18.0, 18.0, 18.0])):
        self.T = t_horizon
        self.N = number_of_nodes
        self.m_steps_per_point = m_steps_per_point
        self.Q = cs.diag(q_diag)
        self.R = cs.diag(r_diag)
        self.min_u = (-acc_limits).tolist()
        self.max_u = acc_limits.tolist()
        
        # Declare model variables
        self.p = cs.MX.sym('p', 3)  # position
        self.v = cs.MX.sym('v', 3)  # velocity
        
        # Full state vector (13-dimensional)
        self.x = cs.vertcat(self.p, self.v)
        self.state_dim = 6

        # Control input vector
        u1 = cs.MX.sym('ax')
        u2 = cs.MX.sym('ay')
        u3 = cs.MX.sym('az')
        self.u = cs.vertcat(u1, u2, u3)
        self.quad_xdot = self.quad_dynamics()
        acados_model = self.acados_setup_model(
            self.quad_xdot(x=self.x, u=self.u)['x_dot'],"quad")
        
        self.acados_ocp_solver = {}

        nx = acados_model.x.size()[0]
        nu = acados_model.u.size()[0]
        ny = nx + nu
        n_param = acados_model.p.size()[0] if isinstance(acados_model.p, cs.MX) else 0
        
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        # sys.path.insert(0, '../common')

        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = acados_model
        ocp.dims.N = self.N
        ocp.solver_options.tf = t_horizon

        # Init params
        ocp.dims.np = n_param
        ocp.parameter_values = np.zeros(n_param)

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = np.diag(np.concatenate((q_diag, r_diag)))
        ocp.cost.W_e = np.diag(q_diag)
        terminal_cost = 0
        ocp.cost.W_e *= terminal_cost

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-3:, -3:] = np.eye(nu)

        ocp.cost.Vx_e = np.eye(nx)

        # Initial reference trajectory (will be overwritten)
        x_ref = np.zeros(nx)
        ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0])))
        ocp.cost.yref_e = x_ref

        # Initial state (will be overwritten)
        ocp.constraints.x0 = x_ref

        # Set constraints
        ocp.constraints.lbu = np.array(self.min_u)
        ocp.constraints.ubu = np.array(self.max_u)
        ocp.constraints.idxbu = np.array([0, 1, 2])  

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' 

        # Compile acados OCP solver if necessary
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file='./' + acados_model.name + '_acados_ocp.json')



    def acados_setup_model(self, symbolic_model, model_name):
        x_dot = cs.MX.sym('x_dot', symbolic_model.shape)
        f_impl = x_dot - symbolic_model
        
        # Acados dynamic model
        model = AcadosModel()
        model.f_expl_expr = symbolic_model
        model.f_impl_expr = f_impl
        model.x = self.x
        model.xdot = x_dot
        model.u = self.u
        model.p = []
        model.name = model_name

        return model

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.
        :param rdrv_d: a 3x3 diagonal matrix containing the D matrix coefficients for a linear drag model as proposed
        by Faessler et al.
        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """
    
        x_dot = cs.vertcat(self.v , self.u)
        return cs.Function('x_dot', [self.x, self.u], [x_dot], ['x', 'u'], ['x_dot'])
    
    def run_optimization(self, initial_state=None, target_pos=None, target_vel=None, return_x=False):
        

        if initial_state is None:
            initial_state = [0, 0, 0] + [0, 0, 0]

        # Set initial state.
        x_init = initial_state
        x_init = np.stack(x_init)
        for j in range(self.N):
            ref = np.concatenate(( target_pos+j/self.N*self.T*target_vel, 2*target_vel, np.array([0, 0, 0])))
            print(ref)
            self.acados_ocp_solver.set(j, "yref", ref)
        self.acados_ocp_solver.set(self.N, "yref", ref[:-3])
        # Set initial condition, equality constraint
        self.acados_ocp_solver.set(0, 'lbx', x_init)
        self.acados_ocp_solver.set(0, 'ubx', x_init)
        # Solve OCP
        self.acados_ocp_solver.solve()

        # Get u
        w_opt_acados = np.ndarray((self.N, 3))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver.get(i + 1, "x")

        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return w_opt_acados if not return_x else (w_opt_acados, x_opt_acados)
        
    
class QuadGuidanceOptimizerCasadi():
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

def main_casadi():
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

    quad_optimizer = QuadGuidanceOptimizerCasadi(T, N, 4, q_diag, r_diag, np.array(max_u))
    
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

def main_acados():
    import time
    import matplotlib.pyplot as plt
    target_pos = np.array([1,1,0])
    target_vel = np.array([0,0,0])
    x_ref = cs.vertcat(target_pos, target_vel)
    
    
    initial_state = np.array([0.0, 0.0, 0.0, 0,0,0])
    initial_u = [0,0,0.0]
    min_u = 3*[-18.0]
    max_u = 3*[18.0]
    m_steps_per_point = 4
    T = 3.0 # Time horizon
    N = 50  # number of control intervals
    q_diag = [3]*3 + [0,0,0]
    r_diag = [0.1]*3

    quad_optimizer = QuadGuidanceOptimizerAcados(T, N, 4, q_diag, r_diag, np.array(max_u))
    t_start = time.time()
    w_opt, x_opt= quad_optimizer.run_optimization(initial_state=initial_state,target_pos=target_pos, target_vel=target_vel, return_x=True)
    print("time:", time.time()-t_start)
    print("w_opt:", w_opt)
    print("x_opt:", x_opt)
    plt.figure(1)
    plt.clf()
    plt.plot(x_opt[:, 0:3])
    
    plt.figure(2)
    plt.clf()
    plt.plot(x_opt[:, 3:6])
    
    plt.figure(3)
    plt.clf()
    plt.step(np.linspace(0,T,N), w_opt.reshape(3,N).T)
    plt.show()
    
if __name__=="__main__":
    main_acados()

    