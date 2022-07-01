
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from quad_guidance_optimizer import QuadGuidanceOptimizerCasadi

if __name__=="__main__":
    
    sim_time = 10.0
    dt = 0.02

    target_pos = np.array([5.0,5,0])
    target_vel = np.array([-2.0,0.0,0])
    x_ref = cs.vertcat(target_pos, target_vel)
    
    initial_state = np.array([0.0, 0.0, 0.0, 0,0,0])
    initial_u = [0,0,0.0]
    min_u = 3*[-9.0]
    max_u = 3*[9.0]
    m_steps_per_point = 4
    T = 1.0 # Time horizon
    N = 10  # number of control intervals
    q_diag = [0]*3 + [0]*3
    r_diag = [0.1]*3

    quad_optimizer = QuadGuidanceOptimizerCasadi(T, N, 4, q_diag, r_diag, np.array(max_u))
    
    list_pos = []
    list_vel = []
    list_acc = []
    list_target_pos = []
    x = initial_state
    list_pos.append(np.copy(x[0:3]))
    list_vel.append(np.copy(x[3:6]))
    list_target_pos.append(np.copy(target_pos))
    avg_vel = 15.0
    commanded_acc = np.array([0,0,0], dtype=np.float64)
    for i in range(int(sim_time/dt)):

        if np.dot(target_pos - x[0:3], target_vel) >= 0:
            time_to_go = np.linalg.norm(target_pos - x[0:3]) / np.abs(np.linalg.norm(target_vel)- avg_vel)
        else:
            time_to_go = np.linalg.norm(target_pos - x[0:3]) / np.abs(np.linalg.norm(target_vel)+ avg_vel)

        quad_optimizer = QuadGuidanceOptimizerCasadi(time_to_go, N, 4, q_diag, r_diag, np.array(max_u))
        w_opt = quad_optimizer.solve(initial_state=x, target_pos=target_pos, target_vel=target_vel)
        a1_opt = w_opt[6::9]
        a2_opt = w_opt[7::9]
        a3_opt = w_opt[8::9]
        acc = np.array([a1_opt[0], a2_opt[0], a3_opt[0]])
        
        commanded_acc = acc*0.5 + commanded_acc*0.5
        x[3:6] += commanded_acc*dt
        x[0:3] += x[3:6]*dt +  commanded_acc*dt*dt

        target_pos += target_vel*dt

        list_pos.append(np.copy(x[0:3]))
        list_vel.append(np.copy(x[3:6]))
        list_acc.append(np.copy(commanded_acc))
        list_target_pos.append(np.copy(target_pos))
        if np.linalg.norm(target_pos- x[0:3]) < 0.1:
            print("Collision occured at time:", i*dt)
            break




    plt.figure(1)
    plt.plot(list_pos)
    plt.plot(list_target_pos)

    plt.figure(2)
    plt.plot(list_vel)

    plt.figure(3)
    plt.plot(list_acc)
    plt.figure(4)
    positions = np.vstack(list_pos)
    target_positions = np.vstack(list_target_pos)
    ax = plt.axes(projection='3d')
    ax.plot3D(positions[:,0],positions[:,1],positions[:,2], color="blue")
    ax.plot3D(target_positions[:,0],target_positions[:,1],target_positions[:,2], color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Create cubic bounding box to simulate equal aspect ratio
    X =positions[:,0]
    Y = positions[:,1]
    Z = -positions[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


    plt.show()