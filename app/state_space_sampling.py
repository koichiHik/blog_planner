
import sys
sys.path.append("../mod")

import numpy as np
import matplotlib.pyplot as plt

# Original Package
from bicycle_model import BicycleModelTrajGenerator
from boundary_state_generation import generate_uniform_boundary_states

if __name__ == "__main__":

    plt.figure(figsize=(10, 10))
    plt.xlim(-4, 6)
    plt.ylim(-5, 5)    


    # Initial Problem Setting.
    wheel_base = 2.0
    b_traj = BicycleModelTrajGenerator(wheel_base)

    #p = np.array([3, 3, 5.0, 45, -45, 45, -45])
    p = np.array([3, 3, 5.0, 45, 0, 45, 0])
    x_0 = np.array([0, 0, 0])

    v_coeffs = np.zeros(5)
    w_coeffs = np.zeros(5)
    x_N = generate_uniform_boundary_states(p, x_0)

    print(x_N)

    for i in range(x_N.shape[1]):
        traj = b_traj.generate_trajectory(np.append(x_N[0:3, i], np.array([1, 0])),
         np.append(x_N[3:6, i], np.array([1, 0])), v_coeffs, w_coeffs, 0, 1, 0.01)
        #traj = b_traj.generate_trajectory_wo_vel_constraint(x_N[0:3, i], x_N[3:6, i], v_coeffs, w_coeffs, 0, 1, 0.01)
        if (traj is not None):
            plt.plot(traj[0, :], traj[1, :])
        else:
            print("Failed")

    plt.scatter(0, 0,s=30,c='red')
    plt.scatter(x_N[3,:], x_N[4,:],s=30,c='blue')
    plt.show()
