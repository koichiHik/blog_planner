
import sys
sys.path.append("../mod")

import numpy as np
import matplotlib.pyplot as plt

# Original Package
from bicycle_model import BicycleModelTrajGenerator, JacobianCalculator

if __name__ == "__main__":

    # Initial Problem Setting.
    wheel_base = 2.0
    t_st = 0
    t_end = 1
    dt = 0.01
    x_init = np.array([0, 0, 0, 0, 0])
    x_final = np.array([7, 4, 0, 0, 0])
    init_v_coeffs = np.zeros(3)
    init_w_coeffs = np.zeros(3)
    params = np.append(init_v_coeffs, init_w_coeffs)

    b_traj = BicycleModelTrajGenerator(wheel_base)
    jacobian_calc = JacobianCalculator(b_traj)

    # Adjust Plot
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.scatter(x_init[0], x_init[1], s=100, marker="o")
    plt.scatter(x_final[0], x_final[1], s=100, marker="o")    

    for i in range(6):

        v_coeffs = params[0: init_v_coeffs.shape[0]]
        w_coeffs = params[init_v_coeffs.shape[0]: params.shape[0]]
        
        # Forward Integration.
        x_tf = b_traj.compute_states(x_init, v_coeffs, w_coeffs, t_st, t_end, dt)

        # Plot Result of Forward Integration.
        plt.plot(x_tf[0,:], x_tf[1,:], label="Iteration Count : " + str(i))

        # Compute Jacobian via Linearization.        
        jaco = jacobian_calc.calc_jacobian(x_init, x_final, v_coeffs, w_coeffs, t_st, t_end, dt)
        
        # Compute Parameter Correction Vector via Newton Methods
        dx = x_final - x_tf[:,-1]
        jaco_pseudo_inv = np.linalg.pinv(jaco)
        dp = np.matmul(-jaco_pseudo_inv, dx)
        params = params + dp

    # Compute Result with Final Parameter Vector.
    x_tf = b_traj.compute_states(x_init, v_coeffs, w_coeffs, t_st, t_end, dt)

    # Plot Final Result
    plt.plot(x_tf[0,:], x_tf[1,:], label="Final Result")
    plt.legend()
    plt.show()