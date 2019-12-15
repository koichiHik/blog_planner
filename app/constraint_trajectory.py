
import sys
sys.path.append("../mod")

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Original Package
from bicycle_model import BicycleModelTrajGenerator, JacobianCalculator
from bicycle_model import pi_2_npi

def normalize_angle(state):

    state[2] = pi_2_npi(state[2])
    state[4] = pi_2_npi(state[4])
    state[7] = pi_2_npi(state[7])
    state[9] = pi_2_npi(state[9])

    return state


if __name__ == "__main__":

    # Initial Problem Setting.
    wheel_base = 2.0
    t_st = 0
    t_end = 1
    dt = 0.01
    x_init = np.array([0, 0, 0, 0, 0])
    x_final = np.array([7, 5, 0, 0, 0])
    init_v_coeffs = np.zeros(3)
    init_w_coeffs = np.zeros(3)
    params = np.append(init_v_coeffs, init_w_coeffs)

    b_traj = BicycleModelTrajGenerator(wheel_base)
    jacobian_calc = JacobianCalculator(b_traj)

    # Adjust Plot
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2,2)
    pose = fig.add_subplot(gs[0,:])
    vel = fig.add_subplot(gs[1,0])
    angle = fig.add_subplot(gs[1,1])    
    
    pose.title.set_text("Vehicle Trajectory")
    vel.title.set_text("Vehicle Velocity Profile")
    angle.title.set_text("Vehicle Steering Angle Profile")

    pose.scatter(x_init[0], x_init[1], s=100, marker="o")
    pose.scatter(x_final[0], x_final[1], s=100, marker="o")    

    for i in range(6):

        v_coeffs = params[0: init_v_coeffs.shape[0]]
        w_coeffs = params[init_v_coeffs.shape[0]: params.shape[0]]
        
        # Forward Integration.
        x_tf = b_traj.compute_states(x_init, v_coeffs, w_coeffs, t_st, t_end, dt)

        # Plot Result of Forward Integration.
        pose.plot(x_tf[0,:], x_tf[1,:], label="Iteration Count : " + str(i))
        angle.plot(np.arange(t_st, t_end + dt, dt), x_tf[2, :])

        # Compute Jacobian via Linearization.        
        jaco = jacobian_calc.calc_jacobian(x_init, x_final, v_coeffs, w_coeffs, t_st, t_end, dt)

        # Compute Parameter Correction Vector via Newton Methods
        dx = np.append(x_init, x_final, axis=0) - np.append(x_tf[:,0], x_tf[:,-1], axis=0)

        # Normalize Angle Between PI to -PI
        dx = normalize_angle(dx)

        # Apply Correction Vector to Params.
        jaco_pseudo_inv = np.linalg.pinv(jaco)
        dp = np.matmul(-jaco_pseudo_inv, dx)
        params = params + dp

    # Compute Result with Final Parameter Vector.
    x_tf = b_traj.compute_states(x_init, v_coeffs, w_coeffs, t_st, t_end, dt)

    # Plot Final Result
    pose.plot(x_tf[0,:], x_tf[1,:], label="Final Result")
    vel.plot(np.arange(t_st, t_end + dt, dt), x_tf[3,:])
    angle.plot(np.arange(t_st, t_end + dt, dt), x_tf[4, :])
    #angle.plot(np.arange(t_st, t_end + dt, dt), x_tf[2, :])

    print("Error  : " + str(np.append(x_init, x_final, axis=0) 
        - np.append(x_tf[:,0], x_tf[:,-1], axis=0)))

    pose.legend()
    plt.show()