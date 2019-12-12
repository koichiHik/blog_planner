
import sys
sys.path.append("../mod")

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Original Package
from bicycle_model import BicycleModelTrajGenerator, JacobianCalculator

if __name__ == "__main__":

    # Initial Problem Setting.
    wheel_base = 2.0
    t_st = 0
    t_end = 1
    dt = 0.01
    x_init = np.array([0, 0, 0, 0, 0])
    x_final = np.array([7, 5, 0, 0, 0])

    b_traj = BicycleModelTrajGenerator(wheel_base)

    # Adjust Plot
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2,2)
    pose = fig.add_subplot(gs[:,:])
    
    pose.title.set_text("Vehicle Trajectory")
    pose.set_aspect("equal")
    pose.scatter(x_init[0], x_init[1], s=100, marker="o")
    pose.scatter(x_final[0], x_final[1], s=100, marker="o")    

    n_sample = 40

    w_st = -60
    w_end = 60
    for i in range(n_sample):

        w_val = w_st + (w_end - w_st) / n_sample * i
        print(w_val)
        v_coeffs = np.array([0, 0, 0, 2])
        w_coeffs = np.array([0, 0, 0, w_val / 180.0 * math.pi])
        
        # Forward Integration.
        x_tf = b_traj.compute_states(x_init, v_coeffs, w_coeffs, t_st, t_end, dt)

        # Plot Result of Forward Integration.
        pose.plot(x_tf[0,:], x_tf[1,:], label="Iteration Count : " + str(i))

    plt.show()