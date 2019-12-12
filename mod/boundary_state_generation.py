
import math
import numpy as np

def generate_search_space():
    pass

def generate_uniform_boundary_states(p, x_0):

    # p[0] : np, p[1] : nh, p[2] : d, 
    # p[3] : alpha_min, p[4] : alpha_max, 
    # p[5] : phi_min, p[6] : phi_max
    n_p = int(p[0])
    n_h = int(p[1])
    d = p[2]
    alpha_min = p[3] / 180.0 * math.pi
    alpha_max = p[4] / 180.0 * math.pi    
    phi_min = p[5] / 180.0 * math.pi
    phi_max = p[6] / 180.0 * math.pi    
    phi_0 = x_0[2] / 180.0 * math.pi

    x_n = None
    for p_idx in range(n_p):
        for h_idx in range(n_h):
            alpha = (alpha_min + (alpha_max - alpha_min) * p_idx / (n_p - 1))
            x = x_0[0] + d * math.cos(alpha + phi_0)
            y = x_0[1] + d * math.sin(alpha + phi_0)
            phi = phi_0 + phi_min + (phi_max - phi_min) * h_idx / (n_h - 1) + alpha
            x_f = np.array([x, y, phi])
            if (x_n is None):
                x_n = np.append(x_0, x_f).reshape(6, 1)
            else:
                x_n = np.append(x_n, np.append(x_0, x_f).reshape(6, 1), axis=1)
                
    return x_n