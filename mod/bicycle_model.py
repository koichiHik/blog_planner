
import numpy as np
import math

class PolynomialCalculator:

    def __init__(self, coeffs):
        self.__coeffs = coeffs

    def calc(self, x):
        
        if (type(x) is np.ndarray):
            val = np.zeros(x.shape[0])
        else:
            val = 0

        order = self.__coeffs.shape[0] - 1
        for i in range(self.__coeffs.shape[0]):
            val = val + self.__coeffs[i] * x ** (order - i)

        return val

class JacobianCalculator():

    def __init__(self, b_traj, ep=0.00000001):
        self.__ep = ep
        self.__b_traj = b_traj

    def calc_jacobian(self, x_0, x_f, v_coeffs, w_coeffs, t_st, t_end, dt):
        jaco = np.zeros([x_0.shape[0], v_coeffs.shape[0] + w_coeffs.shape[0]])
        params = np.append(v_coeffs, w_coeffs, axis=0)

        for i in range(jaco.shape[1]):
            d = np.zeros(jaco.shape[1])
            d[i] = self.__ep
            n_params = params - d
            p_params = params + d

            # Numerical Integration
            x_f_n = self.__b_traj.compute_states(x_0, n_params[0:v_coeffs.shape[0]], n_params[v_coeffs.shape[0]:params.shape[0]], t_st, t_end, dt)[:, -1]
            x_f_p = self.__b_traj.compute_states(x_0, p_params[0:v_coeffs.shape[0]], p_params[v_coeffs.shape[0]:params.shape[0]], t_st, t_end, dt)[:, -1]

            # Differentiation
            diff_n = x_f - x_f_n
            diff_p = x_f - x_f_p
            jaco[:, i] = (diff_p - diff_n) / (2*self.__ep)

        return jaco

class BicycleModelTrajGenerator:

    def __init__(self, wheel_base):
        self.__wheel_base = wheel_base

    def compute_states(self, x_0, v_coeffs, w_coeffs, t_st, t_end, dt):
        # x : x_0[0], y : x_0[1], phi : x_0[2], v : x_0[3], w : x_0[4]
        t_vec = np.arange(t_st, t_end + dt, dt)
        v_prof = PolynomialCalculator(v_coeffs).calc(t_vec)
        w_prof = PolynomialCalculator(w_coeffs).calc(t_vec)
        
        states_prof = self.compute_pose_prof(x_0, v_prof, w_prof, t_vec, dt)
        
        return np.append(np.append(states_prof, np.array([v_prof]), axis=0), np.array([w_prof]), axis=0)

    def compute_pose_prof(self, x_0, v_prof, w_prof, t_vec, dt):
        num = t_vec.shape[0]
        pose = np.zeros([3, num])

        pose[0, 0] = x_0[0]
        pose[1, 0] = x_0[1]
        pose[2, 0] = x_0[2]

        for i in range(1, num):
            v = (v_prof[i - 1] + v_prof[i]) / 2.0
            w = (w_prof[i - 1] + w_prof[i]) / 2.0

            # Phi
            pose[2, i] = pose[2, i - 1] + dt * v / self.__wheel_base * math.tan(w)
            phi = (pose[2, i] + pose[2, i-1]) / 2.0

            # x, y
            pose[0, i] = pose[0, i - 1] + dt * v * math.cos(phi) 
            pose[1, i] = pose[1, i - 1] - dt * v * math.sin(phi)

        return pose            



