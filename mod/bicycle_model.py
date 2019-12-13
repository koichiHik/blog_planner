
import numpy as np
import math
import copy

def pi_2_npi(val):
    return np.arctan2(np.sin(val), np.cos(val))

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
        jaco = np.zeros([x_0.shape[0] + x_f.shape[0], v_coeffs.shape[0] + w_coeffs.shape[0]])
        params = np.append(v_coeffs, w_coeffs, axis=0)
        x_tgt = np.append(x_0, x_f, axis=0)

        for i in range(jaco.shape[1]):
            d = np.zeros(jaco.shape[1])
            d[i] = self.__ep
            n_params = params - d
            p_params = params + d

            # Numerical Integration
            states = self.__b_traj.compute_states(x_0, n_params[0:v_coeffs.shape[0]], n_params[v_coeffs.shape[0]:params.shape[0]], t_st, t_end, dt)
            x_f_n = np.append(states[:, 0], states[:, -1], axis=0)
            states = self.__b_traj.compute_states(x_0, p_params[0:v_coeffs.shape[0]], p_params[v_coeffs.shape[0]:params.shape[0]], t_st, t_end, dt) 
            x_f_p = np.append(states[:, 0], states[:, -1], axis=0)

            # Differentiation
            diff_n = x_tgt - x_f_n
            diff_p = x_tgt - x_f_p
            diff = diff_p - diff_n

            diff[2] = pi_2_npi(diff[2])
            diff[4] = pi_2_npi(diff[4])
            diff[7] = pi_2_npi(diff[7])
            diff[9] = pi_2_npi(diff[9])

            jaco[:, i] = diff / (2*self.__ep)

        return jaco

    def calc_jacobian_wo_velocity(self, x_0, x_f, v_coeffs, w_coeffs, t_st, t_end, dt):
        jaco = np.zeros([x_0.shape[0] + x_f.shape[0], v_coeffs.shape[0] + w_coeffs.shape[0]])
        params = np.append(v_coeffs, w_coeffs, axis=0)
        x_tgt = np.append(x_0, x_f, axis=0)

        for i in range(jaco.shape[1]):
            d = np.zeros(jaco.shape[1])
            d[i] = self.__ep
            n_params = params - d
            p_params = params + d

            # Numerical Integration
            states = self.__b_traj.compute_poses(x_0, n_params[0:v_coeffs.shape[0]], n_params[v_coeffs.shape[0]:params.shape[0]], t_st, t_end, dt)
            x_f_n = np.append(states[:, 0], states[:, -1], axis=0)
            states = self.__b_traj.compute_poses(x_0, p_params[0:v_coeffs.shape[0]], p_params[v_coeffs.shape[0]:params.shape[0]], t_st, t_end, dt) 
            x_f_p = np.append(states[:, 0], states[:, -1], axis=0)

            # Differentiation
            diff_n = x_tgt - x_f_n
            diff_p = x_tgt - x_f_p
            jaco[:, i] = (diff_p - diff_n) / (2*self.__ep)

        return jaco


class BicycleModelTrajGenerator:

    def __init__(self, wheel_base):
        self.__wheel_base = wheel_base
        self.__jacobian_calc = JacobianCalculator(self)

    def compute_states(self, x_0, v_coeffs, w_coeffs, t_st, t_end, dt):
        # x : x_0[0], y : x_0[1], phi : x_0[2], v : x_0[3], w : x_0[4]
        t_vec = np.arange(t_st, t_end + dt, dt)
        v_prof = PolynomialCalculator(v_coeffs).calc(t_vec)
        w_prof = PolynomialCalculator(w_coeffs).calc(t_vec)
        
        print(w_prof / math.pi * 180)

        states_prof = self.compute_pose_prof(x_0, v_prof, w_prof, t_vec, dt)
        
        return np.append(np.append(states_prof, np.array([v_prof]), axis=0), np.array([w_prof]), axis=0)

    def compute_poses(self, x_0, v_coeffs, w_coeffs, t_st, t_end, dt):
        # x : x_0[0], y : x_0[1], phi : x_0[2], v : x_0[3], w : x_0[4]
        t_vec = np.arange(t_st, t_end + dt, dt)
        v_prof = PolynomialCalculator(v_coeffs).calc(t_vec)
        w_prof = PolynomialCalculator(w_coeffs).calc(t_vec)
        
        states_prof = self.compute_pose_prof(x_0, v_prof, w_prof, t_vec, dt)
        
        return states_prof

    def compute_pose_prof(self, x_0, v_prof, w_prof, t_vec, dt):
        num = t_vec.shape[0]
        pose = np.zeros([3, num])

        pose[0, 0] = x_0[0]
        pose[1, 0] = x_0[1]
        pose[2, 0] = pi_2_npi(x_0[2])

        for i in range(1, num):
            v = (v_prof[i - 1] + v_prof[i]) / 2.0
            w = (w_prof[i - 1] + w_prof[i]) / 2.0

            # Phi
            pose[2, i] = pi_2_npi(pose[2, i - 1] + dt * v * math.tan(w) / self.__wheel_base)
            #phi = (pose[2, i] + pose[2, i-1]) / 2.0
            #phi = pi_2_npi(phi)
            phi = pose[2, i]

            # x, y
            pose[0, i] = pose[0, i - 1] + dt * v * math.cos(phi) 
            pose[1, i] = pose[1, i - 1] + dt * v * math.sin(phi)

        return pose            

    def generate_trajectory(self, x_0, x_f, v_coeffs_0, w_coeffs_0, t_st, t_end, dt):

        params = np.append(v_coeffs_0, w_coeffs_0, axis=0)
        v_coeffs = copy.copy(v_coeffs_0)
        w_coeffs = copy.copy(w_coeffs_0)

        for i in range(20):

            v_coeffs = params[0:v_coeffs_0.shape[0]]
            w_coeffs = params[v_coeffs_0.shape[0]:]            

            # Forward Integration.
            x_tf = self.compute_states(x_0, v_coeffs, w_coeffs, t_st, t_end, dt)

            # Compute Jacobian via Linearization.        
            jaco = self.__jacobian_calc.calc_jacobian(x_0, x_f, v_coeffs, w_coeffs, t_st, t_end, dt)
            
            # Compute Parameter Correction Vector via Newton Methods
            dx = np.append(x_0, x_f, axis=0) - np.append(x_tf[:,0], x_tf[:,-1], axis=0)
            dx[2] = pi_2_npi(dx[2])
            dx[7] = pi_2_npi(dx[7])
            dx[4] = pi_2_npi(dx[4])
            dx[9] = pi_2_npi(dx[9])            

            jaco_pseudo_inv = np.linalg.pinv(jaco)
            dp = np.matmul(-jaco_pseudo_inv, dx)
            params = params + dp

        # Compute Result with Final Parameter Vector.
        x_tf = self.compute_states(x_0, v_coeffs, w_coeffs, t_st, t_end, dt)

        if (np.linalg.norm(x_tf[:, 0] - x_0) < 0.0001 and np.linalg.norm(x_tf[:, -1] - x_f) < 0.0001):
            return x_tf
        else:
            return None


    def generate_trajectory_wo_vel_constraint(self, x_0, x_f, v_coeffs_0, w_coeffs_0, t_st, t_end, dt):

        params = np.append(v_coeffs_0, w_coeffs_0, axis=0)
        v_coeffs = copy.copy(v_coeffs_0)
        w_coeffs = copy.copy(w_coeffs_0)

        for i in range(20):

            v_coeffs = params[0:v_coeffs_0.shape[0]]
            w_coeffs = params[v_coeffs_0.shape[0]:]            

            # Forward Integration.
            x_tf = self.compute_poses(x_0, v_coeffs, w_coeffs, t_st, t_end, dt)

            # Compute Jacobian via Linearization.        
            jaco = self.__jacobian_calc.calc_jacobian_wo_velocity(x_0, x_f, v_coeffs, w_coeffs, t_st, t_end, dt)
            
            # Compute Parameter Correction Vector via Newton Methods
            dx = np.append(x_0, x_f, axis=0) - np.append(x_tf[:,0], x_tf[:,-1], axis=0)
            jaco_pseudo_inv = np.linalg.pinv(jaco)
            dp = np.matmul(-jaco_pseudo_inv, dx)
            params = params + dp

        # Compute Result with Final Parameter Vector.
        x_tf = self.compute_poses(x_0, v_coeffs, w_coeffs, t_st, t_end, dt)

        if (np.linalg.norm(x_tf[:, 0] - x_0) < 0.0001 and np.linalg.norm(x_tf[:, -1] - x_f) < 0.0001):
            return x_tf
        else:
            return None




