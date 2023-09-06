import numpy as np
import utilities as util
from Dynamics import QuadrotorTranslationalDynamics, AerialManipulatorTaskDynamics


class PVFC:
    def __init__(self, robot_parameters, controller_parameters):
        if type(robot_parameters).__name__ == 'QuadrotorParams': self.dynamics = QuadrotorTranslationalDynamics(robot_parameters)
        elif type(robot_parameters).__name__ == 'AMParams': self.dynamics = AerialManipulatorTaskDynamics(robot_parameters)
        else: raise NotImplementedError("Robot parameters not implemented for the controller.")
        self.m_r, self.E_bar, self.gamma, self.theta_K_p, self.theta_K_d = controller_parameters

    def computeControlAction(self, q_dot, q_r_dot, V_bar, V_bar_dot, M_bar, C_bar):
        ''' 
        input: 
            robot velocity, reservoir position,velocity, velocity field,field gradient, time step. augmented dynamics - q_dot, q_r, q_r_dot, Vbar, Vbar_dot, dt, M_bar, C_bar
        output: 
            PVFC action to the robot, - tau_c + tau_f
        '''        
        qbar_dot = np.vstack([q_dot, q_r_dot])
        w_bar = M_bar@(V_bar_dot) + C_bar@(V_bar)
        p_bar = M_bar@(qbar_dot)
        P_bar = M_bar@(V_bar)
        tau_c = ((w_bar@P_bar.T - P_bar@w_bar.T)@(qbar_dot)) / (2 * self.E_bar)
        tau_f = self.gamma*(P_bar@p_bar.T - p_bar@P_bar.T)@(qbar_dot)
        return tau_c + tau_f

    def updateReservoirState(self, F, q_r, q_r_dot, dt):
        q_r_ddot = F[-1,-1] / self.m_r
        q_r_dot += q_r_ddot*dt
        q_r += q_r_dot*dt
        return q_r, q_r_dot
    
    def computeDesiredAttitude(self, tau, q, q_dot):
        # thrust decomposition
        Lambda = tau[:2]
        # gravity compensation
        Lambda += np.array([[0., self.dynamics.m_total*self.dynamics.g]]).T
        thrust, theta_d = util.decomposeThrustVector(Lambda)
        return np.array([thrust, theta_d]).T
    
    def computeAndAugmentDynamics(self, q, q_dot):
        M, C = self.dynamics.computeDynamics(q, q_dot)
        M_bar, C_bar = util.augmentDynamics(M, C, self.m_r)
        return M_bar, C_bar
    
    def computAttitudeControl(self, q, q_dot, thrust, theta_d):
        raise NotImplementedError("PVFC computAttitudeControl function not implemented for the controller.")
    
    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        raise NotImplementedError("PVFC step function not implemented for the controller.")
        
        
class TranslationalPVFC(PVFC):
    def __init__(self, robot_parameters, controller_parameters):
        super().__init__(robot_parameters, controller_parameters)

    def computAttitudeControl(self, q, q_dot, u_attitude):
        thrust, theta_d = u_attitude[0], u_attitude[1]
        tau_theta = self.dynamics.I*(self.theta_K_p*(theta_d - q[2,0]) + self.theta_K_d*(0. - q_dot[2,0]))
        return np.array([[thrust, tau_theta]]).T

    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        M_bar, C_bar = self.computeAndAugmentDynamics(q, q_dot)
        q_trans_dot = q_dot[:2,:]
        F = self.computeControlAction(q_trans_dot, q_r_dot, V_bar, V_bar_dot, M_bar, C_bar)
        q_r, q_r_dot = self.updateReservoirState(F, q_r, q_r_dot, dt)
        u_attitude = self.computeDesiredAttitude(F, q, q_dot)
        u = self.computAttitudeControl(q, q_dot, u_attitude)
        return u, u_attitude, F, q_r, q_r_dot


class TaskPVFC(PVFC):
    def __init__(self, robot_parameters, controller_parameters):
        super().__init__(robot_parameters, controller_parameters)

    def computAttitudeControl(self, q, q_dot, u_attitude, tau):
        thrust, theta_d = u_attitude[0], u_attitude[1]
        tau_theta_d = self.dynamics.I*(self.theta_K_p*(theta_d - q[2,0]) + self.theta_K_d*(0. - q_dot[2,0]))
        return np.array([[thrust, tau_theta_d + tau[-1,0], tau[-1,0]]]).T
        
    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        M_bar, C_bar = self.computeAndAugmentDynamics(q, q_dot)
        _, q_T_dot = util.configToTask(q, q_dot, self.dynamics.tool_length)
        F = self.computeControlAction(q_T_dot, q_r_dot, V_bar, V_bar_dot, M_bar, C_bar)
        q_r, q_r_dot = self.updateReservoirState(F, q_r, q_r_dot, dt)
        _, J, _ = util.computeTransforms(q, q_dot, self.dynamics.tool_length)
        tau = self.computeConfigForceControl(F[:-1,:], J)
        u_attitude = self.computeDesiredAttitude(tau, q, q_dot)
        u = self.computAttitudeControl(q, q_dot, u_attitude, tau)
        return u, u_attitude, F, q_r, q_r_dot
    
    def computeConfigForceControl(self, F, J):
        # Force-based control
        '''
        input: 
            Task space control force and AM jacobian - F, J
        output: 
            config space force command - tau
        '''
        #TODO: implement force-based redundant task space control from https://journals.sagepub.com/doi/abs/10.1177/0278364908091463
        return J.T@F

    def computeConfigAccControl(self, q, q_dot, J, J_dot, a_x):
        # Acceleration-based control
        '''
        input: 
            velocity, jacobians, end-effector acceleration command - q_dot, J, J_dot, a_x
        output: 
            config space acceleration command - tau
        '''
        a_q = np.linalg.pinv(J)@(a_x - J_dot@q_dot)
        M, C, G, B = self.dynamics.computeDynamics(q, q_dot)
        return M@a_q + C@q_dot + G
