import numpy as np
import utilities as util
from Dynamics import QuadrotorTranslationalDynamics, AerialManipulatorTaskDynamics


class BaseControl:
    def __init__(self, robot_params, attitude_control_params):
        if type(robot_params).__name__ == 'quadrotor_params': self.dynamics = QuadrotorTranslationalDynamics(robot_params)
        elif type(robot_params).__name__ == 'AM_params': self.dynamics = AerialManipulatorTaskDynamics(robot_params)
        else: raise NotImplementedError("Robot parameters not implemented for the controller.")
        self.theta_K_p, self.theta_K_d = attitude_control_params

    def computeDesiredAttitude(self, tau):
        # thrust decomposition
        Lambda = tau[:2]
        # gravity compensation
        Lambda += np.array([[0., self.dynamics.m_total*self.dynamics.g]]).T
        thrust, theta_d = util.decomposeThrustVector(Lambda)
        return np.array([thrust, theta_d]).T
        
    def computAttitudeControl(self, q, q_dot, u_attitude):
        thrust, theta_d = u_attitude[0], u_attitude[1]
        tau_theta_d = self.dynamics.I*(self.theta_K_p*(theta_d - q[2,0]) + self.theta_K_d*(0. - q_dot[2,0]))
        return np.array([[thrust, tau_theta_d]]).T
    
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
    
    def computeDynamics(self, q, q_dot):
        return self.dynamics.computeDynamics(q, q_dot)
    
    def step(self, *args, **kwargs):
        raise NotImplementedError("step() method not implemented.")
    
    def computeControlAction(self, *args, **kwargs):
        raise NotImplementedError("computeControlAction() method not implemented.")
        

class PDControl(BaseControl):
    def __init__(self, robot_params, attitude_control_params, pd_control_params):
        super().__init__(robot_params, attitude_control_params)
        self.K_p, self.K_d = pd_control_params

    def computeControlAction(self, q_T_dot, V_bar, V_bar_dot, M, C):
        return M@V_bar_dot[:2] + C@V_bar[:2] - self.K_d*(q_T_dot - V_bar[:2])
    
    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        M, C = self.computeDynamics(q, q_dot)
        _, q_T_dot = util.configToTask(q, q_dot, self.dynamics.tool_length)
        F = self.computeControlAction(q_T_dot, V_bar, V_bar_dot, M, C)
        _, J, _ = util.computeTransforms(q, q_dot, self.dynamics.tool_length)
        tau = self.computeConfigForceControl(F, J)
        u_attitude = self.computeDesiredAttitude(tau)
        u = self.computAttitudeControl(q, q_dot, u_attitude)
        return u, u_attitude, F, 0, 0  # return hard-coded q_r, q_r_dot for now
    

class PVFC(BaseControl):
    def __init__(self, robot_parameters, attitude_control_params, pvfc_params):
        super().__init__(robot_parameters, attitude_control_params)
        self.m_r, self.E_bar, self.gamma = pvfc_params

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
    
    def computeAndAugmentDynamics(self, q, q_dot):
        M, C = self.computeDynamics(q, q_dot)
        M_bar, C_bar = util.augmentDynamics(M, C, self.m_r)
        return M_bar, C_bar
        
        
class TranslationalPVFC(PVFC):
    def __init__(self, robot_parameters, controller_parameters):
        super().__init__(robot_parameters, controller_parameters)

    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        M_bar, C_bar = self.computeAndAugmentDynamics(q, q_dot)
        q_trans_dot = q_dot[:2,:]
        F = self.computeControlAction(q_trans_dot, q_r_dot, V_bar, V_bar_dot, M_bar, C_bar)
        q_r, q_r_dot = self.updateReservoirState(F, q_r, q_r_dot, dt)
        u_attitude = self.computeDesiredAttitude(F)
        u = self.computAttitudeControl(q, q_dot, u_attitude)
        return u, u_attitude, F, q_r, q_r_dot


class TaskPVFC(PVFC):
    def __init__(self, robot_parameters, attitude_control_params, pvfc_parameters):
        super().__init__(robot_parameters, attitude_control_params, pvfc_parameters)

    def computAttitudeControl(self, q, q_dot, u_attitude, tau):
        u = super().computAttitudeControl(q, q_dot, u_attitude)
        return np.array([[u[0,0], u[1,0] + tau[-1,0], tau[-1,0]]]).T
        
    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        M_bar, C_bar = self.computeAndAugmentDynamics(q, q_dot)
        _, q_T_dot = util.configToTask(q, q_dot, self.dynamics.tool_length)
        F = self.computeControlAction(q_T_dot, q_r_dot, V_bar, V_bar_dot, M_bar, C_bar)
        q_r, q_r_dot = self.updateReservoirState(F, q_r, q_r_dot, dt)
        _, J, _ = util.computeTransforms(q, q_dot, self.dynamics.tool_length)
        tau = self.computeConfigForceControl(F[:-1,:], J)
        u_attitude = self.computeDesiredAttitude(tau)
        u = self.computAttitudeControl(q, q_dot, u_attitude, tau)
        return u, u_attitude, F, q_r, q_r_dot
    
    
