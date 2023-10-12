import numpy as np
import utilities as util
from Dynamics import QuadrotorTranslationalDynamics, AerialManipulatorTaskDynamics
from params import quadrotor_params, AM_params

class BaseControl:
    def __init__(self, robot_params, attitude_control_params):
        if isinstance(robot_params, quadrotor_params): self.dynamics = QuadrotorTranslationalDynamics(robot_params)
        elif isinstance(robot_params, AM_params): self.dynamics = AerialManipulatorTaskDynamics(robot_params)
        else: raise NotImplementedError("Robot parameters not implemented for the controller.")
        self.theta_K_p, self.theta_K_d = attitude_control_params

    def decomposeThrust(self, tau):
        Lambda = tau[:2]
        Lambda += np.array([[0., self.dynamics.m_total*self.dynamics.g]]).T  # gravity compensation
        thrust, theta_d = util.decomposeThrustVector(Lambda)
        return np.array([thrust, theta_d]).T
        
    def computeAttitudeControl(self, q, q_dot, tau, task_space=False):
        thrust, theta_d = self.decomposeThrust(tau)
        tau_theta_d = self.dynamics.I*(self.theta_K_p*(theta_d - q[2,0]) + self.theta_K_d*(0. - q_dot[2,0]))
        if task_space: return np.array([[thrust, tau_theta_d + tau[-1,0], tau[-1,0]]]).T
        else: return np.array([[thrust, tau_theta_d]]).T
    
    def computeConfigForceControl(self, q, q_dot, F_T):
        # Force-based control
        _, J, _ = util.computeTransforms(q, q_dot, self.dynamics.tool_length)
        '''
        input: 
            config space state and task space control force - q, q_dot, F
        output: 
            config space force command - tau
        '''
        #TODO: implement force-based redundant task space control from https://journals.sagepub.com/doi/abs/10.1177/0278364908091463
        return J.T@F_T

    def computeConfigAccControl(self, q, q_dot, a_x):
        # Acceleration-based control
        _, J, J_dot = util.computeTransforms(q, q_dot, self.dynamics.tool_length)
        '''
        input: 
            velocity, jacobians, end-effector acceleration command - q_dot, J, J_dot, a_x
        output: 
            config space acceleration command - tau
        '''
        a_q = np.linalg.pinv(J)@(a_x - J_dot@q_dot)
        M, C, G, _ = self.computeDynamics(q, q_dot)
        return M@a_q + C@q_dot + G
    
    def computeDynamics(self, q, q_dot):
        return self.dynamics.computeDynamics(q, q_dot)
    
    def step(self, q, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, dt):
        M, C = self.computeDynamics(q, q_dot)
        if self.dynamics.__class__.__name__==util.RobotInfo.AM.value.dynamics_type:  # AM in task space
            _, q_T_dot = util.configToTask(q, q_dot, self.dynamics.tool_length)
            F_T, f_r, q_r, q_r_dot = self.computeControlAction(q_T_dot, q_r, q_r_dot, V_bar, V_bar_dot, M, C, dt)
            F = self.computeConfigForceControl(q, q_dot, F_T)
            u = self.computeAttitudeControl(q, q_dot, F, task_space=True)
            return u, F_T, f_r, q_r, q_r_dot
        elif self.dynamics.__class__.__name__==util.RobotInfo.QUAD.value.dynamics_type:  # quad in configuration space
            F, f_r, q_r, q_r_dot = self.computeControlAction(q_dot[:-1,:], q_r, q_r_dot, V_bar, V_bar_dot, M, C, dt)
            u = self.computeAttitudeControl(q, q_dot, F, task_space=False)
            return u, F, f_r, q_r, q_r_dot
        else:    
            raise NotImplementedError("Dynamics not defined for the controller.")
        
    def computeControlAction(self, *args, **kwargs):
        raise NotImplementedError("computeControlAction() method not implemented.")
    

class PassiveBaseControl(BaseControl):
    def __init__(self, robot_params, attitude_control_params, *args):
        super().__init__(robot_params, attitude_control_params)
        self.m_r, self.E_bar, self.gamma = args[0]

    def updateReservoirState(self, tau_r, q_r, q_r_dot, dt):        
        q_r_ddot = tau_r / self.m_r
        q_r_dot += q_r_ddot*dt
        q_r += q_r_dot*dt
        return q_r, q_r_dot
    
    def computeDynamics(self, q, q_dot):
        M, C = super().computeDynamics(q, q_dot)
        return util.augmentDynamics(M, C, self.m_r)


class PDControl(BaseControl):
    def __init__(self, robot_params, attitude_control_params, *args):
        super().__init__(robot_params, attitude_control_params)
        self.K_p, self.K_d = args[0]

    def computeControlAction(self, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, M, C, dt):
        ''' 
        input: 
            robot state, reservoir state, desired velocity field, dynamics, time step
        output: 
            control action to the robot, reservoir and updated reservoir state, - tau, tau_r, q_r, q_r_dot
        '''       
        tau = M@V_bar_dot[:2] + C@V_bar[:2] - self.K_p*(q_dot - V_bar[:2])
        tau_r = np.array(0.)  # no reservoir control
        return tau, tau_r, q_r, q_r_dot


class AugmentedPDControl(PassiveBaseControl):
    def __init__(self, robot_params, attitude_control_params, *args):
        super().__init__(robot_params, attitude_control_params, args[0])
        self.K_p, self.K_d = args[1]

    def computeControlAction(self, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, M, C, dt):
        qbar_dot = np.vstack([q_dot, q_r_dot])
        tau_bar = M@V_bar_dot + C@V_bar + self.K_p*(V_bar - qbar_dot)
        q_r, q_r_dot = self.updateReservoirState(tau_bar[-1,-1], q_r, q_r_dot, dt)
        return tau_bar[:2,:], tau_bar[-1,-1], q_r, q_r_dot
 

class PVFC(PassiveBaseControl):
    def __init__(self, robot_parameters, attitude_control_params, *args):
        super().__init__(robot_parameters, attitude_control_params, *args)

    def computeControlAction(self, q_dot, q_r, q_r_dot, V_bar, V_bar_dot, M_bar, C_bar, dt):
        ''' 
        input: 
            robot state, reservoir state, desired velocity field, dynamics, time step
        output: 
            control action to the robot, reservoir and updated reservoir state, - tau, tau_r, q_r, q_r_dot
        '''        
        qbar_dot = np.vstack([q_dot, q_r_dot])
        w_bar = M_bar@(V_bar_dot) + C_bar@(V_bar)
        p_bar = M_bar@(qbar_dot)
        P_bar = M_bar@(V_bar)
        tau_c = ((w_bar@P_bar.T - P_bar@w_bar.T)@(qbar_dot)) / (2 * self.E_bar)
        tau_f = self.gamma*(P_bar@p_bar.T - p_bar@P_bar.T)@(qbar_dot)
        tau_bar = tau_c + tau_f
        q_r, q_r_dot = self.updateReservoirState(tau_bar[-1,-1], q_r, q_r_dot, dt)
        return tau_bar[:2,:], tau_bar[-1,-1], q_r, q_r_dot
    