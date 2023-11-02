import numpy as np
import algorithm_core.utilities as util

class QuadrotorDynamics:
    def __init__(self, robot_parameters, g=9.81):
        self.m_total, self.I = robot_parameters
        self.g = g
    def computeDynamics(self, q, q_dot):
        M = np.array([[self.m_total, 0, 0], 
                [0, self.m_total, 0], 
                [0, 0, self.I]])
        C = np.zeros((3,3))
        G = np.array([[0], [self.m_total*self.g], [0]])
        B = np.array([[np.sin(q[-1,0]), 0],
            [np.cos(q[-1,0]), 0],
            [0, 1]])
        return M, C, G, B
    
class QuadrotorTranslationalDynamics(QuadrotorDynamics):
    # Dynamics for the PVFC controller in translational space of the quadrotor
    def __init__(self, robot_parameters, g=9.81):
        super().__init__(robot_parameters, g)

    def computeDynamics(self, q, q_dot):
        M, C, G, B = super().computeDynamics(q, q_dot)
        M_trans = M[:2,:2]
        C_trans = C[:2,:2]
        return M_trans, C_trans


class AerialManipulatorDynamics:
    def __init__(self, robot_parameters, g=9.81):
        self.m, self.m_t, self.I, self.I_t, self.tool_length = robot_parameters
        self.g = g
        self.m_total = self.m + self.m_t
    def computeDynamics(self, q, q_dot):
        M = np.array([[self.m_total, 0, 0, -self.tool_length*self.m_t*np.sin(q[3,0])], 
            [0, self.m_total, 0, -self.tool_length*self.m_t*np.cos(q[3,0])], 
            [0, 0, self.I, 0], 
            [-self.tool_length*self.m_t*np.sin(q[3,0]), -self.tool_length*self.m_t*np.cos(q[3,0]), 0, self.I_t + self.tool_length**2*self.m_t]])
        C = np.array([[0, 0, 0, -self.tool_length*self.m_t*np.cos(q[3,0])*q_dot[3,0]], 
            [0, 0, 0, self.tool_length*self.m_t*np.sin(q[3,0])*q_dot[3,0]], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]]) 
        G = np.array([[0], 
            [self.g*(self.m_total)], 
            [0], 
            [-self.g*self.tool_length*self.m_t*np.cos(q[3,0])]])
        B = np.array([[np.sin(q[2,0]), 0, 0], 
            [np.cos(q[2,0]), 0, 0], 
            [0, 1, -1],
            [0, 0, 1]])
        return M, C, G, B


class AerialManipulatorTaskDynamics(AerialManipulatorDynamics):
    # Dynamics for the PVFC controller in task space of the aerial manipulator
    def __init__(self, robot_parameters):
        super().__init__(robot_parameters)
    def computeDynamics(self, q, q_dot):
        M, C, G, B = super().computeDynamics(q, q_dot)
        _, J, J_dot = util.computeTransforms(q, q_dot, self.tool_length)
        M_tilde = np.linalg.inv(J@np.linalg.inv(M)@J.T)
        C_tilde = np.linalg.inv(J@np.linalg.inv(M)@J.T)@((J@np.linalg.inv(M)@C - J_dot)@np.linalg.pinv(J))
        return M_tilde, C_tilde
    