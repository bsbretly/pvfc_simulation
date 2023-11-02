import numpy as np
import algorithm_core.utilities as util
from algorithm_core.Dynamics import QuadrotorDynamics, AerialManipulatorDynamics
from algorithm_core.params import quadrotor_params, AM_params

class Robot:
    def __init__(self, robot_parameters):
        if isinstance(robot_parameters, quadrotor_params): self.dynamics = QuadrotorDynamics(robot_parameters)
        elif isinstance(robot_parameters, AM_params): self.dynamics = AerialManipulatorDynamics(robot_parameters)
        else: raise NotImplementedError("Robot parameters not implemented for the robot.")

    def step(self, q, q_dot, u, F_e, dt):
        M, C, G, B = self.dynamics.computeDynamics(q, q_dot)

        q_ddot = np.linalg.inv(M)@(B@u + F_e - G - C@q_dot)
        q_dot += q_ddot*dt 
        q += q_dot*dt 
        return q, q_dot


class Quadrotor(Robot):
    def __init__(self, robot_params):
        super().__init__(robot_params)

    def step(self, q, q_dot, u, F_e, dt):
        F_e = np.vstack((F_e, 0))
        return super().step(q, q_dot, u, F_e, dt)
        
    
class AerialManipulator(Robot):
    def __init__(self, robot_params):
        super().__init__(robot_params)

    def step(self, q, q_dot, u, F_e, dt):
        _, J, _ = util.computeTransforms(q, q_dot, self.dynamics.tool_length)
        F_e = J.T@F_e
        return super().step(q, q_dot, u, F_e, dt)
    
