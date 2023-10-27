import numpy as np
import sim_utilities as sim_util
import pvfc_core.utilities as util
from pvfc_core.Planner import UpRampVelocityField

class Sim:
    def __init__(self, planner, controller, robot, ramp_force_params=None):
        self.planner = planner
        self.controller = controller
        self.robot = robot
        if ramp_force_params is not None: self.ramp_k, self.ramp_mu = ramp_force_params 
        
    def run(self, q, q_dot, q_r, q_r_dot, sim_time=10, dt=0.01):
        us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs, V_dots = ([] for i in range(9))  # initialize lists to store simulation data
        ts = np.arange(0, sim_time, dt)
        for t in ts:
            if self.planner.__class__.__name__ == UpRampVelocityField.__name__:
                q_T, q_dot_T = util.configToTask(q, q_dot, self.robot.dynamics.tool_length)
                f_e = sim_util.computeRampForce(self.ramp_k, self.ramp_mu, self.planner.p1, self.planner.p2, q_T, q_dot_T)
            else: f_e = np.zeros((2,1))
            qs.append(q.copy()), q_dots.append(q_dot.copy()), q_r_dots.append(q_r_dot.copy())  # save states before update
            V, V_dot = self.planner.step(q, q_dot)
            u, F, F_r, q_r, q_r_dot = self.controller.step(q, q_dot, q_r, q_r_dot, V, V_dot, dt)
            Vs.append(V.copy()), V_dots.append(V_dot.copy()), f_es.append(f_e.copy()), us.append(u.copy()), Fs.append(F.copy()), F_rs.append(F_r.copy())   
            q, q_dot = self.robot.step(q, q_dot, u, f_e, dt)  # update state
        return ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs, V_dots
    