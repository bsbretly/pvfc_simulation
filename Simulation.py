import numpy as np
import utilities as util

class Sim:
    def __init__(self, planner, controller, robot, ramp_force_params):
        self.planner = planner
        self.controller = controller
        self.robot = robot
        self.ramp_k, self.ramp_mu = ramp_force_params 
        
    def run(self, q, q_dot, q_r, q_r_dot, f_e, sim_time=10, dt=0.001):
        us, Fs, f_es, qs, q_dots, q_r_dots = ([] for i in range(6))
        ts = np.arange(0, sim_time, dt)
        for t in ts:
            # implement external force due to ramp interaction
            if self.planner.__class__.__name__ == 'UpRampVelocityField':
                q_T, q_dot_T = util.configToTask(q, q_dot, self.robot.dynamics.tool_length)
                f_e = util.computeRampForce(self.ramp_k, self.ramp_mu, self.planner.p1, self.planner.p2, q_T, q_dot_T)
            V, V_dot = self.planner.step(q, q_dot)
            u, u_attitude, F, q_r, q_r_dot = self.controller.step(q, q_dot, q_r, q_r_dot, V, V_dot, dt)
            q, q_dot = self.robot.step(q, q_dot, u, f_e, dt)
            f_es.append(f_e.copy())    
            us.append(u.copy()), qs.append(q.copy()), q_dots.append(q_dot.copy()), q_r_dots.append(q_r_dot.copy()), Fs.append(F.copy())
        return ts, us, Fs, f_es, qs, q_dots, q_r_dots
    