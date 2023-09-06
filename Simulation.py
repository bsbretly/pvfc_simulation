import numpy as np
import utilities as util

class Sim:
    def __init__(self, planner, controller, robot):
        self.planner = planner
        self.controller = controller
        self.robot = robot
    def run(self, q, q_dot, q_r, q_r_dot, f_e, sim_time=10, dt=0.001):
        us, Fs, f_es, qs, q_dots, q_r_dots = ([] for i in range(6))
        ts = np.arange(0, sim_time, dt)
        for t in ts:
            f_e = np.zeros((2,1))
            # implement external force due to ramp interaction
            if type(self.planner) is tuple:
                if util.touchRamp(self.planner[1].x_s, self.planner[1].x_e, self.planner[1].z_h, q[0,0], q[1,0]):
                    m = self.planner[1].z_h / (self.planner[1].x_e - self.planner[1].x_s)
                    b = -m * self.planner[1].x_s
                    Delta = np.abs(-m*q[0] + q[1] - b) / np.sqrt(m**2 + 1)
                    n_hat = np.array([[-m], [1.]]) / np.sqrt(1 + m**2)
                    f_e = 1. * Delta * n_hat
                    V, V_dot = self.planner[1].step(q, q_dot)
                else: V, V_dot = self.planner[0].step(q, q_dot)
            else: V, V_dot = self.planner.step(q, q_dot)
            u, u_attitude, F, q_r, q_r_dot = self.controller.step(q, q_dot, q_r, q_r_dot, V, V_dot, dt)
            q, q_dot = self.robot.step(q, q_dot, u, f_e, dt)
            f_es.append(f_e.copy())    
            us.append(u.copy()), qs.append(q.copy()), q_dots.append(q_dot.copy()), q_r_dots.append(q_r_dot.copy()), Fs.append(F.copy())
        return ts, us, Fs, f_es, qs, q_dots, q_r_dots
