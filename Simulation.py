import numpy as np

class Sim:
    def __init__(self, planner, controller, robot):
        self.planner = planner
        self.controller = controller
        self.robot = robot
    def run(self, q, q_dot, q_r, q_r_dot, f_e, sim_time=10, dt=0.001):
        us, Fs, f_es, qs, q_dots, q_r_dots = ([] for i in range(6))
        ts = np.arange(0, sim_time, dt)
        for t in ts:
            V, V_dot = self.planner.step(q, q_dot)
            u, u_attitude, F, q_r, q_r_dot = self.controller.step(q, q_dot, q_r, q_r_dot, V, V_dot, dt)
            # if 10 <= t <= 12: 
            #     q, q_dot = self.robot.step(q, q_dot, u, f_e, dt)
            #     f_es.append(f_e.copy())
            # else: 
            q, q_dot = self.robot.step(q, q_dot, u, np.zeros_like(f_e), dt)
            f_es.append(np.zeros_like(f_e))
            us.append(u.copy()), qs.append(q.copy()), q_dots.append(q_dot.copy()), q_r_dots.append(q_r_dot.copy()), Fs.append(F.copy())
        return ts, us, Fs, f_es, qs, q_dots, q_r_dots
