import numpy as np
import utilities as util

class Sim:
    def __init__(self, planner, controller, robot):
        self.planner = planner
        self.controller = controller
        self.robot = robot
        self.k = .5 # surface interaction constant
    def run(self, q, q_dot, q_r, q_r_dot, f_e, sim_time=10, dt=0.001):
        us, Fs, f_es, qs, q_dots, q_r_dots = ([] for i in range(6))
        ts = np.arange(0, sim_time, dt)
        for t in ts:
            f_e = np.zeros((2,1))
            # implement external force due to ramp interaction
            if type(self.planner) is tuple:
                q_T, q_T_dot = util.configToTask(q, q_dot, self.robot.dynamics.tool_length)
                if util.contactFloor(self.planner[1].x_s, self.planner[1].x_e, q_T):
                    path_planner = self.planner[0]
                    f_e = util.computeFloorForce(self.k) 
                elif util.contactRamp(self.planner[1].x_s, self.planner[1].x_e, self.planner[1].z_h, q_T):
                    path_planner = self.planner[1]
                    f_e = util.computeRampForce(self.k, self.planner[1].x_s, self.planner[1].x_e, self.planner[1].z_h, q_T)
                else: 
                    path_planner = self.planner[0]
            else: path_planner = self.planner
            V, V_dot = path_planner.step(q, q_dot)
            u, u_attitude, F, q_r, q_r_dot = self.controller.step(q, q_dot, q_r, q_r_dot, V, V_dot, dt)
            q, q_dot = self.robot.step(q, q_dot, u, f_e, dt)
            f_es.append(f_e.copy())    
            us.append(u.copy()), qs.append(q.copy()), q_dots.append(q_dot.copy()), q_r_dots.append(q_r_dot.copy()), Fs.append(F.copy())
        return ts, us, Fs, f_es, qs, q_dots, q_r_dots
