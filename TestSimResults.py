import unittest
from numpy.testing import assert_array_equal
from main import run_sim
import pickle
import utilities as util
import numpy as np
from collections import defaultdict

class TestSimResults(unittest.TestCase):
    """
    Test cases for validating simulation results: The simulation results are compared against 
    results from a previous run of the simulation stored in a pickle file.
    """
    TEST_DATA_FILE = 'data/tracking_control_comparison_data.pickle'
    
    @classmethod
    def setUpClass(cls):
        with open(cls.TEST_DATA_FILE, 'rb') as file:
            cls.sim_data = pickle.load(file)
        
        cls.robot_type = util.RobotInfo.AM
        cls.planner_type = util.PlannerInfo.HORIZONTAL
        cls.controller_types = [util.ControllerInfo.PVFC, util.ControllerInfo.AUGMENTEDPD]
        sim_time = np.rint(cls.sim_data['ts'][cls.controller_types[0]][-1])
        dt = sim_time / len(cls.sim_data['ts'][cls.controller_types[0]])
        
        cls.sim_results = defaultdict(list)
        data_keys = ['ts', 'us', 'Fs', 'F_rs', 'f_es', 'qs', 'q_dots', 'q_r_dots', 'Vs']

        for controller_type in cls.controller_types:
            results = run_sim(
                robot_type=cls.robot_type, 
                planner_type=cls.planner_type, 
                controller_type=controller_type, 
                sim_time=sim_time, 
                dt=dt, 
                plot=False, 
                return_data=True
            )
            for key, value in zip(data_keys, results):
                cls.sim_results[key].append(value.copy())

    def test_ts_identical(self):
        self._compare_arrays('ts')
    
    def test_us_identical(self):
        self._compare_arrays('us')

    def test_Fs_identical(self):
        self._compare_arrays('Fs')

    def test_F_rs_identical(self):
        self._compare_arrays('F_rs')
    
    def test_f_es_identical(self):
        self._compare_arrays('f_es')

    def test_qs_identical(self):
        self._compare_arrays('qs')
    
    def test_q_dots_identical(self):
        self._compare_arrays('q_dots')
    
    def test_q_r_dots_identical(self):
        self._compare_arrays('q_r_dots')

    def test_Vs_identical(self):
        self._compare_arrays('Vs')

    def _compare_arrays(self, key):
        for controller_type in self.controller_types:
            assert_array_equal(
                self.sim_data[key][controller_type], 
                self.sim_results[key][self.controller_types.index(controller_type)], 
                err_msg=f"Mismatch in '{key}'"
            )





# class TestSimResults(unittest.TestCase):
#     """
#     Test cases for validating simulation results: The simulation results are compared against the results from a previous run of the simulation stored in a pickle file. The pickle file contains the simulation results for the controller types defined below.
#     """
#     DATA_FILE = 'data/tracking_control_comparison_data.pickle'

#     @classmethod
#     def setUpClass(cls):
#         with open('data/'+ cls.DATA_FILE, 'rb') as file:
#             cls.sim_data = pickle.load(file)
#         cls.robot_type = util.RobotInfo.AM
#         cls.planner_type = util.PlannerInfo.HORIZONTAL
#         cls.controller_types = [util.ControllerInfo.PVFC, util.ControllerInfo.AUGMENTEDPD]
#         sim_time = np.rint(cls.sim_data['ts'][cls.controller_types[0]][-1])
#         dt = sim_time / len(cls.sim_data['ts'][cls.controller_types[0]])
#         plot = False 
#         return_data = True
#         cls.ts, cls.us, cls.Fs, cls.F_rs, cls.f_es, cls.qs, cls.q_dots, cls.q_r_dots, cls.Vs = ([] for _ in range(9))
#         for controller_type in cls.controller_types:
#             ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs = run_sim(robot_type=cls.robot_type, planner_type=cls.planner_type, controller_type=controller_type, sim_time=sim_time, dt=dt, plot=plot, return_data=return_data)
#             cls.ts.append(ts.copy()), cls.us.append(us.copy()), cls.Fs.append(Fs.copy()), cls.F_rs.append(F_rs.copy()), cls.f_es.append(f_es.copy()), cls.qs.append(qs.copy()), cls.q_dots.append(q_dots.copy()), cls.q_r_dots.append(q_r_dots.copy()), cls.Vs.append(Vs.copy())
        
#     # @classmethod
#     # def run_simulation(cls, robot_type, planner_type, controller_type, sim_time, dt, plot, return_data):
#     #     return run_sim(cls.robot_type, cls.planner_type, controller_type, sim_time=sim_time, plot=plot, return_data=return_data)
    
#     def test_ts_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['ts'][controller_type], self.ts[self.controller_types.index(controller_type)], err_msg="Mismatch in 'ts'")

#     def test_us_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['us'][controller_type], self.us[self.controller_types.index(controller_type)], err_msg="Mismatch in 'us'")

#     def test_Fs_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['Fs'][controller_type], self.Fs[self.controller_types.index(controller_type)], err_msg="Mismatch in 'Fs'")

#     def test_F_rs_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['F_rs'][controller_type], self.F_rs[self.controller_types.index(controller_type)], err_msg="Mismatch in 'F_rs'")

#     def test_f_es_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['f_es'][controller_type], self.f_es[self.controller_types.index(controller_type)], err_msg="Mismatch in 'f_es'")

#     def test_qs_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['qs'][controller_type], self.qs[self.controller_types.index(controller_type)], err_msg="Mismatch in 'qs'")

#     def test_q_dots_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['q_dots'][controller_type], self.q_dots[self.controller_types.index(controller_type)], err_msg="Mismatch in 'q_dots'")

#     def test_q_r_dots_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['q_r_dots'][controller_type], self.q_r_dots[self.controller_types.index(controller_type)], err_msg="Mismatch in 'q_r_dots'")
    
#     def test_Vs_identical(self):
#         for controller_type in self.controller_types:
#             assert_array_equal(self.sim_data['Vs'][controller_type], self.Vs[self.controller_types.index(controller_type)], err_msg="Mismatch in 'Vs'")


if __name__ == '__main__':
    unittest.main()
