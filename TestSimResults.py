import unittest
from numpy.testing import assert_array_equal
from main import run_sim
import pickle
import utilities as util

class TestSimResults(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open('data/'+ 'tracking_control_comparison_data.pickle', 'rb') as file:
            cls.sim_data = pickle.load(file)
        cls.robot_type = util.RobotInfo.AM
        cls.planner_type = util.PlannerInfo.HORIZONTAL
        cls.controller_type = util.ControllerInfo.PVFC
        sim_time = 60
        plot = False 
        return_data = True
        cls.ts, cls.us, cls.Fs, cls.F_rs, cls.f_es, cls.qs, cls.q_dots, cls.q_r_dots, cls.Vs = cls.run_simulation(sim_time, plot, return_data)
        
    @classmethod
    def run_simulation(cls, sim_time, plot, return_data):
        return run_sim(cls.robot_type, cls.planner_type, cls.controller_type, sim_time=sim_time, plot=plot, return_data=return_data)
    
    def test_ts_identical(self):
        assert_array_equal(self.sim_data['ts'][self.controller_type], self.ts, err_msg="Mismatch in 'ts'")

    def test_us_identical(self):
        assert_array_equal(self.sim_data['us'][self.controller_type], self.us, err_msg="Mismatch in 'us'")

    def test_Fs_identical(self):
        assert_array_equal(self.sim_data['Fs'][self.controller_type], self.Fs, err_msg="Mismatch in 'Fs'")

    def test_F_rs_identical(self):
        assert_array_equal(self.sim_data['F_rs'][self.controller_type], self.F_rs, err_msg="Mismatch in 'F_rs'")

    def test_f_es_identical(self):
        assert_array_equal(self.sim_data['f_es'][self.controller_type], self.f_es, err_msg="Mismatch in 'f_es'")

    def test_qs_identical(self):
        assert_array_equal(self.sim_data['qs'][self.controller_type], self.qs, err_msg="Mismatch in 'qs'")

    def test_q_dots_identical(self):
        assert_array_equal(self.sim_data['q_dots'][self.controller_type], self.q_dots, err_msg="Mismatch in 'q_dots'")

    def test_q_r_dots_identical(self):
        assert_array_equal(self.sim_data['q_r_dots'][self.controller_type], self.q_r_dots, err_msg="Mismatch in 'q_r_dots'")
    
    def test_Vs_identical(self):
        assert_array_equal(self.sim_data['Vs'][self.controller_type], self.Vs, err_msg="Mismatch in 'Vs'")


if __name__ == '__main__':
    unittest.main()
