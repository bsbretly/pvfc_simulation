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
            cls.loaded_data = pickle.load(file)
        
        cls.sim_data = cls.loaded_data['sim_data']
        cls.robot_type = cls.loaded_data['robot_type']
        cls.planner_type = cls.loaded_data['planner_type']
        cls.controller_types = cls.loaded_data['controller_types']
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


if __name__ == '__main__':
    unittest.main()
