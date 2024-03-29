import numpy as np
from enum import Enum
from collections import namedtuple

DEG_TO_RAD = np.pi/180

# # quad parameters
# m = 1.5
# I = 4.856e-3 
# # aerial manipulator parameters
# m_t = 0.1
# I_t = 1.
# tool_length = 0.15

# quad parameters
m = 1.35
I = 0.0035
# aerial manipulator parameters
m_t = .034
I_t = 0.0026112
tool_length = 0.48

quadrotor_params = namedtuple('quadrotor_params', ['m', 'I'], defaults = (m, I))
AM_params = namedtuple('AM_params', ['m', 'm_t', 'I', 'I_t', 'tool_length'], defaults = (m, m_t, I, I_t, tool_length))

# Control parameters
E_bar = 100.
m_r = .1 
gamma = .1  # passivity-based controller
theta_K_p, theta_K_d = 200., 20.  # attitude controller
K_p, K_d = 10., 0.  # PD controller
attitude_control_params = namedtuple('attitude_control_params', ['theta_K_p', 'theta_K_d'], defaults=(theta_K_p, theta_K_d))
passive_params = namedtuple('passive_params', ['m_r', 'E_bar', 'gamma'], defaults=(m_r, E_bar, gamma))
pd_params = namedtuple('pd_params', ['K_p', 'K_d'], defaults=(K_p, K_d))

# External force
F_e = np.array([[-0.1], [0.2]]) # precomputed external force
plane_k = 1. # stiffness
plane_mu = .6 # kinetic friction coefficient
scale_k = 1e-9 # scale factor for ramp_k
ramp_k = scale_k*2.5e10 # stiffness of concrete [N/m]
ramp_b = 0. # damping of concrete, TODO: implement
ramp_mu = .6 # kinetic friction coefficient of rubber on concrete
plane_force_params = namedtuple('plane_force_params', ['plane_k', 'plane_mu'], defaults=(plane_k, plane_mu))
ramp_force_params = namedtuple('ramp_force_params', ['ramp_k', 'ramp_mu'], defaults=(ramp_k, ramp_mu))

# Planner parameters
point_normal_gain, point_tangent_gain = 4., 1.
z_intercept = 0.
horizontal_normal_gain, horizontal_tangent_gain = 1., 1.
up_ramp_normal_gain, up_ramp_tangent_gain = 1., 1.
p1, p2 = [0.,0.], [5.,.25] # two points on the ramp
delta = 0.1 # distance "beyond" surface to facilitate interaction [m]
x_d, z_d = 0., 1. # desired point
obstacle = False
obs_x, obs_z = 1., 0. # obstacle position
obs_m, obs_n, obs_L, obs_len = 5., 2., 1., 5. # obstacle parameters

base_quad_planner_params = namedtuple('base_quad_planner_params', ['m', 'm_r', 'E_bar'], defaults=(m, m_r, E_bar))
base_AM_planner_params = namedtuple('base_AM_planner_params', ['m_t', 'm_r', 'tool_length', 'E_bar'], defaults=(m_t, m_r, tool_length, E_bar))

base_point_planner_params = namedtuple('base_point_planner_params', ['point_normal_gain', 'point_tangent_gain'], defaults=(point_normal_gain, point_tangent_gain))
base_horizontal_line_planner_params = namedtuple('base_horizontal_line_planner_params', ['horizontal_normal_gain', 'horizontal_tangent_gain'], defaults=(horizontal_normal_gain, horizontal_tangent_gain))
base_up_ramp_planner_params = namedtuple('base_up_ramp_planner_params', ['up_ramp_normal_gain', 'up_ramp_tangent_gain'], defaults=(up_ramp_normal_gain, up_ramp_tangent_gain))

point_planner_params = namedtuple('point_planner_params', ['x_d', 'z_d'], defaults=(x_d, z_d))
horizontal_line_planner_params = namedtuple('horizontal_line_planner_params', ['z_intercept', 'delta'], defaults=(z_intercept, delta))
up_ramp_planner_params = namedtuple('up_ramp_planner_params', ['delta', 'p1', 'p2'], defaults=(delta, p1, p2))
super_quadratic_params = namedtuple('super_quadratic_params', ['obs_x', 'obs_z', 'obs_m', 'obs_n', 'obs_L', 'obs_len'], defaults=(obs_x, obs_z, obs_m, obs_n, obs_L, obs_len))

# AM initial conditions
AM_q = np.array([[0., .0, 0.*DEG_TO_RAD, 90.*DEG_TO_RAD]]).T  # q = [x, z, theta, beta]^T
AM_q_dot = np.array([[0. , 0.01, 0., 0.]]).T 
AM_q_ddot = np.array([[0. , 0., 0., 0.]]).T

# Quad initial conditions
quad_q = np.array([[0., .2, 0.*DEG_TO_RAD]]).T  # q = [x, z, theta]^T
quad_q_dot = np.array([[0.0, -0.05, 0.]]).T
quad_q_ddot = np.array([[0. , 0., 0.]]).T
q_r, q_r_dot, q_r_ddot = np.array(0.), np.array(0.), np.array(0.)


class RobotType(Enum):
    AerialManipulator = AM_params
    Quadrotor = quadrotor_params
