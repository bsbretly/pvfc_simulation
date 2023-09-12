import numpy as np
from collections import namedtuple
DEG_TO_RAD = np.pi/180

# Robot parameters
m, I = 1.5, 4.856e-3 
m_t, I_t, tool_length = .1, 1, 0.15
QuadrotorParams = namedtuple('QuadrotorParams', ['m', 'I'], defaults = (m, I))
AMParams = namedtuple('AMParams', ['m', 'm_t', 'I', 'I_t', 'tool_length'], defaults = (m, m_t, I, I_t, tool_length))

# Control parameters
E_bar = 150
m_r, gamma = .1, .1
theta_K_p, theta_K_d = 200., 20.
ControllerParams = namedtuple('ControllerParams', ['m_r', 'E_bar', 'gamma', 'theta_K_p', 'theta_K_d'], defaults=(m_r, E_bar, gamma, theta_K_p, theta_K_d))

# External force
F_e = np.array([[-0.1], [0.2]]) # precomputed external force
plane_k = .2 # stiffness
plane_mu = .6 # kinetic friction coefficient
ramp_k = 1.
ramp_mu = .6

PlaneForceParams = namedtuple('PlaneForceParams', ['plane_k', 'plane_mu'], defaults=(plane_k, plane_mu))
RampForceParams = namedtuple('RampForceParams', ['ramp_k', 'ramp_mu'], defaults=(ramp_k, ramp_mu))

# Planner parameters
z_intercept = 0.
horizontal_normal_gain, horizontal_tangent_gain = 1., 1.
up_ramp_normal_gain, up_ramp_tangent_gain = 8., 1.
p1, p2 = [0.5,0.], [1.,.25] # two points on the ramp
delta = 0.1 # distance "beyond" surface to facilitate interaction
x_d, z_d = 0.5, 1. # desired point
obstacle = False
obs_x, obs_z = 1., 0. # obstacle position
obs_m, obs_n, obs_L, obs_len = 5., 2., 1., 5. # obstacle parameters

BaseQuadPlannerParams = namedtuple('BaseQuadPlannerParams', ['m', 'm_r', 'E_bar'], defaults=(m, m_r, E_bar))
BaseAMPlannerParams = namedtuple('BaseAMPlannerParams', ['m', 'm_r', 'tool_length', 'E_bar'], defaults=(m, m_r, tool_length, E_bar))
PointPlannerParams = namedtuple('PointPlannerParams', ['V_l', 'V_horizontal_line_tangent_gain', 'x_d', 'z_d'], defaults=(horizontal_normal_gain, horizontal_tangent_gain, x_d, z_d))
horizontal_line_planner_params = namedtuple('horizontal_line_planner_params', ['horizontal_normal_gain', 'horizontal_tangent_gain', 'z_intercept', 'delta'], defaults=(horizontal_normal_gain, horizontal_tangent_gain, z_intercept, delta))
UpRampPlannerParams = namedtuple('horizontal_line_planner_params', ['ramp_normal_gain', 'ramp_tangent_gain', 'delta', 'p1', 'p2'], defaults=(up_ramp_normal_gain, up_ramp_tangent_gain, delta, p1, p2))
SuperQuadraticParams = namedtuple('SuperQuadraticParams', ['obs_x', 'obs_z', 'obs_m', 'obs_n', 'obs_L', 'obs_len'], defaults=(obs_x, obs_z, obs_m, obs_n, obs_L, obs_len))

# initial conditions
AM_q, AM_q_dot = np.array([[0., .4, 0.*DEG_TO_RAD, 90.*DEG_TO_RAD]]).T, np.array([[0.1 , -0.05, 0., 0.]]).T # q = [x, z, theta, Beta]^T
Quad_q, Quad_q_dot = np.array([[0., 0., 0.*DEG_TO_RAD]]).T, np.array([[0. , 0.05, 0.]]).T # q = [x, z, theta]^T
q_r, q_r_dot = 0., 0.

