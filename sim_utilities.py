from enum import Enum
from collections import namedtuple
from algorithm_core.Planner import VelocityPlanner, PointVelocityField, HorinzontalLineVelocityField, UpRampVelocityField, SuperQuadraticField
from algorithm_core.Controller import BaseControl, PassiveBaseControl, PVFC, PDControl, AugmentedPDControl
from Robot import Robot, Quadrotor, AerialManipulator
import algorithm_core.params as params
import matplotlib.pyplot as plt
import algorithm_core.utilities as util
import numpy as np

RobotData = namedtuple('RobotData', ['robot_type', 'dynamics_type'])

class RobotInfo(Enum):
    QUAD = RobotData('Quadrotor', 'QuadrotorTranslationalDynamics')
    AM = RobotData('AerialManipulator', 'AerialManipulatorTaskDynamics')
    
class PlannerInfo(Enum):
    POINT = 'PointVelocityField'
    HORIZONTAL = 'HorinzontalLineVelocityField'
    RAMP = 'UpRampVelocityField'

class ControllerInfo(Enum):
    PVFC = 'PVFC'
    AUGMENTEDPD = 'AugmentedPDControl'
    PD = 'PDControl'

_robots = {
        RobotInfo.AM: (AerialManipulator, params.AM_params(), params.AM_q, params.AM_q_dot, params.AM_q_ddot),
        RobotInfo.QUAD: (Quadrotor, params.quadrotor_params(), params.quad_q, params.quad_q_dot, params.quad_q_ddot)
    }

_base_robot_planners = {
        RobotInfo.AM: params.base_AM_planner_params(),
        RobotInfo.QUAD: params.base_quad_planner_params()
    }

_plans = {
        PlannerInfo.POINT: (PointVelocityField, params.base_point_planner_params(), params.point_planner_params()),
        PlannerInfo.HORIZONTAL: (HorinzontalLineVelocityField, params.base_horizontal_line_planner_params(), params.horizontal_line_planner_params()),
        PlannerInfo.RAMP: (UpRampVelocityField, params.base_up_ramp_planner_params(), params.up_ramp_planner_params())
    }

_controllers = {
        ControllerInfo.PVFC: (PVFC, params.passive_params()),
        ControllerInfo.PD: (PDControl, params.pd_params()),
        ControllerInfo.AUGMENTEDPD: (AugmentedPDControl, params.passive_params(), params.pd_params())
    }

def create_robot(robot: RobotInfo) -> Robot:
    robot_class, robot_params, robot_init_p, robot_init_v, robot_init_a = _robots.get(robot)
    return robot_class(robot_params), robot_params, robot_init_p.copy(), robot_init_v.copy(), robot_init_a.copy()

def create_planner(plan: PlannerInfo, robot: RobotInfo) -> VelocityPlanner:
    base_robot_planner_params = _base_robot_planners.get(robot)
    planner_class, base_planner_params, planner_params = _plans.get(plan)
    planner = planner_class(base_robot_planner_params, base_planner_params, planner_params)
    if params.obstacle: planner += SuperQuadraticField(base_robot_planner_params, params.super_quadratic_params())
    return planner

def create_controller(controller: ControllerInfo, robot_params: namedtuple) -> BaseControl:
    controller_class, *args = _controllers.get(controller)
    return controller_class(robot_params, params.attitude_control_params(), *args)

def get_task_space_state_vectorized(q, q_dot, tool_length):
    q_T, q_T_dot = zip(*[util.config_to_task(q_i, q_dot_i, tool_length) for q_i, q_dot_i in zip(q.T[:,:,None], q_dot.T[:,:,None])])
    return np.concatenate(q_T,axis=1), np.concatenate(q_T_dot,axis=1)

def create_fig(rows,cols,figsize=(16,9),sharex=True):
    fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)
    if rows==1 and cols==1: ax = np.array([ax])
    return fig, ax

def get_config_and_task_state_vectorized(robot, qs, q_dots):
    if robot.__class__.__name__==RobotInfo.QUAD.value.robot_type: return qs, q_dots, qs[:2,:], q_dots[:2,:]
    else: 
        q_Ts, q_T_dots = get_task_space_state_vectorized(qs, q_dots, robot.dynamics.tool_length)
        return qs, q_dots, q_Ts, q_T_dots
    
def check_sim_module_compatiblity(robot_type, planner_type):
    if robot_type==RobotInfo.QUAD and planner_type==PlannerInfo.RAMP: 
        raise ValueError('Robot '+'['+robot_type+']'+' can not be run with planner '+'['+planner_type+']'+': please choose a differnt combination.')
    else: pass
    
def computeRampParams(p1, p2):
    '''
    input: 
        2 (x,z) points on a line - p1, p2
    output: 
        slope and z-intercept of line going through p1, p2 - m, b
    '''
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m*p1[0]
    return m, b

def contactRamp(p1, p2, q_T):
    '''
    input: 
        2 (x,z) points on a line and end-effector poistion - p1, p2, q_T
    output: 
        True if end-effector is in contact with the ramp, else False - bool
    '''
    x_T, z_T = q_T[0,0], q_T[1,0]
    m,b = computeRampParams(p1, p2)
    z_ramp = m*x_T + b
    return z_T <= z_ramp

def compute_ramp_force(k, mu, p1, p2, q_T, q_dot_T): 
    '''
    input: 
        ramp stiffness, coefficient of kinetic friction, points and end-effector posiion - k, mu, p1, p2, q_T
    output: 
        (x,z) force vector - F
    '''
    if contactRamp(p1, p2, q_T): 
        m, b = computeRampParams(p1, p2)
        Delta = np.abs(-m*q_T[0,0] + q_T[1,0] - b) / np.sqrt(m**2 + 1) # distance into the ramp
        n_hat = np.array([[-m], [1.]]) / np.sqrt(1 + m**2)
        force_scaler = k*Delta
        # normal_force = force_scaler*n_hat + b*q_dot_T*n_hat # TODO: implement damping
        normal_force = force_scaler*n_hat
        t_hat = np.array([[1.], [m]]) / np.sqrt(1 + m**2)
        tangent_force = mu*force_scaler*t_hat
        return normal_force + tangent_force
    else: return np.zeros((2,1))