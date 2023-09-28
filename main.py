from Simulation import Sim
from Planner import VelocityPlanner, PointVelocityField, HorinzontalLineVelocityField, UpRampVelocityField, SuperQuadraticField
from Controller import BaseControl, PVFC, PDControl, AugmentedPDControl
from Robot import Robot, Quadrotor, AerialManipulator
from Plotter import PlotSimResults, PlotPassiveSimResults, VizVelocityField, plt 
import utilities as util
import params
from collections import namedtuple
import numpy as np

def check_module_compatiblity(robot_type, planner_type):
    if robot_type==util.RobotInfo.QUAD and planner_type==util.PlannerInfo.RAMP: 
        raise ValueError('Robot '+'['+robot_type.value+']'+' can not be run with planner '+'['+planner_type.value+']'+': please choose a differnt combination.')
    else: pass

def create_robot(robot: util.RobotInfo) -> Robot:
    robots = {
        util.RobotInfo.AM: (AerialManipulator, params.AM_params(), params.AM_q, params.AM_q_dot),
        util.RobotInfo.QUAD: (Quadrotor, params.quadrotor_params(), params.quad_q, params.quad_q_dot)
    }
    robot_class, robot_params, robot_init_p, robot_init_v = robots.get(robot)
    return robot_class(robot_params), robot_params, robot_init_p, robot_init_v

def create_planner(plan: util.PlannerInfo, robot: util.RobotInfo) -> VelocityPlanner:
    base_robot_planners = {
        util.RobotInfo.AM: params.base_AM_planner_params(),
        util.RobotInfo.QUAD: params.base_quad_planner_params()
    }
    base_robot_planner_params = base_robot_planners.get(robot)
    plans = {
        util.PlannerInfo.POINT: (PointVelocityField, params.base_point_planner_params(), params.point_planner_params()),
        util.PlannerInfo.HORIZONTAL: (HorinzontalLineVelocityField, params.base_horizontal_line_planner_params(), params.horizontal_line_planner_params()),
        util.PlannerInfo.RAMP: (UpRampVelocityField, params.base_up_ramp_planner_params(), params.up_ramp_planner_params())
    }
    planner_class, base_planner_params, planner_params = plans.get(plan)
    planner = planner_class(base_robot_planner_params, base_planner_params, planner_params)
    if params.obstacle: planner += SuperQuadraticField(base_robot_planner_params, params.super_quadratic_params())
    return planner

def create_controller(controller: util.ControllerInfo, robot_params: namedtuple) -> BaseControl:
    controllers = {
        util.ControllerInfo.PVFC: (PVFC, params.passive_params()),
        util.ControllerInfo.PD: (PDControl, params.pd_params()),
        util.ControllerInfo.AUGMENTEDPD: (AugmentedPDControl, params.passive_params(), params.pd_params())
    }
    controller_class, *args = controllers.get(controller)
    return controller_class(robot_params, params.attitude_control_params(), *args)

def run_sim(robot_type=util.RobotInfo.AM, planner_type=util.PlannerInfo.RAMP, controller_type=util.ControllerInfo.PVFC, sim_time=10, dt=0.01, plot=True):
    # create modules
    robot, robot_params, q_0, q_dot_0 = create_robot(robot_type)
    planner = create_planner(planner_type, robot_type)
    controller = create_controller(controller_type, robot_params)
    
    sim = Sim(planner, controller, robot, params.ramp_force_params())
    ts, u, F, F_r, f_e, q, q_dot, q_r_dot, V, V_dot = sim.run(q_0, q_dot_0, params.q_r, params.q_r_dot, sim_time=sim_time, dt=dt)  # run simulation
    us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs =  np.concatenate(u,axis=1), np.concatenate(F,axis=1), (F_r), np.concatenate(f_e,axis=1), np.concatenate(q,axis=1), np.concatenate(q_dot,axis=1), (q_r_dot), np.concatenate(V,axis=1)
    if plot: plot_results(planner, controller, robot, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs)

def get_task_space_state(q, q_dot, tool_length):
    q_T, q_T_dot = zip(*[util.configToTask(q_i, q_dot_i, tool_length) for q_i, q_dot_i in zip(q.T[:,:,None], q_dot.T[:,:,None])])
    return np.concatenate(q_T,axis=1), np.concatenate(q_T_dot,axis=1)

def create_fig(rows,cols,figsize=(16,9),sharex=True):
    fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)
    if rows==1 and cols==1: ax = np.array([ax])
    return fig, ax

def determine_controlled_state(robot_type, qs, q_dots):
    if robot_type==util.RobotInfo.QUAD.value.robot_type: return qs[:2,:], q_dots[:2,:]  # working in the translational state of 2D quadrotor
    else: return get_task_space_state(qs, q_dots, params.tool_length)

def plot_augmented_sim_results(planner, controller, robot, ts, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs):
    plotter = PlotPassiveSimResults(planner, controller, robot)
    fig1, ax1 = create_fig(2, 1)
    plotter.plotVelocityTracking(fig1, ax1, ts, q_dots, q_r_dots, Vs)
    
    fig2, ax2 = create_fig(2, 1)
    plotter.plotPowerAnnihilation(fig2, ax2, ts, Fs, F_rs, q_dots, q_r_dots)
    
    fig3, ax3 = create_fig(3, 1)
    plotter.plotPassivity(fig3, ax3, ts, q_dots, q_r_dots, f_es)
    return plotter

def plot_sim_results(planner, controller, robot, ts, q_dots, Vs):
    plotter = PlotSimResults(planner, controller, robot)
    fig1, ax1 = create_fig(2, 1)
    plotter.plotVelocityTracking(fig1, ax1, ts, q_dots, Vs)
    return plotter

def plot_sim_summary(plotter, planner_type, ts, us, qs):
    fig, ax = create_fig(1, 1)
    if planner_type==util.PlannerInfo.RAMP.value:
        plotter.plotRamp(fig, ax, qs, color='black')
        plotter.plotTaskState(fig, ax, qs, color='green')
    plotter.plotVelocityField(fig, ax, qs)
    plotter.plotRobot(fig, ax, ts, qs, us, num=8)
    plotter.plotConfigState(fig, ax, qs, color='blue')

def plot_results(planner, controller, robot, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs):
    robot_type = robot.__class__.__name__
    controller_type = controller.__class__.__name__
    planner_type = planner.__class__.__name__
    q, q_dots = determine_controlled_state(robot_type, qs, q_dots)
    if controller_type in [util.ControllerInfo.PVFC.value, util.ControllerInfo.AUGMENTEDPD.value]: 
        plotter = plot_augmented_sim_results(planner, controller, robot, ts, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs)
    else: plotter = plot_sim_results(planner, controller, robot, ts, q_dots, Vs)
    plot_sim_summary(plotter, planner_type, ts, us, qs)
    plt.show()

def run_velocity_field_viz():
    planner = UpRampVelocityField(params.base_AM_planner_params(), params.base_up_ramp_planner_params(), params.up_ramp_planner_params(), visualize=True)
    plotter = VizVelocityField(planner)
    fig, ax = create_fig(1, 1)
    z_T = np.linspace(0, 1, 10)
    x_T = np.linspace(0, 4, 20) 
    plotter.plotRamp(fig, ax, x_T, color='black')
    plotter.plotVelocityField(fig, ax, x_T, z_T)
    plt.show()

if __name__ == '__main__':
    robot_type: util.RobotInfo = util.RobotInfo.AM
    planner_type: util.PlannerInfo = util.PlannerInfo.HORIZONTAL
    controller_type: util.ControllerInfo = util.ControllerInfo.AUGMENTEDPD
    check_module_compatiblity(robot_type, planner_type)
    run_sim(robot_type=robot_type, planner_type=planner_type, controller_type=controller_type, sim_time=60, plot=True)  
    # run_velocity_field_viz()
