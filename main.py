from Simulation import Sim
from Planner import VelocityPlanner, PointVelocityField, HorinzontalLineVelocityField, UpRampVelocityField, SuperQuadraticField
from Controller import BaseControl, PVFC, PDControl, AugmentedPDControl
from Robot import Robot, Quadrotor, AerialManipulator
from Plotter import PlotSimResults, PlotPassiveSimResults, VizVelocityField, plt 
import utilities as util
import params
from collections import namedtuple
import numpy as np
from enum import Enum

class RobotType(Enum):
    AM = 'AM'
    QUAD = 'quad'

class PlannerType(Enum):
    POINT = 'point'
    HORIZONTAL = 'horizontal'
    RAMP = 'ramp'

class ControllerType(Enum):
    PVFC = 'PVFC'
    PDCONTROL = 'PDControl'
    AUGMENTEDPDCONTROL = 'AugmentedPDControl'

def createRobot(robot: RobotType) -> Robot:
    robots = {
        RobotType.AM: (AerialManipulator, params.AM_params(), params.AM_q, params.AM_q_dot),
        RobotType.QUAD: (Quadrotor, params.quadrotor_params(), params.quad_q, params.quad_q_dot)
    }
    robot_class, robot_params, robot_init_p, robot_init_v = robots.get(robot)
    return robot_class(robot_params), robot_params, robot_init_p, robot_init_v

def createPlanner(plan: PlannerType, robot: RobotType) -> VelocityPlanner:
    base_robot_planners = {
        RobotType.AM: params.base_AM_planner_params(),
        RobotType.QUAD: params.base_quad_planner_params()
    }
    base_robot_planner_params = base_robot_planners.get(robot)
    plans = {
        PlannerType.POINT: (PointVelocityField, params.base_point_planner_params(), params.point_planner_params()),
        PlannerType.HORIZONTAL: (HorinzontalLineVelocityField, params.base_horizontal_line_planner_params(), params.horizontal_line_planner_params()),
        PlannerType.RAMP: (UpRampVelocityField, params.base_up_ramp_planner_params(), params.up_ramp_planner_params())
    }
    planner_class, base_planner_params, planner_params = plans.get(plan)
    planner = planner_class(base_robot_planner_params, base_planner_params, planner_params)
    if params.obstacle: planner += SuperQuadraticField(base_robot_planner_params, params.super_quadratic_params())
    return planner

def createController(controller: ControllerType, robot_params: namedtuple) -> BaseControl:
    controllers = {
        ControllerType.PVFC: (PVFC, params.passive_params()),
        ControllerType.PDCONTROL: (PDControl, params.pd_params()),
        ControllerType.AUGMENTEDPDCONTROL: (AugmentedPDControl, params.passive_params(), params.pd_params())
    }
    controller_class, *args = controllers.get(controller)
    return controller_class(robot_params, params.attitude_control_params(), *args)

def runSim(robot_type=RobotType.AM, planner_type=PlannerType.RAMP, controller_type=ControllerType.PVFC, sim_time=10, dt=0.01, plot=True):
    # create modules
    robot, robot_params, q_0, q_dot_0 = createRobot(robot_type)
    planner = createPlanner(planner_type, robot_type)
    controller = createController(controller_type, robot_params)
    
    sim = Sim(planner, controller, robot, params.ramp_force_params())
    ts, u, F, F_r, f_e, q, q_dot, q_r_dot, V, V_dot = sim.run(q_0, q_dot_0, params.q_r, params.q_r_dot, params.F_e, sim_time=sim_time, dt=dt)  # run simulation
    us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs =  np.concatenate(u,axis=1), np.concatenate(F,axis=1), (F_r), np.concatenate(f_e,axis=1), np.concatenate(q,axis=1), np.concatenate(q_dot,axis=1), (q_r_dot), np.concatenate(V,axis=1)
    if plot: plotResults(planner, controller, robot, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs)

def getTaskSpaceState(q, q_dot, tool_length):
    q_T, q_T_dot = zip(*[util.configToTask(q_i, q_dot_i, tool_length) for q_i, q_dot_i in zip(q.T[:,:,None], q_dot.T[:,:,None])])
    q_Ts, q_T_dots = np.concatenate(q_T,axis=1), np.concatenate(q_T_dot,axis=1)
    return q_Ts, q_T_dots
    
def plotResults(planner, controller, robot, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs):
    #  sim visualization
    if robot.__class__.__name__=='Quadrotor': q_Ts, q_T_dots = qs[:2,:], q_dots[:2,:]  # working in the translational state of 2D quadrotor
    else: q_Ts, q_T_dots = getTaskSpaceState(qs, q_dots, robot.dynamics.tool_length)
    if controller.__class__.__name__ in ['PVFC', 'AugmentedPDControl']:  # augmented plots
        plotter = PlotPassiveSimResults(planner, controller, robot)
        
        fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
        plotter.plotPowerAnnihilation(fig, ax, ts, Fs, F_rs, q_T_dots, q_r_dots)
        
        fig, ax = plt.subplots(3, 1, figsize=(16,9), sharex=True)
        plotter.plotPassivity(fig, ax, ts, q_T_dots, q_r_dots, f_es)
        
        fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
        plotter.plotVelocityTracking(fig, ax, ts, q_T_dots, q_r_dots, Vs)
    else: 
        plotter = PlotSimResults(planner, controller, robot)
        fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
        plotter.plotVelocityTracking(fig, ax, ts, q_T_dots, Vs)

    fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
    if robot.__class__.__name__=='AerialManipulator':
        plotter.plotRamp(fig, ax, q_Ts, color='black')
        plotter.plotTaskState(fig, ax, q_Ts, color='green')
    plotter.plotVelocityField(fig, ax, qs)
    plotter.plotRobot(fig, ax, ts, qs, us, num=8)
    plotter.plotConfigState(fig, ax, qs, color='blue')
    plt.show()

def runVizVelocityField():
    planner = UpRampVelocityField(params.base_AM_planner_params(), params.base_up_ramp_planner_params(), params.up_ramp_planner_params(), visualize=True)
    plotter = VizVelocityField(planner)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
    # make test arrays of states
    z_T = np.linspace(0, 1, 10)
    x_T = np.linspace(0, 4, 20) 
    plotter.plotRamp(fig, ax, x_T, color='black')
    plotter.plotVelocityField(fig, ax, x_T, z_T)
    plt.show()


if __name__ == '__main__':
    # Choose which sim to run, sims are in 2D (x, z) plane
    robot_type: RobotType = RobotType.AM  
    planner_type = PlannerType.RAMP
    controller_type = ControllerType.PVFC  
    robot, planner, controller, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs = \
        runSim(robot_type=robot_type, planner_type=planner_type, controller_type=controller_type, sim_time=60, plot=True)  
    # runVizVelocityField()
