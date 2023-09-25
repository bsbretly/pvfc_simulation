from Simulation import Sim
from Planner import PointVelocityField, HorinzontalLineVelocityField, UpRampVelocityField, SuperQuadraticField
from Controller import PVFC, PDControl, PassivePDControl
from Robot import Quadrotor, AerialManipulator
from Plotter import PlotSimResults, PlotPassiveSimResults, VizVelocityField, plt 
import utilities as util
import params
import numpy as np

def createPlanner(plan, robot_params):
    plans = {
        'point': [PointVelocityField, params.base_point_planner_params(), params.point_planner_params()],
        'horizontal': [HorinzontalLineVelocityField, params.base_horizontal_line_planner_params(), params.horizontal_line_planner_params()],
        'ramp': [UpRampVelocityField, params.passive_params(), params.base_up_ramp_planner_params(), params.up_ramp_planner_params()]
    }
    if type(robot_params).__name__=='AM_params': base_robot_planner_params = params.base_AM_planner_params()
    else: base_robot_planner_params = params.base_quad_planner_params()
    planner_class, base_planner_params, planner_params = plans.get(plan, PointVelocityField)
    planner = planner_class(base_robot_planner_params, base_planner_params, planner_params)
    if params.obstacle:
        planner += SuperQuadraticField(params.base_AM_planner_params(), params.super_quadratic_params())
    return planner

def createRobot(robot_type):
    if robot_type == 'AM': return AerialManipulator(params.AM_params()), params.AM_params(), params.AM_q, params.AM_q_dot
    else: return Quadrotor(params.quadrotor_params()), params.quadrotor_params(), params.quad_q, params.quad_q_dot


def createController(controller, robot_params):
    controllers = {
        'PVFC': [PVFC, params.passive_params()],
        'PDControl': [PDControl, params.pd_params()],
        'PassivePDControl': [PassivePDControl, params.passive_params(), params.pd_params()]
    }
    controller_class, *args = controllers.get(controller, PVFC)
    return controller_class(robot_params, params.attitude_control_params(), *args)

def runSim(robot='AM', plan='ramp', control='PVFC', sim_time=10, dt=0.01, plot_passivity=False):
    # define modules
    
    robot, robot_params, q_0, q_dot_0 = createRobot(robot)
    planner = createPlanner(plan, robot_params)
    controller = createController(control, robot_params)
    
    # run simulation
    sim = Sim(planner, controller, robot, params.ramp_force_params())
    ts, u, F, F_r, f_e, q, q_dot, q_r_dot = sim.run(q_0, q_dot_0, params.q_r, params.q_r_dot, params.F_e, sim_time=sim_time, dt=dt)

    # get task space variables
    q_T, q_T_dot = zip(*[util.configToTask(q_i, q_dot_i, robot.dynamics.tool_length) for q_i, q_dot_i in zip(np.concatenate(q,axis=1).T[:,:,None], np.concatenate(q_dot,axis=1).T[:,:,None])])

    # get desired velocity field
    V_T, V_T_dot = zip(*[sim.planner.step(q_i, q_dot_i) for q_i, q_dot_i in zip(np.concatenate(q,axis=1).T[:,:,None], np.concatenate(q_dot,axis=1).T[:,:,None])])

    # concatenate data
    us, Fs, F_rs, f_es, qs, q_dots, q_Ts, q_T_dots, q_r_dots, V_Ts =  np.concatenate(u,axis=1), np.concatenate(F,axis=1), (F_r), np.concatenate(f_e,axis=1), np.concatenate(q,axis=1), np.concatenate(q_dot,axis=1), np.concatenate(q_T,axis=1), np.concatenate(q_T_dot,axis=1), (q_r_dot), np.concatenate(V_T,axis=1)

    #  sim visualization
    if control in ['PVFC', 'PassivePDControl']:  # passivity plots
        plotter = PlotPassiveSimResults(planner, controller, robot)
        if plot_passivity:
            fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
            plotter.plotPowerAnnihilation(fig, ax, ts, Fs, F_rs, q_T_dots, q_r_dots)
            
            fig, ax = plt.subplots(3, 1, figsize=(16,9), sharex=True)
            plotter.plotPassivity(fig, ax, ts, q_T_dots, q_r_dots, f_es)
            
            fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
            plotter.plotVelocityTracking(fig, ax, ts, q_T_dots, q_r_dots, V_Ts)
    else: 
        plotter = PlotSimResults(planner, controller, robot)
        fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
        plotter.plotVelocityTracking(fig, ax, ts, q_T_dots, V_Ts)

    fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
    plotter.plotVelocityField(fig, ax, qs)
    plotter.plotRamp(fig, ax, q_Ts, color='black')
    plotter.plotRobot(fig, ax, ts, qs, us, num=8)
    plotter.plotConfigState(fig, ax, qs, color='blue')
    plotter.plotTaskState(fig, ax, q_Ts, color='green')

    

    
    
        
    plt.show()

# def runQuadSim(control='PVFC', sim_time=10, dt=0.01, plot_passivity=False):
#     # define modules
#     planner = PointVelocityField(params.base_quad_planner_params(), params.base_point_planner_params(), params.point_planner_params())
#     if params.obstacle: planner += SuperQuadraticField(params.base_quad_planner_params(), params.super_quadratic_params())
#     controller = PVFC(params.quadrotor_params(), params.attitude_control_params(), params.passive_params())
#     robot = Quadrotor(params.quadrotor_params())
    
#     # run simulation
#     sim = Sim(planner, controller, robot)
#     ts, u, F, f_e, q, q_dot, q_r_dot = sim.run(params.quad_q, params.quad_q_dot, params.q_r, params.q_r_dot, params.F_e, sim_time=sim_time, dt=dt)

#     # get desired velocity field
#     V, V_dot = zip(*[sim.planner.step(q_i, q_dot_i) for q_i, q_dot_i in zip(np.concatenate(q,axis=1).T[:,:,None], np.concatenate(q_dot,axis=1).T[:,:,None])])

#     # concatenate data
#     us, Fs, f_es, qs, q_dots, q_r_dots, Vs =  np.concatenate(u,axis=1), np.concatenate(F,axis=1), np.concatenate(f_e,axis=1), np.concatenate(q,axis=1), np.concatenate(q_dot,axis=1), (q_r_dot), np.concatenate(V,axis=1)

#     # plot
#     fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
#     plotter = PlotSimResults(planner, controller, robot)
#     plotter.plotRobot(fig, ax, ts, qs, us)
#     plotter.plotVelocityField(fig, ax, qs)
#     plotter.plotConfigState(fig, ax, qs, color='blue')
#     plt.show()

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
    robot = 'Quad'  # robot = 'AM', 'Quad'
    plan = 'point' # plan = 'point', 'horizontal', 'ramp'
    controller = 'PVFC'  # control = 'PVFC', 'PDControl', 'PassivePDControl'
    runSim(robot=robot, plan=plan, control=controller, sim_time=60)  
    # runVizVelocityField()
    # TODO: plot tracking error 