from Simulation import Sim
from Planner import PointVelocityField, PlanarVelocityField, RampVelocityField, SuperQuadraticField
from Controller import TranslationalPVFC, TaskPVFC
from Robot import Quadrotor, AerialManipulator
from Plotter import PlotSimResults, PlotAMSimResults, TestVelocityField, plt 
import utilities as util
import params
import numpy as np

def ramp_condition(q, rampPlanner):
    return rampPlanner.x_s <= q[0,0] <= rampPlanner.x_e and q[1,0] <= rampPlanner.z_h

def runAMSim(sim_time=10, dt=0.01):
    # define modules
    us, Fs, f_es, qs, q_dots, q_Ts, q_T_dots, q_r_dots, V_Ts = [], [], [], [], [], [], [], [], []
    planners = PlanarVelocityField(params.BaseAMPlannerParams(), params.PlanarPlannerParams()), RampVelocityField(params.BaseAMPlannerParams(), params.RampPlannerParams())
    if params.obstacle: planner += SuperQuadraticField(params.BasePlannerParams(), params.SuperQuadraticParams())
    controller = TaskPVFC(params.AMParams(), params.ControllerParams())
    robot = AerialManipulator(params.AMParams())
    
    # run simulation
    sim = Sim(planners, controller, robot)
    ts, u, F, f_e, q, q_dot, q_r_dot = sim.run(params.AM_q, params.AM_q_dot, params.q_r, params.q_r_dot, params.F_e, sim_time=sim_time, dt=dt)

    # get task space variables
    q_T, q_T_dot = zip(*[util.configToTask(q_i, q_dot_i, robot.dynamics.tool_length) for q_i, q_dot_i in zip(np.concatenate(q,axis=1).T[:,:,None], np.concatenate(q_dot,axis=1).T[:,:,None])])
    if type(planners) is tuple:
        planner_results = [
            planners[1].step(q_i, q_dot_i) if ramp_condition(q_i, planners[1]) else planners[0].step(q_i, q_dot_i)
            for q_i, q_dot_i in zip(np.concatenate(q, axis=1).T[:,:,None], np.concatenate(q_dot, axis=1).T[:,:,None])
        ]
        V_T, V_T_dot = zip(*planner_results)
    else: V_T, V_T_dot = zip(*[sim.planner.step(q_i, q_dot_i) for q_i, q_dot_i in zip(np.concatenate(q,axis=1).T[:,:,None], np.concatenate(q_dot,axis=1).T[:,:,None])])

    # concatenate data
    us.append(np.concatenate(u,axis=1)), Fs.append(np.concatenate(F,axis=1)), f_es.append(np.concatenate(f_e,axis=1)), qs.append(np.concatenate(q,axis=1)), q_dots.append(np.concatenate(q_dot,axis=1)), q_Ts.append(np.concatenate(q_T,axis=1)), q_T_dots.append(np.concatenate(q_T_dot,axis=1)), q_r_dots.append(q_r_dot), V_Ts.append(np.concatenate(V_T,axis=1))

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
    plotter = PlotAMSimResults(planners, controller, robot,ts, us, Fs, f_es, qs, q_dots, q_Ts, q_T_dots, q_r_dots, V_Ts)
    plotter.plotRamp(fig, ax, q_Ts)
    plotter.plotRobot(fig, ax, ts, qs, us)
    plotter.plotVelocityField(fig, ax, q_Ts)
    plotter.plotConfigState(fig, ax, qs, color='r')
    plotter.plotTaskState(fig, ax, color='g')

    plt.show()

def runQuadSim(sim_time=10, dt=0.01):
    # define modules
    us, Fs, f_es, qs, q_dots, q_r_dots, V_Ts = [], [], [], [], [], [], []
    planner = PointVelocityField(params.BaseQuadPlannerParams(), params.PointPlannerParams())
    if params.obstacle: planner += SuperQuadraticField(params.BasePlannerParams(), params.SuperQuadraticParams())
    controller = TranslationalPVFC(params.QuadrotorParams(), params.ControllerParams())
    robot = Quadrotor(params.QuadrotorParams())
    
    # run simulation
    sim = Sim(planner, controller, robot)
    ts, u, F, f_e, q, q_dot, q_r_dot = sim.run(params.Quad_q, params.Quad_q_dot, params.q_r, params.q_r_dot, params.F_e, sim_time=sim_time, dt=dt)

    V, V_dot = zip(*[sim.planner.step(q_i, q_dot_i) for q_i, q_dot_i in zip(np.concatenate(q,axis=1).T[:,:,None], np.concatenate(q_dot,axis=1).T[:,:,None])])

    # concatenate data
    us.append(np.concatenate(u,axis=1)), Fs.append(np.concatenate(F,axis=1)), f_es.append(np.concatenate(f_e,axis=1)), qs.append(np.concatenate(q,axis=1)), q_dots.append(np.concatenate(q_dot,axis=1)), q_r_dots.append(q_r_dot), V_Ts.append(np.concatenate(V,axis=1))

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
    plotter = PlotSimResults(planner, controller, robot,ts, us, Fs, f_es, qs, q_dots, q_r_dots, V_Ts)
    plotter.plotRobot(fig, ax, ts, qs, us)
    plotter.plotVelocityField(fig, ax, qs)
    plotter.plotConfigState(fig, ax, color='r')
    plt.show()

def runTestVelocityField():
    planarPlanner = PlanarVelocityField(params.BaseAMPlannerParams(), params.PlanarPlannerParams())
    rampPlanner = RampVelocityField(params.BaseAMPlannerParams(), params.RampPlannerParams())
    plotter = TestVelocityField(planarPlanner, rampPlanner)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16,9), sharex=True)
    # make a test grid of states
    x = np.arange(0, 5, 0.1)
    z = np.arange(0, 5, 0.1)
    p_x, p_y = np.meshgrid(x, z, indexing='ij')
    V_x, V_y = np.zeros((x.size,z.size)), np.zeros((x.size,z.size))
    plotter.plotVelocityField(fig, ax, p_x, p_y, V_x, V_y)
    plt.show()

if __name__ == '__main__':
    # Choose which sim to run, sims are in 2D (x, z) plane
    runAMSim(sim_time=60)
    # runQuadSim(sim_time=90)
    # runTestVelocityField()
    # TODO sim for quad moving in a circle
    # TODO sim for quad moving to desired (x, z) point
    
