from Simulation import Sim
from Plotter import PlotSimResults, PlotPassiveSimResults, VizVelocityField, ControlComparison, plt 
import sim_utilities as sim_util
import algorithm_core.params as params
import numpy as np
import pickle

def run_sim(robot_type=sim_util.RobotInfo.AM, planner_type=sim_util.PlannerInfo.RAMP, controller_type=sim_util.ControllerInfo.PVFC, sim_time=10, dt=0.01, plot=True, return_data=False):
    sim_util.check_sim_module_compatiblity(robot_type, planner_type)
    # create modules
    robot, robot_params, q_0, q_dot_0, q_ddot_0 = sim_util.create_robot(robot_type)
    planner = sim_util.create_planner(planner_type, robot_type)
    controller = sim_util.create_controller(controller_type, robot_params)

    sim = Sim(planner, controller, robot, params.ramp_force_params())
    ts, u, F, F_r, f_e, q, q_dot, q_r_dot, V, V_dot = sim.run(q_0, q_dot_0, params.q_r.copy(), params.q_r_dot.copy(), sim_time=sim_time, dt=dt)  # run simulation
    us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs, V_dots =  np.concatenate(u,axis=1), np.concatenate(F,axis=1), (F_r), np.concatenate(f_e,axis=1), np.concatenate(q,axis=1), np.concatenate(q_dot,axis=1), (q_r_dot), np.concatenate(V,axis=1), np.concatenate(V_dot,axis=1)
    if plot: plot_results(planner, controller, robot, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs)
    if return_data: return ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs

def run_tracking_performance_comparo(robot_type=sim_util.RobotInfo.AM, planner_type=sim_util.PlannerInfo.RAMP, controller_types=[sim_util.ControllerInfo.PVFC, sim_util.ControllerInfo.AUGMENTEDPD, sim_util.ControllerInfo.PD], sim_time=60, dt=0.01, gen_data=False):
    data_name = 'tracking_control_comparison_data.pickle'
    if gen_data:
        keys = ['ts', 'us', 'Fs', 'F_rs', 'f_es', 'qs', 'q_dots', 'q_r_dots', 'Vs', 'q_Ts', 'q_T_dots']
        sim_data = {key: {controller_type: None for controller_type in controller_types} for key in keys}
        sim_keys = ['ts', 'us', 'Fs', 'F_rs', 'f_es', 'qs', 'q_dots', 'q_r_dots', 'Vs']
        for controller_type in controller_types:
            sim_results = run_sim(robot_type, planner_type, controller_type, sim_time=sim_time, dt=dt, plot=False, return_data=True)
            for key, value in zip(sim_keys, sim_results): 
                sim_data[key][controller_type] = value
            if robot_type==sim_util.RobotInfo.AM:   
                sim_data['q_Ts'][controller_type], sim_data['q_T_dots'][controller_type] = sim_util.get_task_space_state_vectorized(sim_data['qs'][controller_type], sim_data['q_dots'][controller_type], params.tool_length)
            else: sim_data['q_Ts'][controller_type], sim_data['q_T_dots'][controller_type] = sim_data['qs'][controller_type], sim_data['q_dots'][controller_type]
        try:
            data_pickle = {
                "sim_data": sim_data,
                "robot_type": robot_type,
                "planner_type": planner_type,
                "controller_types": controller_types
            }
            with open('data/' + data_name, 'wb') as file: 
                pickle.dump(data_pickle, file)
        except IOError as e: print(f"File writing error: {e}")
    # Load data from pickle file
    else:
        try:
            with open('data/' + data_name, 'rb') as file: data_pickle = pickle.load(file)
        except IOError as e: print(f"File reading error: {e}")
    sim_data = data_pickle['sim_data']
    data_present = all(all(controller_type in sim_data[key] for controller_type in controller_types) for key in sim_data)
    if not data_present: raise KeyError(f"Some or all desired controller types are missing in the pickle file {data_name}.")
    robot_params = sim_util.create_robot(robot_type)[1]
    controllers = [sim_util.create_controller(controller_type, robot_params) for controller_type in sim_util.ControllerInfo]
    plotter = ControlComparison(controllers, sim_data)
    plotter.plot_comparo(plot_error=True, plot_velocity_tracking=True, controller_types=controller_types)

    plt.show()
    
def plot_augmented_sim_results(planner, controller, robot, ts, Fs, F_rs, f_es, qs, q_dots, q_Ts, q_T_dots, q_r_dots, Vs):
    plotter = PlotPassiveSimResults(planner, controller, robot)
    fig1, ax1 = sim_util.create_fig(2, 1)
    plotter.plotVelocityTracking(fig1, ax1, ts, qs, q_dots, q_T_dots, q_r_dots, Vs)
    
    # fig2, ax2 = sim_util.create_fig(2, 1)
    # plotter.plotPowerAnnihilation(fig2, ax2, ts, Fs, F_rs, q_T_dots, q_r_dots)
    
    # fig3, ax3 = sim_util.create_fig(3, 1)
    # plotter.plotPassivity(fig3, ax3, ts, qs, q_dots, q_Ts, q_T_dots, q_r_dots, f_es)
    return plotter

def plot_sim_results(planner, controller, robot, ts, q_dots, Vs):
    plotter = PlotSimResults(planner, controller, robot)
    fig1, ax1 = sim_util.create_fig(2, 1)
    plotter.plotVelocityTracking(fig1, ax1, ts, q_dots, Vs)
    return plotter

def plot_sim_summary(plotter, robot_type, planner_type, ts, us, qs, q_Ts):
    fig, ax = sim_util.create_fig(1, 1)
    if planner_type==sim_util.PlannerInfo.RAMP.value:
        plotter.plotRamp(fig, ax, q_Ts, color='black')
        plotter.plotTaskState(fig, ax, q_Ts, color='green')
    if robot_type==sim_util.RobotInfo.AM.value.robot_type: plotter.plotVelocityField(fig, ax, q_Ts)
    else: plotter.plotVelocityField(fig, ax, qs)
    plotter.plotRobot(fig, ax, ts, qs, q_Ts, us, num=8)
    plotter.plotConfigState(fig, ax, qs, color='blue')

def plot_results(planner, controller, robot, ts, us, Fs, F_rs, f_es, qs, q_dots, q_r_dots, Vs):
    robot_type = robot.__class__.__name__
    controller_type = controller.__class__.__name__
    planner_type = planner.__class__.__name__
    qs, q_dots, q_Ts, q_T_dots = sim_util.get_config_and_task_state_vectorized(robot, qs, q_dots)
    if controller_type in [sim_util.ControllerInfo.PVFC.value, sim_util.ControllerInfo.AUGMENTEDPD.value]: 
        plotter = plot_augmented_sim_results(planner, controller, robot, ts, Fs, F_rs, f_es, qs, q_dots, q_Ts, q_T_dots, q_r_dots, Vs)
    else: plotter = plot_sim_results(planner, controller, robot, ts, q_T_dots, Vs)
    plot_sim_summary(plotter, robot_type, planner_type, ts, us, qs, q_Ts)
    plt.show()

def run_velocity_field_viz(planner_type, robot_type):
    planner = sim_util.create_planner(planner_type, robot_type)
    plotter = VizVelocityField(planner)
    fig, ax = sim_util.create_fig(1, 1)
    x_T = np.linspace(0, 4, 20) 
    z_T = np.linspace(0, 1, 20)
    q_T = np.array([x_T, z_T])
    plotter.plotRamp(fig, ax, q_T, color='black')
    plotter.plotVelocityField(fig, ax, x_T, z_T)
    plt.show()


if __name__ == '__main__':
    robot_type: sim_util.RobotInfo = sim_util.RobotInfo.QUAD
    planner_type: sim_util.PlannerInfo = sim_util.PlannerInfo.POINT
    controller_type: sim_util.ControllerInfo = sim_util.ControllerInfo.PVFC
    run_sim(robot_type, planner_type, controller_type, sim_time=90, plot=True, return_data=False)
    # controller_types = [sim_util.ControllerInfo.PVFC, sim_util.ControllerInfo.AUGMENTEDPD]#sim_util.ControllerInfo.AUGMENTEDPD]  # define which controllers to compare
    # run_tracking_performance_comparo(robot_type, planner_type, controller_types, sim_time=5, dt=0.01, gen_data=True)  # runs comparo for all controllers
    # run_velocity_field_viz(planner_type, robot_type)  # to visualize the velocity field
