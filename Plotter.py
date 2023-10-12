import matplotlib.lines as mlines
import numpy as np
import formatPlots as fp
import matplotlib.pyplot as plt
import utilities as util
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

class PlotSimResults:
    def __init__(self, planner, controller, robot):
        self.planner, self.controller, self.robot = planner, controller, robot
        fp.setupPlotParams()

    def plotVelocityField(self, fig, ax, qs):
        minimum = np.amin(np.concatenate((qs[0,:], qs[1,:])))
        maximum = np.amax(np.concatenate((qs[0,:], qs[1,:])))
        array = np.linspace(0, maximum, 25)
        x_max = np.amax(qs[0,:])
        z_max = np.amax(qs[1,:])
        x_array = np.linspace(0, x_max, 10) #int(maximum/5))
        z_array = np.linspace(0, z_max, 10) #int(maximum/5))
        p_x, p_z, V_x, V_z = self.createGrid(x_array,z_array)
        for i in range(p_x.shape[0]):
            for j in range(p_x.shape[1]):
                q_T = np.array([p_x[i, j], p_z[i, j]]).reshape(-1,1)
                V = self.planner.plotStep(q_T)
                V_x[i,j] = V[0]
                V_z[i,j] = V[1]
        magnitude = np.sqrt(V_x**2 + V_z**2)
        max_magnitude = np.max(magnitude)
        fraction_of_axis = .2
        shorter_axis_length = min(x_max, z_max)
        scale = max_magnitude / (fraction_of_axis * shorter_axis_length)
        
        ax[0].quiver(p_x, p_z, V_x, V_z, color='k', pivot='middle', alpha=0.3, angles='xy', scale_units='xy', scale=scale, width=0.001*min(x_max, z_max))
        return fig, ax
    
    def createGrid(self, x, y):
        p_x, p_y = np.meshgrid(x, y, indexing='ij')
        return p_x, p_y, np.zeros((x.size,y.size)), np.zeros((x.size,y.size))
    
    def plotConfigState(self, fig, ax, qs, color='blue', linestyle='--'):
        ax[0].plot(qs[0,:], qs[1,:], color=color, linestyle=linestyle)

    def display(self, fig, ax, q, q_T, max_x, max_z):
        if self.robot.__class__.__name__ == util.RobotInfo.AM.value.robot_type: 
            x, z, theta, Beta = q[0], q[1], q[2], q[3]
            ax[0].plot([x, q_T[0]], [z, q_T[1]], 'green')  # plot tool
            ax[0].plot(q_T[0], q_T[1], 'go', label='tool tip')  # plot end-effector
        else: x, z, theta = q[0], q[1], q[2]
        ax[0].plot(x, z, 'bo', label='quadrotor')  # quad CoM
        r = 0.05*max_x
        dx = r*np.cos(-theta)
        dz = r*np.sin(-theta)
        quad_x = [x-dx, x+dx]
        quad_z = [z-dz, z+dz]
        ax[0].plot(quad_x, quad_z, 'blue')  # quad wings
        ax[0].set_xlabel(r'$x\ [m]$')
        ax[0].set_ylabel(r'$z\ [m]$')
        return fig, ax

    def plotRamp(self, fig, ax, q_Ts, color='black', linestyle='-'):
        m,b = util.computeRampParams(self.planner.p1, self.planner.p2)
        x = np.linspace(0, q_Ts[0,-1], 100)
        ax[0].plot(x, m*x + b, color=color, linestyle=linestyle, linewidth=2)
        return fig, ax

    def plotPositionTracking(self, fig, ax, x_d, z_d):
        ax[0].plot(self.ts, np.zeros_like(self.ts) + x_d, 'g--', label=r'$x_d$')
        ax[0].plot(self.ts, self.qs[0,:], 'b', label=r'$x$')
        ax[0].set_ylabel(r'$x\ [m]$')
        ax[0].legend()

        ax[1].plot(self.t, np.zeros_like(self.t) + z_d, 'g--', label=r'$z_d$')
        ax[1].plot(self.t, self.qs[1,:], 'b', label=r'$z$')
        ax[1].set_ylabel(r'$z\ [m]$')
        ax[1].legend()
        return fig, ax

    def plot_velocity(self, fig, ax, ts, q_T_dots, Vs):
        ax[0].plot(ts, q_T_dots[0,:], label=r'$\dot{x}_T$')
        ax[0].plot(ts, Vs[0,:], '--', label=r'$V_x$')
        ax[0].set_ylabel(r'$v\ [m/s]$')
        ax[0].legend()
        
        ax[1].plot(ts, q_T_dots[1,:], label=r'$\dot{z}_T$')
        ax[1].plot(ts, Vs[1,:], '--', label=r'$V_z$')
        ax[1].set_ylabel(r'$v\ [m/s]$')
        ax[1].legend()
        ax[1].set_xlabel(r'$t\ [s]$')
        return fig, ax
    
    def plotTaskState(self, fig, ax, q_Ts, color='r', linestyle='--'):
        ax[0].plot(q_Ts[0,:], q_Ts[1,:], color=color, linestyle=linestyle)
    
    def plotRobot(self, fig, ax, ts, qs, q_Ts, us, num=8):
        idxs = np.rint(np.linspace(0, len(ts)-1, num)).astype(int)
        max_x, max_z = max(qs[0]), max(qs[1])
        for i in idxs:
            self.display(fig, ax, qs[:,i:i+1], q_Ts[:,i:i+1], max_x, max_z)  # display takes configuration state
        quad = mlines.Line2D([], [], color='blue', marker='o', linestyle='-', markersize=5, label='quadrotor')
        if self.robot.__class__.__name__==util.RobotInfo.AM.value.robot_type:
            tool = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5, label='tool tip')
            ax[0].legend(handles=[quad, tool])
        else: ax[0].legend(handles=[quad])
    
    
class PlotPassiveSimResults(PlotSimResults):
    def __init__(self, planner, controller, robot):
        if robot.__class__.__name__==util.RobotInfo.AM.value.robot_type: self.m = robot.dynamics.m_t  # task space
        elif robot.__class__.__name__==util.RobotInfo.QUAD.value.robot_type: self.m = robot.dynamics.m_total  # configuration space
        else: raise NotImplementedError("Unknown robot class in Plotter!")
        super().__init__(planner, controller, robot)

    def plotVelocityTracking(self, fig, ax, ts, qs, q_dots, q_T_dots, q_r_dots, Vs):
        # fig, ax = super().plotVelocityTracking(fig, ax, ts, q_T_dots, Vs)
        K_bar = self.compute_kinetic_energy_vectorized(qs, q_dots, q_T_dots, q_r_dots)[-1]
        beta = np.sqrt(K_bar/self.controller.E_bar).squeeze()
        q_bar_dots = np.vstack((q_T_dots, q_r_dots))
        beta_error = q_bar_dots - beta*Vs
        ax[0].plot(ts, beta_error[0], label=r'$e_{\beta,x}$')
        # ax[0].plot(ts, beta*Vs[0,:], label=r'$\beta V_x$')  # plot Beta velocity tracking
        ax[0].plot(ts, np.zeros_like(ts), 'r--')
        ax[1].plot(ts, beta_error[1], label=r'$e_{\beta,z}$')
        # ax[1].plot(ts, beta*Vs[1,:], label=r'$\beta V_z$')
        ax[1].plot(ts, np.zeros_like(ts), 'r--')
        ax[0].legend()
        ax[1].legend()
        return fig, ax
    
    def plotPassivity(self, fig, ax, ts, qs, q_dots, q_Ts, q_T_dots, q_r_dots, f_es):
        for i in range(2):
            ax[0].plot(ts, f_es[i])
        ax[0].legend([r'$\tilde{f}_{e,x}$', r'$\tilde{f}_{e,z}$'], loc='lower right')
        ax[0].set_ylabel(r'$[N]$')
        K, K_r, K_bar = self.compute_kinetic_energy_vectorized(qs, q_dots, q_T_dots, q_r_dots)
        K_bar_dot = np.gradient(K_bar.squeeze(), ts)
        f_e_power =  np.array([q_T_dots[:,i].reshape(-1,1).T@f_es[:,i].reshape(-1,1) for i in range(q_T_dots.shape[1])]).squeeze()

        ax[1].plot(ts, K, label=r'$K$')
        ax[1].set_ylabel(r'$[J]$')
        ax[1].plot(ts, K_r, label=r'$K_r$')
        ax[1].legend()

        ax[2].plot(ts, f_e_power, 'r', label=r'$\dot{\boldsymbol{x}}^T \tilde{\boldsymbol{F}}_e$')
        ax[2].plot(ts, K_bar_dot, 'c--', label=r'$\dot{\bar{K}}$')
        ax[2].set_ylabel(r'$[W]$')
        ax[2].legend()
        ax[2].set_xlabel(r'$t\ [s]$')
        # ax[2].set_ylim(top=0.08)
        ax[2].set_xlim(0, ts[-1])
        plt.suptitle(r'$\text{Passivity Analysis}$')
        return fig, ax
    
    def plotEbarBetaError(self, E_bars):
        fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
        m_bar = np.array([[self.roboot.m, 0, 0], [0, self.roboot.m, 0], [0, 0, self.controller.m_r]])
        for i in range(len(E_bars)):
            q_bar_dots = np.vstack((self.q_T_dots[i], self.q_r_dots[i]))
            K_bar = np.concatenate([0.5*q_bar_dot.T@m_bar@q_bar_dot for q_bar_dot in q_bar_dots.T[:,:,None]])
            beta = np.sqrt(K_bar/E_bars[i]).squeeze()
            e_bar_beta = q_bar_dots - beta*self.Vs[i]
            ax[0].plot(self.ts, beta, label=r'$\beta:\bar{E} = ' + str(E_bars[i]) + '$')
            ax[1].plot(self.ts, np.linalg.norm(e_bar_beta, axis=0), label=r'$\Vert \bar{\boldsymbol{e}}_{\beta} \Vert:\bar{E} = ' + str(E_bars[i]) + '$')
        ax[0].legend()
        ax[1].plot(self.ts, np.zeros(self.ts.shape), '--')
        ax[1].set_ylabel(r'$[m]$')
        ax[1].set_xlabel(r'$t\ [s]$')
        ax[1].legend()
        ax[1].set_xlim(0, self.ts[-1])
        plt.suptitle(r'$\bar{E}\ \text{Beta Error Comparison}$')
        return fig, ax
    
    def plotEbarPassivity(self, E_bars):
        fig, ax = plt.subplots(2, 1, figsize=(16,9), sharex=True)
        m_bar = np.array([[self.m, 0, 0], [0, self.m, 0], [0, 0, self.controller.m_r]])
        m = np.array([[self.m, 0], [0, self.m]])

        for i in range(len(E_bars)):
            q_bar_dots = np.vstack((self.q_T_dots[i], self.q_r_dots[i]))
            K = np.concatenate([0.5*q_T_dot.T@m@q_T_dot for q_T_dot in self.q_T_dots[i].T[:,:,None]])
            K_r = np.array([0.5*self.controller.m_r*q_r_dot**2 for q_r_dot in self.q_r_dots[i]])
            K_bar = np.concatenate([0.5*q_bar_dot.T@m_bar@q_bar_dot for q_bar_dot in q_bar_dots.T[:,:,None]])
            K_bar_dot = np.gradient(K_bar.squeeze(), self.ts)
            f_e_power =  np.array([self.q_T_dots[i][:,j].reshape(-1,1).T@self.f_es[i][:,j].reshape(-1,1) for j in range(self.q_T_dots[i].shape[1])]).squeeze()

            ax[0].plot(self.ts, K, label=r'$K:\bar{E} = ' + str(E_bars[i]) + '$')
            ax[0].plot(self.ts, K_r, label=r'$K_r:\bar{E} = ' + str(E_bars[i]) + '$')
            ax[1].plot(self.ts, f_e_power, label=r'$\dot{\boldsymbol{x}}^T \tilde{\boldsymbol{F}}_e: \bar{E} = ' + str(E_bars[i]) + '$')
            ax[1].plot(self.ts, K_bar_dot, '--', label=r'$\dot{\bar{K}}: \bar{E} = ' + str(E_bars[i]) + '$')
        for i in range(2):
            plt.sca(ax[i])
            lines = ax[i].get_lines()
            include1 = [0,2,4]
            include2 = [1,3,5]
            legend1 = plt.legend([lines[i] for i in include1],[lines[i].get_label() for i in include1], loc=2)
            legend2 = plt.legend([lines[i] for i in include2],[lines[i].get_label() for i in include2], loc=1)
            plt.gca().add_artist(legend1)
            plt.gca().add_artist(legend2)
        ax[0].set_ylabel(r'$[J]$')
        ax[1].set_ylabel(r'$[W]$')
        ax[1].set_xlabel(r'$t\ [s]$')
        ax[1].set_xlim(5, 25)
        plt.suptitle(r'$\bar{E}\ \text{Passivity Comparison}$')
        return fig, ax
    
    def plotPowerAnnihilation(self, fig, ax, ts, Fs, F_rs, q_T_dots, q_r_dots):
        f_power =  np.array([q_T_dots[:,i:i+1].T@Fs[:,i:i+1] for i in range(q_T_dots.shape[1])]).squeeze()
        f_r_power = np.array(q_r_dots)*np.array(F_rs)

        ax[0].plot(ts, f_r_power, label=r'$x_r f_r$')
        ax[0].plot(ts, f_power, label=r'$\dot{\boldsymbol{x}}^T \tilde{\boldsymbol{F}}$')
        ax[0].legend()
        ax[0].set_ylabel(r'$[W]$')

        ax[1].plot(ts, f_power + f_r_power, linestyle=(0,(1,1)), label=r'$\dot{\boldsymbol{x}}^T \tilde{\boldsymbol{F}} + x_r f_r$')
        ax[1].legend()
        # ax[1].set_ylim(-0.02, 0.02)
        ax[1].set_ylabel(r'$[W]$')
        ax[1].set_xlabel(r'$t\ [s]$')
        ax[1].set_xbound(0, ts[-1])
        plt.suptitle(r'$\text{Power Annihilation}$')
        return fig, ax
    
    def compute_kinetic_energy_vectorized(self, qs, q_dots, q_T_dots, q_r_dots):
        cols = qs.shape[1]
        q_bar_dots = np.vstack((q_T_dots, q_r_dots))
        M_bar = [self.controller.computeDynamics(qs[:, i:i+1], q_dots[:, i:i+1])[0] for i in range(cols)]
        K = [0.5 * q_T_dots[:, i:i+1].T @ M_i[:-1,:-1] @ q_T_dots[:, i:i+1] for i, M_i in enumerate(M_bar)]
        K_r = [0.5*M_i[-1,-1]*q_r_dots[i]**2 for i, M_i in enumerate(M_bar)]
        K_bar = [0.5 * q_bar_dots[:, i:i+1].T @ M_i @ q_bar_dots[:, i:i+1] for i, M_i in enumerate(M_bar)]
        return np.squeeze(K), np.squeeze(K_r), np.squeeze(K_bar)
    

class ControlComparison(PlotPassiveSimResults):
    def __init__(self, controllers, data):
        self.controllers, self.data = controllers, data  # list of controller objects to compare
        fp.setupPlotParams()
    
    def plot_beta(self, fig, axs, ts, qs, q_dots, q_T_dots, q_r_dots):
        K_bar = self.compute_kinetic_energy_vectorized(qs, q_dots, q_T_dots, q_r_dots)[-1]
        beta = np.sqrt(K_bar/self.controller.E_bar).squeeze()
        for ax in axs: ax.plot(ts, beta, label=r'$\beta$')   
        return fig, axs

    def plot_error(self, fig, ax, control_type, ts, qs, q_dots, q_T_dots, q_r_dots, Vs):
        dim = ['x', 'z']
        if control_type in [util.ControllerInfo.PVFC, util.ControllerInfo.AUGMENTEDPD]:  # augmented controllers
            K_bar = self.compute_kinetic_energy_vectorized(qs, q_dots, q_T_dots, q_r_dots)[-1]
            beta = np.sqrt(K_bar/self.controller.E_bar).squeeze()
            q_T_bar_dots = np.vstack((q_T_dots, q_r_dots))
            beta_error = q_T_bar_dots - beta*Vs
            ax[0].plot(ts, beta_error[0], label=r'$\bar{e}_{\beta,'+dim[0]+',' + control_type.name + r'}$')
            ax[0].plot(ts, beta, label=r'$\beta_{' + control_type.name + r'}$')
            ax[1].plot(ts, beta_error[1], label=r'$\bar{e}_{\beta,'+dim[1]+',' + control_type.name + r'}$')
            ax[1].plot(ts, beta, label=r'$\beta_{' + control_type.name + r'}$')
        else:
            error = q_T_dots - Vs[:-1,:]
            ax[0].plot(ts, error[0], label=r'$e_{' +dim[0]+','+ control_type.name + r'}$')
            ax[1].plot(ts, error[1], label=r'$e_{' +dim[1]+','+ control_type.name + r'}$')
        ax[0].plot(ts, np.zeros_like(ts), 'r--')
        ax[1].plot(ts, np.zeros_like(ts), 'r--')
        ax[0].set_ylabel(r'$e_'+dim[0]+'\ [m]$', fontsize=30)
        ax[1].set_ylabel(r'$e_'+dim[1]+'\ [m]$', fontsize=30)
        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel(r'$t\ [s]$', fontsize=30)
        plt.xlim(0,max(np.rint(ts)))
        return fig, ax

    def plot_comparo(self, plot_error=True, plot_velocity_tracking=True, controller_types=None):
        if controller_types == None: raise NotImplementedError("Must specify controller types to plot.")
        fig1, axs1 = util.create_fig(2, 1)
        error_title = 'Error Comparison'
        for i, control_type in enumerate(controller_types):
            self.controller = self.controllers[i]  # controller object
            if plot_error:    
                fig1, axs1 = self.plot_error(fig1, axs1, control_type, self.data['ts'][control_type], self.data['qs'][control_type], self.data['q_dots'][control_type], self.data['q_T_dots'][control_type], self.data['q_r_dots'][control_type], self.data['Vs'][control_type])
            if plot_velocity_tracking:
                fig, axs = util.create_fig(2, 1)
                fig, axs = self.plot_velocity(fig, axs, self.data['ts'][control_type], self.data['q_T_dots'][control_type], self.data['Vs'][control_type])
                if control_type != util.ControllerInfo.PD: fig, axs = self.plot_beta(fig, axs, self.data['ts'][control_type], self.data['qs'][control_type], self.data['q_dots'][control_type], self.data['q_T_dots'][control_type], self.data['q_r_dots'][control_type])
                fig.suptitle(control_type.value, fontsize=30)
                for ax in axs: ax.legend()
            else: raise NotImplementedError("Nothing to plot: set one of the plotting flags to True.")
        fig1.suptitle(error_title, fontsize=30)
            

class VizVelocityField(PlotSimResults):
    def __init__(self, planner):
        self.planner = planner
        fp.setupPlotParams()

    def plotVelocityField(self, fig, ax, x_T, z_T):
        x_max = np.amax(x_T)
        p_x, p_z, V_x, V_z = self.createGrid(x_T,z_T)
        for i in range(p_x.shape[0]):
            for j in range(p_x.shape[1]):
                q_T = np.array([p_x[i, j], p_z[i, j]]).reshape(-1,1)
                V = self.planner.plotStep(q_T)
                V_x[i,j] = V[0]
                V_z[i,j] = V[1]
        ax[0].quiver(p_x, p_z, V_x, V_z, color='k', pivot='middle', alpha=0.3, angles='xy', scale_units='xy', scale=50, width=0.0005*x_max)
        ax[0].set_xlabel(r'$x\ [m]$')
        ax[0].set_ylabel(r'$z\ [m]$')
        return fig, ax
    
    def plotTest(self, fig, ax, x_T, y_T):
        ax[0].plot(x_T, y_T, color='black', linestyle='-')