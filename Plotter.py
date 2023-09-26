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
        
        ax.quiver(p_x, p_z, V_x, V_z, color='k', pivot='middle', alpha=0.3, angles='xy', scale_units='xy', scale=scale, width=0.001*min(x_max, z_max))
        return fig, ax
    
    def createGrid(self, x, y):
        p_x, p_y = np.meshgrid(x, y, indexing='ij')
        return p_x, p_y, np.zeros((x.size,y.size)), np.zeros((x.size,y.size))
    
    def plotConfigState(self, fig, ax, qs, color='blue', linestyle='--'):
        ax.plot(qs[0,:], qs[1,:], color=color, linestyle=linestyle)

    def display(self, fig, ax, q, max_x):
        if self.robot.__class__.__name__ == 'AerialManipulator': 
            x, z, theta, Beta = q[0], q[1], q[2], q[3]
            q_T, _ = util.configToTask(q, np.zeros_like(q), self.robot.dynamics.tool_length)
            ax.plot([x, q_T[0]], [z, q_T[1]], 'green')  # plot tool
            ax.plot(q_T[0], q_T[1], 'go', label='tool tip')  # plot end-effector
        else: x, z, theta = q[0], q[1], q[2]
        wing_span = 0.05*max_x
        ax.plot(x, z, 'bo', label='quadrotor')
        quad_x = [x-wing_span*np.cos(-theta), x+wing_span*np.cos(-theta)]
        quad_z = [z-wing_span*np.sin(-theta), z+wing_span*np.sin(-theta)]
        ax.plot(quad_x, quad_z, 'blue')
        ax.set_xlabel(r'$x\ [s]$')
        ax.set_ylabel(r'$z\ [m]$')
        return fig, ax

    def plotRamp(self, fig, ax, q_Ts, color='black', linestyle='-'):
        m,b = util.computeRampParams(self.planner.p1, self.planner.p2)
        x = np.linspace(0, q_Ts[0,-1], 100)
        ax.plot(x, m*x + b, color=color, linestyle=linestyle, linewidth=2)
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

    def plotVelocityTracking(self, fig, ax, ts, q_T_dots, Vs):
        ax[0].plot(ts, q_T_dots[0,:], 'b', label=r'$\dot{x}_T$')
        ax[0].plot(ts, Vs[0,:], 'g--', label=r'$V_x$')
        ax[0].set_ylabel(r'$v\ [m/s]$')
        ax[0].legend()
        
        ax[1].plot(ts, q_T_dots[1,:], 'b', label=r'$\dot{z}_T$')
        ax[1].plot(ts, Vs[1,:], 'g--', label=r'$V_z$')
        ax[1].set_ylabel(r'$v\ [m/s]$')
        ax[1].legend()
        ax[1].set_xlabel(r'$t\ [s]$')
        return fig, ax
    
    def plotTaskState(self, fig, ax, q_Ts, color='r', linestyle='--'):
        ax.plot(q_Ts[0,:], q_Ts[1,:], color=color, linestyle=linestyle)
    
    def plotRobot(self, fig, ax, ts, qs, us, num=8):
        idxs = np.rint(np.linspace(0, len(ts)-1, num)).astype(int)
        max_x = max(qs[0])
        for i in idxs:
            self.display(fig, ax, qs[:,i:i+1], max_x)
        quad = mlines.Line2D([], [], color='blue', marker='o', linestyle='-', markersize=5, label='quadrotor')
        if self.robot.__class__.__name__=='AerialManipulator':
            tool = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=5, label='tool tip')
            ax.legend(handles=[quad, tool])
        else: ax.legend(handles=[quad])


class PlotPassiveSimResults(PlotSimResults):
    def __init__(self, planner, controller, robot):
        super().__init__(planner, controller, robot)

    def plotVelocityTracking(self, fig, ax, ts, q_T_dots, q_r_dots, Vs):
        fig, ax = super().plotVelocityTracking(fig, ax, ts, q_T_dots, Vs)
        q_bar_dots = np.vstack((q_T_dots, q_r_dots))
        m_bar = np.array([[self.robot.dynamics.m, 0, 0], [0, self.robot.dynamics.m, 0], [0, 0, self.controller.m_r]])
        K_bar = np.concatenate([0.5*q_bar_dot.T@m_bar@q_bar_dot for q_bar_dot in q_bar_dots.T[:,:,None]])
        Beta = np.sqrt(K_bar/self.controller.E_bar).squeeze()
        ax[0].plot(ts, Beta*Vs[0,:], 'r--', label=r'$\beta V_x$')  # plot Beta velocity tracking
        ax[1].plot(ts, Beta*Vs[1,:], 'r--', label=r'$\beta V_z$')
        ax[0].legend()
        ax[1].legend()
        return fig, ax
    
    def plotPassivity(self, fig, ax, ts, q_T_dots, q_r_dots, f_es):
        for i in range(2):
            ax[0].plot(ts, f_es[i])
        ax[0].legend([r'$\tilde{f}_{e,x}$', r'$\tilde{f}_{e,z}$'], loc='lower right')
        ax[0].set_ylabel(r'$[N]$')
        q_bar_dots = np.vstack((q_T_dots, q_r_dots))
        m_bar = np.array([[self.robot.dynamics.m, 0, 0], [0, self.robot.dynamics.m, 0], [0, 0, self.controller.m_r]])
        m = np.array([[self.robot.dynamics.m, 0], [0, self.robot.dynamics.m]])
        K = np.concatenate([0.5*q_T_dot.T@m@q_T_dot for q_T_dot in q_T_dots.T[:,:,None]])
        K_r = np.array([0.5*self.controller.m_r*q_r_dot**2 for q_r_dot in q_r_dots])
        K_bar = np.concatenate([0.5*q_bar_dot.T@m_bar@q_bar_dot for q_bar_dot in q_bar_dots.T[:,:,None]])
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
        m_bar = np.array([[self.robot.dynamics.m, 0, 0], [0, self.robot.dynamics.m, 0], [0, 0, self.controller.m_r]])
        m = np.array([[self.robot.dynamics.m, 0], [0, self.robot.dynamics.m]])

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
        f_power =  np.array([q_T_dots[:,i].reshape(-1,1).T@Fs[:,i].reshape(-1,1) for i in range(q_T_dots.shape[1])]).squeeze()
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
        ax.quiver(p_x, p_z, V_x, V_z, color='k', pivot='middle', alpha=0.3, angles='xy', scale_units='xy', scale=50, width=0.0005*x_max)
        ax.set_xlabel(r'$x\ [s]$')
        ax.set_ylabel(r'$z\ [m]$')
        return fig, ax
    
    def plotTest(self, fig, ax, x_T, y_T):
        ax.plot(x_T, y_T, color='black', linestyle='-')