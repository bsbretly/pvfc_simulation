import numpy as np
import sympy as sp
import utilities as util
DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 1/DEG_TO_RAD


class VelocityPlanner: 
    def __init__(self, base_planner_params, *args):
        if type(base_planner_params).__name__ == 'BaseQuadPlannerParams': 
            self.task_space = False
            self.m, self.m_r, self.E_bar = base_planner_params
        elif type(base_planner_params).__name__ == 'BaseAMPlannerParams': 
            self.task_space = True
            self.m, self.m_r, self.tool_length, self.E_bar = base_planner_params
        else: raise NotImplementedError("Planner parameters not implemented.")
        
        self.q_sym = sp.symbols('x z')
        self.setParams(*args)
        self.init_V_sym()
        self.lambdify_augment_V()

    def step(self, q, q_dot):
        '''
        input: 
            robot position, robot velocity, reservoir velocity - q, q_dot
        output:
            desired augmented velocity field and desired augmented velocity field gradient - Vbar, Vbar_dot 
        '''   
        if self.task_space:     
            q, q_dot = util.configToTask(q, q_dot, self.tool_length)
        return self.computeVelocityField(q), self.computeVelocityFieldGradient(q, q_dot)
    
    def plotStep(self, q_T):
        return self.computeVelocityField(q_T)
        1
    def computeVelocityField(self, q_T):
        try:
            Vbar = self.field(float(q_T[0]), float(q_T[1]))
        except Exception as e:
            print(e)
            return np.zeros((3,1))
        if np.isnan(Vbar).any():
            return np.zeros((3,1))
        return Vbar.reshape(-1, 1)

    def computeVelocityFieldGradient(self, q_T, q_T_dot):
        dV_bar_dqbar = np.zeros((3, 3))
        try:
            dV_bar_dq = self.field_jacobian(float(q_T[0]), float(q_T[1]))
            dV_bar_dqbar[0:3,0:2] = dV_bar_dq
            # The last column of dV_bar_dqbar is all zeros because the velocity field is not a function of the reservoir state
            q_temp = np.array([q_T_dot[0], q_T_dot[1], np.array([0])])
            Vbar_dot = dV_bar_dqbar@q_temp
        except Exception as e:
            print(e)
            return np.zeros((3,1))
        if np.isnan(Vbar_dot).any():
            return np.zeros((3,1))
        
        return Vbar_dot.reshape(-1, 1)
    
    def lambdify_augment_V(self):
        field_aug = self.augment_V(self.V_sym)
        self.field = sp.lambdify(self.q_sym, field_aug, modules='numpy')
        jacobian = field_aug.jacobian(self.q_sym)
        self.field_jacobian = sp.lambdify(self.q_sym, jacobian, modules='numpy')

    def augment_V(self, field):
        wheel_field = sp.sqrt((2. / self.m_r) * (sp.Matrix([self.E_bar]) - (0.5 * field.T@field*self.m)))
        augmented_field = field.row_insert(3, wheel_field)
        return augmented_field
    
    def init_V_sym(self):
        ''' this method has to initialize self.V_sym'''
        raise NotImplementedError("Must override init_V_sym")

    def setParams(self, *args):
        ''' this method has to initialize the parameters of the subclass'''
        raise NotImplementedError("Must override setParams")
    
    def __iadd__(self, other):
        self.V_sym = self.V_sym + other.V_sym
        self.lambdify_augment_V()
        return self

class PointVelocityField(VelocityPlanner):
    # Velocity field from https://ieeexplore.ieee.org/document/8779551
    def __init__(self, base_planner_params, point_planner_params):
        super().__init__(base_planner_params, point_planner_params)

    def setParams(self, point_planner_params):
        self.planar_len, self.alpha, self.x_d, self.z_d = point_planner_params
        
    def init_V_sym(self): 
        Q = sp.Matrix([self.x_d, self.z_d])
        n = (Q - sp.Matrix([self.q_sym[0] , self.q_sym[1]]))
        t_hat = sp.Matrix([0, 0])
        self.V_sym = self.planar_len * (n + self.alpha*t_hat)


class PlanarVelocityField(VelocityPlanner):
    # Velocity field from https://ieeexplore.ieee.org/document/8779551
    def __init__(self, base_planner_params, planar_planner_params):
        super().__init__(base_planner_params, planar_planner_params)

    def setParams(self, planarPlannerParams):
        self.planar_len, self.alpha = planarPlannerParams
        
    def init_V_sym(self): 
        Q = sp.Matrix([self.q_sym[0], -0.1])
        n = (Q - sp.Matrix([self.q_sym[0] , self.q_sym[1]]))
        t_hat = sp.Matrix([1, 0])
        self.V_sym = self.planar_len * (n + self.alpha*t_hat)


class RampVelocityField(VelocityPlanner):
    # Velocity field from https://ieeexplore.ieee.org/document/8779551
    def __init__(self, base_planner_params, ramp_planner_params):
        super().__init__(base_planner_params, ramp_planner_params)

    def setParams(self, rampPlannerParams):
        # ramp starts at x_s, ends at x_e, and has final height z_h
        self.planar_len, self.alpha, self.x_s, self.x_e, self.z_h = rampPlannerParams
        
    def init_V_sym(self): 
        Q = sp.Matrix([self.q_sym[0], self.q_sym[1]])
        n = (Q - sp.Matrix([self.q_sym[0] , self.q_sym[1]]))
        # tangent vector is the equation of a line
        m = self.z_h / (self.x_e - self.x_s)
        t_hat = 1/np.sqrt(1 + m**2)*sp.Matrix([1, m])
        self.V_sym = self.planar_len * (n + self.alpha*t_hat)


class SuperQuadraticField(VelocityPlanner):
    # Superquadriatic vector field from https://pure.strath.ac.uk/ws/portalfiles/portal/73873700/strathprints006243.pdf
    def __init__(self, base_planner_params, super_quadratic_params):
        super().__init__(base_planner_params, super_quadratic_params)

    def setParams(self, superQuadraticParams): 
        self.obs_x, self.obs_z, self.obs_m, self.obs_n, self.obs_L, self.obs_len = superQuadraticParams

    def init_V_sym(self):
        H = (self.q_sym[0]-self.obs_x)**self.obs_n + (self.q_sym[1]-self.obs_z)**self.obs_n
        shaping_fn = 1/(1+(1/self.obs_L)*(H**(1/self.obs_n))**self.m) 
        grad_shaping = sp.Matrix([shaping_fn]).jacobian(self.q_sym)
        repulsive_field = self.obs_len*sp.Matrix([-grad_shaping[0], -grad_shaping[1]]) 
        self.V_sym = repulsive_field
    