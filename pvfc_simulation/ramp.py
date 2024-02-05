import numpy as np

class Ramp():
    def __init__(self, p1=None, p2=None, ramp_k=None, ramp_mu=None):
        self.p1, self.p2 = p1, p2
        self.ramp_k, self.ramp_mu = ramp_k, ramp_mu
        self.m, self.b = self.computeRampParams(p1, p2)
    
    def computeRampParams(self, p1, p2):
        '''
        input: 
            2 (x,z) points on a line - p1, p2
        output: 
            slope and z-intercept of line going through p1, p2 - m, b
        '''
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m*p1[0]
        return m, b

    def contactRamp(self, q_T):
        '''
        input: 
            Two (x,z) points on a line and end-effector (x,z) poistion - p1, p2, q_T
        output: 
            True if end-effector is in contact with the ramp, else False - bool
        '''
        x_T, z_T = q_T[0,0], q_T[1,0]
        z_ramp = self.m*x_T + self.b

        return self.p1[0] <= x_T <= self.p2[0] and z_T <= z_ramp

    def compute_ramp_force(self, q_T, q_dot_T): 
        '''
        input: 
            ramp stiffness, coefficient of kinetic friction, points and end-effector posiion - k, mu, p1, p2, q_T
        output: 
            (x,z) force vector - F
        '''
        if self.contactRamp(q_T): 
            Delta = np.abs(-self.m*q_T[0,0] + q_T[1,0] - self.b) / np.sqrt(self.m**2 + 1) # distance into the ramp
            n_hat = np.array([[-self.m], [1.]]) / np.sqrt(1 + self.m**2)
            force_scaler = self.ramp_k*Delta
            # normal_force = force_scaler*n_hat + b*q_dot_T*n_hat # TODO: implement damping
            normal_force = force_scaler*n_hat
            t_hat = np.array([[1.], [self.m]]) / np.sqrt(1 + self.m**2)
            tangent_force = self.ramp_mu*force_scaler*t_hat
            return normal_force + tangent_force
        else: return np.zeros((2,1))