import numpy as np

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

def computeRampForce(k, mu, p1, p2, q_T): 
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
        normal_force = force_scaler*n_hat
        t_hat = np.array([[1.], [m]]) / np.sqrt(1 + m**2)
        tangent_force = mu*force_scaler*t_hat
        return normal_force + tangent_force
    else: return np.zeros((2,1))

def computeTransforms(q, q_dot, tool_length):
    '''
    input: 
        configuration state and AM tool length - q, q_dot, tool_length
    output: 
        forward kinematics, jacobians - K, J, J_dot
    '''
    # forward kinematics: configuration space to task space
    K = np.array([[tool_length*np.cos(q[3,0])], 
        [-tool_length*np.sin(q[3,0])]])
    # configuration space to task space jacobian
    J = np.array([[1, 0, 0, -tool_length*np.sin(q[3,0])], 
        [0, 1, 0, -tool_length*np.cos(q[3,0])]]) 
    J_dot = np.array([[0, 0, 0, -tool_length*np.cos(q[3,0])*q_dot[3,0]], 
        [0, 0, 0, tool_length*np.sin(q[3,0])*q_dot[3,0]]])
    return K, J, J_dot

def configToTask(q, q_dot, tool_length):
    '''
    input: 
        state and AM tool length - q, q_dot, tool_length
    output: 
        tool-tip position and velocity - q_T, q_T_dot
    '''
    K, J, _ = computeTransforms(q, q_dot, tool_length)

    return K + q[:2], J@q_dot
    
def decomposeThrustVector(Lambda):
    '''
    input: 
        thrust vector - Lambda
    output: 
        desired thrust along body z-axis of quadrotor, desired pitch - thrust, theta_d
    '''
    thrust = np.linalg.norm(Lambda)
    Lambda_hat = Lambda/thrust
    theta_d = np.arctan2(Lambda_hat[0], Lambda_hat[1])[0]
    return thrust, theta_d

def augmentDynamics(M, C, m_r):
    M_bar = np.pad(M, ((0,1),(0,1)), 'constant')
    M_bar[-1,-1] = m_r
    C_bar = np.pad(C, ((0,1),(0,1)), 'constant')
    return M_bar, C_bar
