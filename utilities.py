import numpy as np

def computeRampParams(x_s, x_e, z_h):
    m = z_h / (x_e - x_s)
    b = -m * x_s
    return m, b

def computeFloorForce(k):
    return k*np.array([[0.], [1.]]) # floor force

def computeRampForce(k, x_s, x_e, z_h, q_T): 
    m,b = computeRampParams(x_s, x_e, z_h)
    Delta = np.abs(-m*q_T[0,0] + q_T[1,0] - b) / np.sqrt(m**2 + 1)
    n_hat = np.array([[-m], [1.]]) / np.sqrt(1 + m**2)
    return k * Delta * n_hat
     
def touchFloor(x_s, x_e, q_T):
    x_T, z_T = q_T[0,0], q_T[1,0]
    return (x_T < x_s or x_T > x_e) and z_T <= 0

def touchRamp(x_s, x_e, z_h, q_T):
    x_T, z_T = q_T[0,0], q_T[1,0]
    m,b = computeRampParams(x_s, x_e, z_h)
    z_b = m*x_T + b
    return z_T <= z_b and x_s <= x_T <= x_e

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
