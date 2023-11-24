import numpy as np

def compute_transforms(q, q_dot, tool_length):
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

def config_to_task(q, q_dot, tool_length, q_ddot=None):
    '''
    input: 
        state and AM tool length - q, q_dot, tool_length
    output: 
        tool-tip position and velocity - q_T, q_T_dot
    '''
    K, J, J_dot = compute_transforms(q, q_dot, tool_length)
    if isinstance(q_ddot, type(None)): return K + q[:2], J@q_dot
    # Acceleration: J_dot@q_dot + J@q_ddot
    return K + q[:2], J@q_dot 
    
def decompose_thrust_vector(Lambda):
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

def augment_dynamics(M, C, m_r):
    M_bar = np.pad(M, ((0,1),(0,1)), 'constant')
    M_bar[-1,-1] = m_r
    C_bar = np.pad(C, ((0,1),(0,1)), 'constant')
    return M_bar, C_bar
