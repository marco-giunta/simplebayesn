import numpy as np

def get_mean_int(global_params: dict):
    mu = np.array([global_params['M0_int'], global_params['c0_int'], global_params['x0']]).reshape((3, 1))
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    return M @ mu

def get_cov_int(global_params: dict):
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    S = np.diag(np.array([global_params['sigma_int2'], global_params['sigmac_int2'], global_params['sigmax2']]))
    C = M @ S @ M.T
    return C

def get_mean_int_numeric(global_params: dict):
    mu = np.array([global_params['M0_int'], global_params['c0_int'], global_params['x0']]).reshape((3, 1))
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    return M @ mu

def get_cov_int_numeric(global_params: dict):
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    S = np.diag(np.array([global_params['sigma_int2'], global_params['sigmac_int2'], global_params['sigmax2']]))
    C = M @ S @ M.T
    return C

def get_cov_int_analytic(global_params: dict):
    alpha        = global_params['alpha']
    beta_int     = global_params['beta_int']
    alphac_int   = global_params['alphac_int']
    sigma_int2   = global_params['sigma_int2']
    sigmac_int2  = global_params['sigmac_int2']
    sigmax2      = global_params['sigmax2']

    C = np.zeros((3, 3))

    C[0, 0] = ((alpha + beta_int * alphac_int) ** 2) * sigmax2 \
               + (beta_int ** 2) * sigmac_int2 \
               + sigma_int2

    C[1, 1] = (alphac_int ** 2) * sigmax2 + sigmac_int2
    C[2, 2] = sigmax2

    C[0, 1] = (alpha + beta_int * alphac_int) * alphac_int * sigmax2 + beta_int * sigmac_int2
    C[1, 0] = C[0, 1]

    C[0, 2] = (alpha + beta_int * alphac_int) * sigmax2
    C[2, 0] = C[0, 2]

    C[1, 2] = alphac_int * sigmax2
    C[2, 1] = C[1, 2]

    return C

def get_mean_int_analytic(global_params):
    M0_int     = global_params['M0_int']
    c0_int     = global_params['c0_int']
    x0         = global_params['x0']
    alpha      = global_params['alpha']
    beta_int   = global_params['beta_int']
    alphac_int = global_params['alphac_int']

    mean = np.zeros((3, 1))

    mean[0, 0] = M0_int + beta_int * c0_int + (alpha + beta_int * alphac_int) * x0
    mean[1, 0] = c0_int + alphac_int * x0
    mean[2, 0] = x0
    
    return mean