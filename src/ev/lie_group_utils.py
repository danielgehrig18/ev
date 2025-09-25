import numpy as np
import numba
import scipy.linalg
from scipy.linalg import block_diag


#@numba.jit(nopython=True)
def Ad_vee_gal_cross_gal(x):
    m = (len(x) - 20)//10
    blocks = [Ad_vee_gal(exp_gal(x[10*i:10*i+10])) for i in range(m)]

    block = np.zeros((20, 20))
    Y, b = exp_gal_cross_gal(x[-20:-10], x[-10:])
    A_Y = Ad_vee_gal(Y)
    block[:10, :10] = A_Y
    block[10:, 10:] = A_Y
    block[10:,:10] = ad_vee_gal(b) @ A_Y
    blocks.append(block)

    out = block_diag(*blocks)

    return out

def inv_group(X):
    Xs, Xi = X
    Xs_inv = inv_gal_gal(Xs)
    Xi_inv = np.linalg.inv(Xi)
    return Xs_inv, Xi_inv

#@numba.jit(nopython=True)
def control_input_action_group(X, u):
    us, ui = u
    (C, gamma), Xi = X
    w, tau = us[:10], us[10:]

    Ad = Ad_vee_gal(np.linalg.inv(C))
    w_circ = Ad @ (w - gamma)
    tau_circ = Ad @ tau

    us_circ = np.concatenate([w_circ, tau_circ])
    return us_circ, ui.copy()

@numba.jit(nopython=True)
def ad_vee_gal(b):
    out = np.zeros((10,10))
    w = b[0:3]
    v = b[3:6]
    r = b[6:9]
    a = b[9]

    out[0:3,0:3] = hat_so3(w)

    out[3:6,0:3] = hat_so3(v)
    out[3:6,3:6] = hat_so3(w)

    out[6:9,0:3] = hat_so3(r)
    out[6:9,3:6] = - np.eye(3) * a
    out[6:9,6:9] = hat_so3(w)
    out[6:9, 9] = v

    return out


def log_gal_cross_gal(X):
    C, gamma = X
    e_C = log_gal(C)
    e_gamma = np.linalg.inv(J_L_gal(e_C)) @ gamma
    return (e_C, e_gamma)


#@numba.jit(nopython=True)
def log_gal_multi(X):
    if len(X.shape) == 2:
        return log_gal(X)
    return np.stack([log_gal(x) for x in X])

#numba.jit(nopython=True)
def exp_gal_multi(X):
    if len(X.shape) == 1:
        return exp_gal(X)
    return np.stack([exp_gal(x) for x in X])

#@numba.jit(nopython=True)
def log_gal(X):
    A = X[:3,:3]
    a = X[:3,3]
    b = X[:3,4]
    c = X[3,4]

    log_A = log_so3(A)
    Gamma1, Gamma2 = Gamma1_Gamma2_gal(log_A)
    Gamma1_inv = np.linalg.inv(Gamma1)
    Xi = (b - c * Gamma2 @ Gamma1_inv @ a)

    out = np.zeros(10)
    out[0:3] = log_A
    out[3:6] = Gamma1_inv @ a
    out[6:9] = Gamma1_inv @ Xi
    out[9] = c
    assert not np.isnan(out).any()
    return out

@numba.jit(nopython=True)
def rho(C, C_i, y):
    return log_gal(np.linalg.inv(C_i) @ exp_gal(y) @ C)

#@numba.jit(nopython=True)
def vee_so3(w):
    return np.stack([
        w[...,2,1], w[...,0,2], w[...,1,0]
    ], axis=-1)

#@numba.jit(nopython=True)
def log_so3(R):
    trace = np.broadcast_to(np.einsum("...ii->...", R)[...,None,None], R.shape)
    cos_theta = (trace - 1 ) / 2

    if type(cos_theta) is np.float64 or type(cos_theta) is np.float32:
        cos_theta = min([1, cos_theta])
        cos_theta = max([-1, cos_theta])
    else:
        cos_theta[(cos_theta <= -1)] = -1
        cos_theta[(cos_theta >= 1)] = 1

    theta = np.arccos(cos_theta)

    out = (R - np.swapaxes(R, axis2=-1, axis1=-2))/2
    mask = theta>=1e-4
    out[mask] *= theta[mask] / np.sin(theta[mask])

    return vee_so3(out)

#@numba.jit(nopython=True)
def J_L_so3(w):
    w_hat = hat_so3(w)
    theta = np.linalg.norm(w, axis=-1)

    k1 = np.where(theta<1e-4, 1/2, (1-np.cos(theta)) / theta**2)
    k2 = np.where(theta<1e-4, 1/6, (theta - np.sin(theta)) / theta**3)

    V = np.eye(3) + k1[...,None,None] * w_hat + k2[...,None,None] * w_hat @ w_hat

    return V

#@numba.jit(nopython=True)
def log_se3(x):
    R = x[...,:3,:3]
    t = x[...,:3, 3]

    w = log_so3(R)
    V = J_L_so3(w)
    t = (np.linalg.inv(V) @ t[...,None])[...,0]

    return np.concatenate([w, t], axis=-1)

@numba.jit(nopython=True)
def product_gal(X1, X2):
    Y1, b1 = X1
    Y2, b2 = X2

    Y = Y1 @ Y2
    b = b1 + Ad_vee_gal(Y1) @ b2

    return (Y, b)

def product_group(X1, X2):
    x1, x1_i = X1
    x2, x2_i = X2
    x = product_gal(x1, x2)
    x_i = np.einsum("...ij,...jk->...ik", x1_i, x2_i)
    return (x, x_i)

@numba.jit(nopython=True)
def Omega_gal(w, v, r, a):
    return Q1_gal(w, r) - a * Q2_gal(w, v)

@numba.jit(nopython=True)
def Q1_gal(w, z):
    z_hat = hat_so3(z)
    w_hat = hat_so3(w)

    wn = np.linalg.norm(w)+1e-18

    c = np.cos(wn)
    s = np.sin(wn)
    wn2, wn3, wn4, wn5, wn6, wn7 = wn**2, wn**3, wn**4, wn**5, wn**6, wn**7

    k1 = (wn - s) / wn3
    k2 = (wn2 + 2 * c - 2) / (2 * wn4)
    k3 = (2 * wn - 3 * s + wn * c) / (2 * wn5)

    wz = w_hat @ z_hat
    zw = z_hat @ w_hat
    wzw = w_hat @ zw
    wwz = w_hat @ wz
    zww = zw @ w_hat
    wzww = wzw @ w_hat
    wwzw = w_hat @ wzw

    return 0.5 * z_hat + k1 * (wz + zw + wzw)  + k2 * (wwz + zww - 3 * wzw) + k3 * (wzww + wwzw)

@numba.jit(nopython=True)
def Q2_gal(w, z):
    z_hat = hat_so3(z)
    w_hat = hat_so3(w)

    wn = np.linalg.norm(w)+1e-18

    c = np.cos(wn)
    s = np.sin(wn)
    wn2, wn3, wn4, wn5, wn6, wn7 = wn**2, wn**3, wn**4, wn**5, wn**6, wn**7

    k1 = (wn2 + 2 * c  - 2 ) / (2 * wn4)
    k2 = (wn3 - 6 * wn + 6 * s) / (6 * wn5)
    k3 = (-2 * c - wn * s + 2) / wn4
    k4 = (wn3 + 6 * wn * c + 6 * wn - 12 * s) / (6 * wn5)
    k5 = (-3 * wn * c - (wn2 - 3) * s) / (4 * wn5)
    k6 = (wn * c + 2 * wn - 3 * s) / (4 * wn5)
    k7 = ((wn2 - 8) * c - 5 * wn * s + 8) / (4 * wn6)
    k8 = (2 * wn3 + 15 * wn * c + 3 * (wn2 - 5) * s) / (12 * wn7)

    zw = z_hat @ w_hat
    wz = w_hat @ z_hat
    zww = zw @ w_hat
    wwz = w_hat @ wz
    wzw = wz @ w_hat
    wwzw = w_hat @ wzw
    wzww = wzw @ w_hat
    wwzww = w_hat @ wzww

    return 1/6 * z_hat + k1 * zw + k2 * zww + k3 * wz + k4 * wwz + k5 * wzw + k6 * wwzw + k7 * wzww + k8 * wwzww

@numba.jit(nopython=True)
def Q3_gal(w, z):
    z_hat = hat_so3(z)
    w_hat = hat_so3(w)

    wn = np.linalg.norm(w)

    c = np.cos(wn)
    s = np.sin(wn)
    wn2, wn3, wn4, wn5, wn6, wn7 = wn**2, wn**3, wn**4, wn**5, wn**6, wn**7

    k1 = (-2 * c - wn * s + 2) / wn4
    k2 = (wn2 + 2 * c  - 2 ) / (2 * wn4)
    k3 = (wn3 + 6*wn*c + 6*wn - 12*s) / (6 * wn5)
    k4 = (12*s - 12*wn*c - 3*wn2*s - wn3) / (6 * wn5)
    k5 = (wn3 - 6 * wn + 6 * s) / (6 * wn5)
    k6 = (4 + wn2 + wn2*c - 4*wn*s - 4*c) / (4 * wn6)
    k7 = k6

    zw = z_hat @ w_hat
    wz = w_hat @ z_hat
    zww = zw @ w_hat
    wwz = w_hat @ wz
    wzw = wz @ w_hat
    wwzw = w_hat @ wzw
    wzww = wzw @ w_hat

    return 1/6 * z_hat + k1 * wz + k2 * zw + k3 * wwz + k4 * wzw + k5 * zww +  k6 * wwzw + k7 * wzww




@numba.jit(nopython=True)
def U1_gal(w):
    wn = np.linalg.norm(w) + 1e-18
    w_hat = hat_so3(w)

    s = np.sin(wn)
    c = np.cos(wn)

    wn2, wn3, wn4 = wn**2, wn**3, wn**4

    k1 = (s - wn * c) / wn3
    k2 = (wn2 - 2 * wn * s - 2 * c + 2) / (2 * wn4)

    return 0.5 * np.eye(3) + k1 * w_hat + k2 * w_hat @ w_hat

@numba.jit(nopython=True)
def J_L_gal(x):
    w = x[:3]
    v = x[3:6]
    r = x[6:9]
    a = x[9]

    out = np.eye(10)
    Gamma1, Gamma2 = Gamma1_Gamma2_gal(w)
    out[:3, :3] = Gamma1
    out[3:6,3:6] = Gamma1
    out[6:9,6:9] = Gamma1
    out[6:9,9] = Gamma2 @ v

    out[3:6,:3] = Q1_gal(w, v)
    out[6:9,:3] = Omega_gal(w, v, r, a)
    out[6:9,3:6] = -a * U1_gal(w)

    return out

def jacobians_group(X, u, g, dt):
    u_s, u_i = u
    x, x_i = X
    A, B, _ = jacobians_gal(x, u_s, g, dt)
    m = len(u_i)
    A_i = np.zeros((m, 10, 10))
    A_i[:,list(range(10)), list(range(10))] = 1
    B_i = np.zeros((m, 10, 10))

    return (A, A_i), (B, B_i)


def jacobians_gal(X_k, u, g, dt):
    Y_k, b_k = X_k

    w, tau = u[:10], u[10:]

    exp_g = exp_gal(g*dt)
    Y_k_1 = exp_g @ Y_k @ exp_gal((w - b_k)*dt)

    B = np.eye(20)
    Ad_vee_Y_k = Ad_vee_gal(Y_k)
    Ad_vee_Y_k_1 = Ad_vee_gal(Y_k_1)
    w_circ = Ad_vee_Y_k @ (w - b_k)

    Ad_g = Ad_vee_gal(exp_g)
    B[:10,:10] = -Ad_g @ J_L_gal(w_circ * dt) @ Ad_vee_Y_k  * dt
    B[10:,10:] = Ad_vee_Y_k_1 * dt

    A = np.eye(20)
    A[:10,:10] = Ad_g
    A[:10,10:] = Ad_g @ J_L_gal(w_circ * dt) * dt
    A[10:,10:] = Ad_g @ Ad_vee_gal(exp_gal(w_circ * dt))

    Phi_b = np.eye(20)
    #Phi_b[:10, 10:] = B[:10,:10]

    return A, B, Phi_b

def normal_coords_gal_cross_gal(xi_circ, s):
    return log_gal_cross_gal(state_action_inv_gal(xi_circ, s))


@numba.jit(nopython=True)
def inv_gal(X):
    C, gamma = X
    C_inv = np.linalg.inv(C)
    return C_inv, - Ad_vee_gal(C_inv) @ gamma

#@numba.jit(nopython=True)
def exp_se3(x):
    w = x[...,0:3]
    v = x[...,3:6]
    out = np.zeros(shape=x.shape[:-1] + (4,4))
    out[...,-1,-1] = 1

    out[...,:3,:3] = exp_so3(w)
    out[...,:3, 3] = (Gamma1(w) @ v[...,None])[...,0]
    return out

@numba.jit(nopython=True)
def ad_vee_se3(b):
    out = np.zeros(shape=b.shape[:-1] + (6,6))
    w = b[...,0:3]
    v = b[...,3:6]

    w_hat = hat_so3(w)
    out[...,0:3,0:3] = w_hat
    out[...,3:6,0:3] = hat_so3(v)
    out[...,3:6,3:6] = w_hat

    return out

#@numba.jit(nopython=True)



@numba.jit(nopython=True)
def lift_discrete_gal(xi, u, g, dt):
    w, tau = u[:10], u[10:]
    Y, b = xi

    Ad = Ad_vee_gal(np.linalg.inv(Y))
    Lambda_1 = exp_gal(Ad @ g * dt) @ exp_gal((w - b) * dt)
    Lambda_2 = b - Ad_vee_gal(Lambda_1) @ (b + tau * dt)

    return Lambda_1, Lambda_2

def ad_vee_group(delta_s, delta_i):
    m = len(delta_i) // 10
    blocks = [ad_vee_gal(delta_i[10*i:10*i+10]) for i in range(m)]
    ad_s = np.zeros((20,20))
    ad_s[:10, :10] = ad_vee_gal(delta_s[:10])
    ad_s[10:, 10:] = ad_vee_gal(delta_s[:10])
    ad_s[10:, :10] = ad_vee_gal(delta_s[10:])
    blocks.append(ad_s)
    return scipy.linalg.block_diag(*blocks)

#@numba.jit(nopython=True)
def lift_diff_group(xi, u, g):
    (Y, b), xi_i = xi
    us, ui = u
    w, tau = us[:10], us[10:]
    m = len(ui)

    Ad = Ad_vee_gal(np.linalg.inv(Y))

    Lambda_i = np.zeros((m,10))
    Lambda_s = np.concatenate([Ad @ g + w - b, ad_vee_gal(b) @ (Ad @ g + w)  - tau])
    return Lambda_s, Lambda_i

@numba.jit(nopython=True)
def lift_discrete_group(xi, u, g, dt):
    xi, xi_i = xi
    u, ui = u
    Lambda = lift_discrete_gal(xi, u, g, dt)

    m = len(ui)
    Lambda_i  = np.zeros((m, 5, 5))
    Lambda_i[:,0,0] = 1
    Lambda_i[:,1,1] = 1
    Lambda_i[:,2,2] = 1
    Lambda_i[:,3,3] = 1
    Lambda_i[:,4,4] = 1
    return Lambda, Lambda_i

@numba.jit(nopython=True)
def lift_gal_diff(xi, u):
    w, tau = u[:10], u[10:]
    _, b = xi

    Lambda_1 = hat_gal(w - b)
    Lambda_2 = hat_gal(ad_vee_gal(b) @ w - tau)

    return Lambda_1, Lambda_2



@numba.jit(nopython=True)
def state_action_gal(xi, X):
    Y, b = xi
    C, gamma = X

    Y_out = Y @ C
    b_out = Ad_vee_gal(np.linalg.inv(C)) @ (b - gamma)

    return Y_out, b_out

def state_action_inv_group(xi_circ, s):
    xi, xi_i = s
    xi_circ, xi_circ_i = xi_circ
    X_i = np.linalg.matmul(xi_circ_i, xi_i)
    X = state_action_inv_gal(xi_circ, xi)
    return (X, X_i)


@numba.jit(nopython=True)
def state_action_inv_gal(xi_circ, s):
    Y_circ, b_circ = xi_circ
    Y, b = s

    Y_out = np.linalg.inv(Y_circ) @ Y
    b_out = b_circ-Ad_vee_gal(Y_out) @ b

    return Y_out, b_out

def hat_group(x):
    u, ui = x[-20:], x[:-20]
    m = len(ui) // 10
    ui_hat = np.stack([hat_gal(ui[i*10:i*10+10]) for i in range(m)])
    u_hat = hat_gal_cross_gal(u[:10], u[10:])

@numba.jit(nopython=True)
def inv_gal_gal(X):
    C, gamma = X
    C_inv = np.linalg.inv(C)
    return C_inv, - Ad_vee_gal(C_inv) @ gamma

@numba.jit(nopython=True)
def Ad_vee_gal(C):
    A = C[:3,:3]
    a = C[:3, 3]
    b = C[:3, 4]
    c = C[ 3, 4]

    out = np.eye(10)
    out[0:3,0:3] = A
    out[3:6,3:6] = A
    out[6:9,6:9] = A
    out[3:6,0:3] = hat_so3(a) @ A
    out[6:9,0:3] = hat_so3(b- c*a) @ A
    out[6:9,3:6] = -c * A
    out[6:9,9] = a

    return out

@numba.jit(nopython=True)
def Ad_vee_gal_gal(X):
    C, gamma = X
    A = np.zeros((20,20))

    Ad = Ad_vee_gal(C)
    A[:10,:10] = Ad
    A[10:,10:] = Ad
    A[10:,:10] = ad_vee_gal(gamma) @ Ad

    return A

@numba.jit(nopython=True)
def Ad_gal(C, u):
    return hat_gal(Ad_vee_gal(C) @ vee_gal(u))

@numba.jit(nopython=True)
def ad_gal(gamma, beta):
    return gamma @ beta - beta @ gamma

@numba.jit(nopython=True)
def hat_gal(w):
    w, v, r, a = w[:3], w[3:6], w[6:9], w[9]

    out = np.zeros((5,5))
    out[:3,:3] = hat_so3(w)
    out[:3, 3] = v
    out[:3, 4] = r
    out[ 3, 4] = a

    return out

@numba.jit(nopython=True)
def hat_so3(x):
    out = np.zeros(shape=(x.shape[:-1] + (3,3)))
    out[...,2,1] = x[...,0]
    out[...,1,0] = x[...,2]
    out[...,0,2] = x[...,1]

    out[...,1,2] = -x[...,0]
    out[...,0,1] = -x[...,2]
    out[...,2,0] = -x[...,1]

    return out

@numba.jit(nopython=True)
def exp_gal(x):
    w = x[0:3]
    v = x[3:6]
    r = x[6:9]
    a = x[9]

    out = np.eye(5)
    out[:3,:3] = exp_so3(w)
    Gamma1, Gamma2 = Gamma1_Gamma2_gal(w)
    out[:3, 3] = Gamma1 @ v
    out[:3, 4] = Gamma1 @ r + a * Gamma2 @ v
    out[ 3, 4] = a

    return out

#@numba.jit(nopython=True)
def Gamma1_Gamma2_gal(w):
    w_hat = hat_so3(w)
    norm = np.linalg.norm(w, axis=-1)

    k1 = np.where(norm < 1e-4, 1/2, (1 - np.cos(norm)) / norm**2)
    k2 = np.where(norm < 1e-4, 1/6, (norm - np.sin(norm)) / norm**3)
    k3 = np.where(norm < 1e-4, 1/24, (norm**2 + 2 * np.cos(norm) - 2) / (2 * norm ** 4))

    Gamma1 = np.eye(3) + k1[...,None,None] * w_hat + k2[...,None,None] * w_hat @ w_hat
    Gamma2 = 0.5 * np.eye(3) + k2[...,None,None] * w_hat + k3[...,None,None] * w_hat @ w_hat

    return Gamma1, Gamma2

#@numba.jit(nopython=True)
def Gamma1(w):
    w_hat = hat_so3(w)
    norm = np.linalg.norm(w, axis=-1)

    k1 = np.where(norm < 1e-4, 1/2, (1 - np.cos(norm)) / norm**2)
    k2 = np.where(norm < 1e-4, 1/6, (norm - np.sin(norm)) / norm**3)

    Gamma1 = np.eye(3) + k1[...,None,None] * w_hat + k2[...,None,None] * w_hat @ w_hat

    return Gamma1

#@numba.jit(fastmath=True)
def exp_so3(w):
    w_hat = hat_so3(w)
    norm = np.sqrt((w**2).sum(-1))
    k1 = np.where(norm < 1e-4, 1, np.sin(norm) / norm)
    k2 = np.where(norm < 1e-4, 1/2, (1 - np.cos(norm)) / norm**2)
    return np.eye(3) + k1[...,None,None] * w_hat + k2[...,None,None] * w_hat @ w_hat


def random_gal():
    v, p = np.random.rand(2, 3)
    R = exp_so3(np.random.rand(3))
    t = np.random.rand()

    G = np.eye(5)
    G[:3, :3] = R
    G[:3, 3] = v
    G[:3, 4] = p
    G[3, 4] = t

    return G

def control_inputs_gal(w, a):
    w = np.concatenate([w, a, np.zeros(3), np.ones(1)])
    tau = np.zeros(10)
    u = np.concatenate([w, tau])
    return u

def control_inputs_group(w, a, num_clone_states=1):
    u = control_inputs_gal(w, a)
    u_i = np.zeros((num_clone_states, 10))
    return u, u_i

@numba.jit(nopython=True)
def L_gal_cross_gal(Delta):
    m = (len(Delta) - 20) // 10
    out = np.eye(len(Delta))
    for i in range(m+1):
        #out[10*i:10*i+10,10*i:10*i+10] = Ad_vee_gal(exp_gal(Delta[i*10:i*10+10])) @ J_L_gal(Delta[i*10:i*10+10])
        out[10*i:10*i+10,10*i:10*i+10] = J_L_gal(Delta[i*10:i*10+10])
    return out

def Gamma_1_adj_gal(xi):
    phi, nu, rho, tau = xi[0:3], xi[3:6], xi[6:9], xi[9:12]
    pn = np.linalg.norm(phi)
    phi_hat = hat_so3(phi)
    s = np.sin(pn)
    c = np.cos(pn)

    Gamma_1_xi = np.eye(10)
    J_phi = J_L_so3(phi)
    N_phi = (0.5 * np.eye(3) + (pn - s) / pn ** 3 * phi_hat + (pn ** 2 + 2 * c - 2) / (
                2 * pn ** 4) * phi_hat @ phi_hat)

    Gamma_1_phi_tau = tau * (
                .5 * np.eye(3) + (s - pn * c) / pn ** 3 * phi_hat + (pn ** 2 - 2 * pn * s - 2 * c + 2) / (
                    2 * pn ** 4) * phi_hat @ phi_hat)

    Gamma_1_xi[0:3, 0:3] = J_phi # A
    Gamma_1_xi[6:9, 3:6] = - Gamma_1_phi_tau # B
    Gamma_1_xi[6:9, 0:3] = Q1_gal(phi, rho) - Q3_gal(phi, nu) * tau

    Gamma_1_xi[6:9, 9  ] = N_phi @ nu

    Gamma_1_xi[3:6, 3:6] = J_phi
    Gamma_1_xi[3:6, 0:3] = Q1_gal(phi, nu)

    Gamma_1_xi[6:9, 6:9] = J_phi
    return Gamma_1_xi

@numba.jit(nopython=True)
def exp_gal_cross_gal(xi, nu):
    # following this paper https://arxiv.org/pdf/2503.02820
    Y = exp_gal(xi)
    b = J_L_gal(xi) @ nu
    return Y, b

def hat_gal_cross_gal(xi, nu):
    X = np.zeros((10,10))
    X[:5,:5] = hat_gal(xi)
    X[5:,5:] = hat_gal(xi)
    X[:5,5:] = hat_gal(nu)
    return X


def vee_gal(X):
    w = vee_so3(X[:3,:3])
    v = X[:3,3]
    r = X[:3,3]
    t = X[3,3:4]

    return np.concatenate([w, v, r, t])


def Ad_vee_se3(C):
    A = C[...,:3, :3]
    a = C[...,:3, 3]

    out = np.zeros(shape=C.shape[:-2] + (6, 6))
    out[...,0:3, 0:3] = A
    out[...,3:6, 3:6] = A
    out[...,3:6, 0:3] = hat_so3(a) @ A

    return out

@numba.jit(nopython=True)
def exp_so3_fast(w):
    w_hat = hat_so3(w)
    norm_2 = (w**2).sum(-1)
    norm = np.sqrt(norm_2)
    mask = norm < 1e-4
    k1 = np.where(mask, 1, np.sin(norm) / norm)
    k2 = np.where(mask, 1/2, (1 - np.cos(norm)) / norm_2)
    w_outer = w[...,:,None] * w[...,None,:]
    return np.eye(3) * np.cos(norm)[...,None,None] + k1[...,None,None] * w_hat + k2[...,None,None] * w_outer

@numba.jit(nopython=True)
def Gamma1_fast(w):
    w_hat = hat_so3(w)
    norm2 = (w**2).sum(-1)
    norm = np.sqrt(norm2)

    mask = norm < 1e-4
    k1 = np.where(mask, 1/2, (1 - np.cos(norm)) / norm2)
    k2 = np.where(mask, 1/6, (norm - np.sin(norm)) / (norm*norm2))
    k3 = np.where(mask, 1, np.sin(norm) / norm)

    w_outer = w[..., :, None] * w[..., None, :]

    return np.eye(3) * k3[...,None,None] + k1[...,None,None] * w_hat + k2[...,None,None] * w_outer

if __name__ == '__main__':
    import tqdm

    C = np.random.rand(10000, 3)
    res1 = Gamma1_fast(C)
    res2 = Gamma1(C)
    print(np.abs(res1-res2).mean())

    for _ in tqdm.tqdm(range(10000)):
        Gamma1_fast(C)



