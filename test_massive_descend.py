import taichi as ti
import numpy as np
import time
import trimesh
from alive_progress import alive_bar
from scipy.optimize import minimize


obj = trimesh.load('/Users/liamrobinson/Documents/mirage/mirage/resources/models/cube.obj')

ti.init(arch=ti.gpu, 
        default_fp=ti.float32,
        fast_math=True,
        advanced_optimization=True,
        num_compile_threads=32,
        opt_level=3,
        unrolling_limit=0)

def vecnorm(v: np.ndarray) -> np.ndarray:
    """Takes the :math:`L_2` norm of the rows of ``v``

    :param v: Input vector
    :type v: np.ndarray [nxm]
    :return: Norm of each row
    :rtype: np.ndarray [n,]
    """
    axis = v.ndim - 1
    return np.linalg.norm(v, axis=axis, keepdims=True)


def dot(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Euclidean dot product between corresponding rows of ``v1`` and ``v2``

    :param v1: First input vector
    :type v1: np.ndarray [nxm]
    :param v2: Second input vector
    :type v2: np.ndarray [nxm]
    :return: Dot products of corresponding rows
    :rtype: np.ndarray [n,]
    """
    axis = v1.ndim - 1
    return np.sum(v1 * v2, axis=axis, keepdims=True)

def hat(v: np.ndarray) -> np.ndarray:
    """Normalizes input np.ndarray n,3 such that each row is unit length. If a row is ``[0,0,0]``, ``hat()`` returns it unchanged

    :param v: Input vectors
    :type v: np.ndarray [n,3]
    :return: Unit vectors
    :rtype: np.ndarray [n,3]
    """
    vm = vecnorm(v)
    vm[vm == 0] = 1.0  # Such that hat([0,0,0]) = [0,0,0]
    return v / vm


@ti.func
def rand_b2(r: float) -> ti.math.vec3:
    return r * rand_s2() * ti.random() ** (1 / 3)

@ti.func
def rand_s2() -> ti.math.vec3:
    return ti.math.vec3(ti.randn(), ti.randn(), ti.randn()).normalized()


@ti.func
def rand_s3() -> ti.math.vec4:
    q = ti.math.vec4(ti.randn(), ti.randn(), ti.randn(), ti.randn()).normalized()
    if q[3] < 0:
        q = -q
    return q


@ti.func
def quat_to_dcm(q: ti.math.vec4) -> ti.math.mat3:
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    C = ti.math.mat3(
        1 - 2 * q1**2 - 2 * q2**2, 2 * (q0 * q1 + q2 * q3), 2 * (q0 * q2 - q1 * q3), 
        2 * (q0 * q1 - q2 * q3), 1 - 2 * q0**2 - 2 * q2**2, 2 * (q1 * q2 + q0 * q3), 
        2 * (q0 * q2 + q1 * q3), 2 * (q1 * q2 - q0 * q3), 1 - 2 * q0**2 - 2 * q1**2
    )
    return C.transpose()

@ti.func
def normalized_convex_light_curve(L: ti.math.vec3, O: ti.math.vec3) -> float:
    b = 0.0
    for i in range(fn.shape[0]):
        fr = brdf_phong(L, O, fn[i], fd[i], fs[i], fp[i])
        b += fr * rdot(fn[i], O) * rdot(fn[i], L)
    return b

@ti.func
def rdot(v1, v2) -> float:
    return ti.max(ti.math.dot(v1,v2), 0.0)

@ti.func
def brdf_phong(
    L: ti.math.vec3,
    O: ti.math.vec3,
    N: ti.math.vec3,
    cd: float,
    cs: float,
    n: float,
) -> float:
    NdL = ti.math.dot(N, L)
    if NdL <= 0.0:
        NdL = np.inf
    R = (2 * ti.math.dot(N, L) * N - L).normalized()
    fd = cd / np.pi
    fs = cs * (n + 2) / (2 * np.pi) * rdot(R, O) ** n / NdL
    return fd + fs

@ti.kernel
def integrate(itensor: ti.math.vec3, perturb_axis: int) -> int:
    h = teval[1] - teval[0]
    for i in current_states:
        for j in range(teval.shape[0]):
            if (j == 0) and (perturb_axis >= 0):
                # perturb along the indicated axis
                current_states[i][perturb_axis] += FINITE_DIFF_STATE_PERTURB
                if perturb_axis < 4:
                    current_states[i][:4] = current_states[i][:4].normalized()

            dcm = quat_to_dcm(current_states[i][:4])
            lc[i][j] = normalized_convex_light_curve(dcm @ L, dcm @ O)
            
            current_states[i][:4], current_states[i][4:] = rk4_step(current_states[i][:4], current_states[i][4:], h, itensor)
            current_states[i][:4] = current_states[i][:4].normalized()

    return 0 # to prevent taichi from lying about timing

@ti.pyfunc
def rk4_step(q0, w0, h, itensor):
    k1q, k1w = deriv(q0, w0, itensor)
    k2q, k2w = deriv(q0 + h * k1q / 2, w0 + h * k1w / 2, itensor)
    k3q, k3w = deriv(q0 + h * k2q / 2, w0 + h * k2w / 2, itensor)
    k4q, k4w = deriv(q0 + h * k3q, w0 + h * k3w, itensor)
    return (
        q0 + h * (k1q + 2 * k2q + 2 * k3q + k4q) / 6, 
        w0 + h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6)

@ti.pyfunc
def deriv(q, w, itensor):
    qmat = 0.5 * ti.math.mat4(
            q[3], -q[2], q[1], q[0],
            q[2], q[3], -q[0], q[1],
            -q[1], q[0], q[3], q[2],
            -q[0], -q[1], -q[2], q[3],
    )
    qdot = qmat @ ti.math.vec4(w.xyz, 0.0)

    wdot = ti.math.vec3(0.0, 0.0, 0.0)
    wdot.x = -1 / itensor[0] * (itensor[2] - itensor[1]) * w.y * w.z
    wdot.y = -1 / itensor[1] * (itensor[0] - itensor[2]) * w.z * w.x
    wdot.z = -1 / itensor[2] * (itensor[1] - itensor[0]) * w.x * w.y

    return qdot, wdot


v1 = obj.vertices[obj.faces[:,0]]
v2 = obj.vertices[obj.faces[:,1]]
v3 = obj.vertices[obj.faces[:,2]]
fnn = np.cross(v2 - v1, v3 - v1)
fan = np.linalg.norm(fnn, axis=1, keepdims=True) / 2
fnn = fnn / fan / 2

fa = ti.field(dtype=ti.f32, shape=fan.shape)
fa.from_numpy(fan.astype(np.float32))

fn = ti.Vector.field(n=3, dtype=ti.f32, shape=fnn.shape[0])
fn.from_numpy(fnn.astype(np.float32))

fd = ti.field(dtype=ti.f32, shape=fnn.shape[0])
fd.fill(0.5)
fs = ti.field(dtype=ti.f32, shape=fnn.shape[0])
fs.fill(0.5)
fp = ti.field(dtype=ti.f32, shape=fnn.shape[0])
fp.fill(3)

L = ti.Vector([1.0, 0.0, 0.0]).normalized()
O = ti.Vector([1.0, 1.0, 0.0]).normalized()

q0 = ti.Vector([0.0, 0.0, 0.0, 1.0])
w0 = ti.Vector([1.0, 1.0, 1.0])
itensor = ti.Vector([1.0, 2.0, 3.0])

FINITE_DIFF_STATE_PERTURB = 1e-3
MAX_ANG_VEL_MAG = 0.3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-6
n_particles = int(1e4)
n_times = 50
tmax = 10.0
states = ['qx', 'qy', 'qz', 'qw', 'wx', 'wy', 'wx']
n_states = len(states)
tspace = np.linspace(0, tmax, n_times, dtype=np.float32)
h = tspace[1] - tspace[0]

teval = ti.field(dtype=ti.f32, shape=n_times)
teval.from_numpy(tspace)

initial_states = ti.Vector.field(n=n_states, dtype=ti.f32, shape=n_particles)
states_v = ti.Vector.field(n=n_states, dtype=ti.f32, shape=n_particles)
current_states = ti.Vector.field(n=n_states, dtype=ti.f32, shape=n_particles)
lc = ti.Vector.field(n=teval.shape[0], dtype=ti.f32, shape=n_particles)
lc_true = ti.field(shape=teval.shape[0], dtype=ti.f32)
lct = np.sin(tspace / 10) * 0 + 0.3
lc_true.from_numpy(hat(lct))

loss = ti.field(dtype=ti.f32, shape=(n_particles,n_states+1)) 


# %%
# Initializing the quaternion and angular velocity guesses

@ti.kernel
def initialize_states() -> int:
    for i in current_states:
        initial_states[i][:4] = rand_s3()
        initial_states[i][4:] = rand_b2(MAX_ANG_VEL_MAG)
        current_states[i] = initial_states[i]
    return 0

@ti.kernel
def reset_states() -> int:
    for i in current_states:
        current_states[i] = initial_states[i]
    return 0


@ti.kernel
def compute_loss(perturbed_state_ind: int) -> int:
    # Computing local loss 
    for i in range(loss.shape[0]):
        lci_norm = lc[i].norm()
        loss[i,perturbed_state_ind] = 0.0
        for j in range(lc_true.shape[0]):
            loss[i,perturbed_state_ind] += lc[i][j]/lci_norm * lc_true[j]
        loss[i,perturbed_state_ind] = ti.acos(loss[i,perturbed_state_ind])
    return 0

@ti.kernel
def update_states() -> int:
    for i in range(loss.shape[0]):
        initial_states[i] += states_v[i]
        initial_states[i][:4] = initial_states[i][:4].normalized() # make sure the quaternion keeps its norm
    return 0


def update_istate(current_istates, delta_x, current_step_size, fprime_current):
    delta_state = -current_step_size * fprime_current
    new_istate = current_istates + delta_state
    new_istate[:,:4] /= np.linalg.norm(new_istate[:,:4], axis=1, keepdims=True)

    breaking_ang_limits = np.linalg.norm(new_istate[:,4:7], axis=1) > MAX_ANG_VEL_MAG
    new_istate[breaking_ang_limits,4:7] *= MAX_ANG_VEL_MAG / np.linalg.norm(new_istate[breaking_ang_limits,4:7], axis=1, keepdims=True) 
    return new_istate


def compute_gradient_from_losses(current_istates, losses):
    fprime = (losses[:,1:]-losses[:,[0]])/FINITE_DIFF_STATE_PERTURB
    fprime[:,:4] -= dot(fprime[:,:4], current_istates[:,:4]) * current_istates[:,:4]
    return fprime


prev_state = initial_states.to_numpy()
prev_gradient = np.zeros((n_particles, n_states))
current_step_size = np.full((n_particles,1), 1e-2)

n_iter = 150
losses = np.zeros((n_iter, n_particles))
states = np.zeros((n_iter, n_particles, n_states))
with alive_bar(n_iter) as bar:
    for i in range(n_iter):
        t0 = time.time()
        if i == 0:
            initialize_states()

        t1 = time.time()
        for pstate in range(-1,n_states):
            reset_states()
            integrate(itensor, pstate)
            compute_loss(pstate+1)
        # print(f'int time: {time.time()-t1:.1e}')
        
        t1 = time.time()
        # print(f'loss time: {time.time()-t1:.1e}')
        t1 = time.time()
        l = loss.to_numpy()
        losses[i] = l[:,0]

        current_istate = initial_states.to_numpy()
        states[i,:,:] = current_istate
        current_fprime = compute_gradient_from_losses(current_istate, l)

        if i == 0:
            new_istate = update_istate(current_istate, 0*current_istate, current_step_size, current_fprime)
            delta_x = new_istate - current_istate
            old_fprime = current_fprime
        else:
            delta_fprime = current_fprime - old_fprime
            print(delta_fprime.shape, delta_x.shape)

            xx = np.sum(np.abs(delta_x * delta_x), axis=1, keepdims=True)
            xg = np.sum(np.abs(delta_x * delta_fprime), axis=1, keepdims=True)
            gg = np.sum(np.abs(delta_fprime * delta_fprime), axis=1, keepdims=True)
            
            # long
            current_step_size = xx / xg
            # short
            # current_step_size = xg / gg
            new_istate = update_istate(current_istate, delta_x, current_step_size, current_fprime)

        new_istate[np.isnan(new_istate[:,0]),:] = prev_state[np.isnan(new_istate[:,0]),:]
        initial_states.from_numpy(new_istate.astype(np.float32))
        print(f'med grad mag: {np.median(np.linalg.norm(current_fprime, axis=1)):.1e}')
        # if np.isnan(l).sum():
        #     print(prev_state[np.isnan(l[:,0]),:])
        #     print(states[i-1, np.isnan(l[:,0]),:])
        #     print(prev_gradient[np.isnan(l[:,0]),:])
        #     endd
        print(f'mean loss: {l.mean():.2f} min loss: {l.min():.1e}')
        # update_states()
        # print(f'update time: {time.time()-t1:.1e}')

        bar()
    
import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.yscale('log')
# plt.show()

fpmag = vecnorm(current_fprime)

states = states[:,fpmag.flatten() < 100,:]

plt.scatter(states[0,:,0], states[0,:,1], s=1, alpha=0.3, c='k')
plt.scatter(states[-1,:,0], states[-1,:,1], s=1, alpha=0.3, c='r')
plt.show()

plt.hist(fpmag, bins=np.geomspace(1e-5,1e2,100))
plt.xscale('log')
plt.show()

plt.hist(losses[-1,:], bins=np.geomspace(1e-5,1e2,100))
plt.xscale('log')
plt.show()

# for i in range(states.shape[-1]):
#     plt.plot(np.linalg.norm(states[:,:,4:], axis=-1))
# plt.show()
