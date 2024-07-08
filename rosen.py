import numpy as np
from scipy.optimize import minimize
from typing import Callable, Literal
import matplotlib.pyplot as plt
import time

def finite_difference_gradient(f: Callable, 
                               x0: np.ndarray, 
                               finite_difference_stepsize: float,
                               difference_type: Literal['forward', 'central']) -> np.ndarray:
    g = np.zeros_like(x0)
    fx0 = f(x0)
    for pind in range(x0.size):
        p = np.zeros_like(x0)
        p[pind] = finite_difference_stepsize
        if difference_type.lower() == 'forward':
            g[pind] = (f(x0 + p/2.0) - fx0) / finite_difference_stepsize
        if difference_type.lower() == 'central':
            g[pind] = (f(x0 + p/2.0) - f(x0 - p)) / finite_difference_stepsize
    return g

def adam(f: Callable,
         x0: np.ndarray,
         alpha: float = 0.5,
         beta1: float = 0.9,
         beta2: float = 0.999,
         maxiter: int = 1000,
         finite_difference_stepsize: float = 1e-6,
         epsilon: float = 1e-8,
         gtol: float = 1e-5,
         difference_type: str = 'forward'):
    x = x0.copy()
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    t = 0
    fx_hist = [f(x)]
    x_hist = [x]
    converged = False

    while not converged:
        t += 1
        g = finite_difference_gradient(f, x, finite_difference_stepsize, difference_type)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        mh = m / (1 - beta1 ** t)
        vh = v / (1 - beta2 ** t)
        x = x - alpha * mh / (np.sqrt(vh) + epsilon)

        x_hist.append(x)
        fx_hist.append(f(x))

        grad_mag = np.linalg.norm(g)
        if t > maxiter:
            status = 'maximum iterations reached'
            converged = True
        if grad_mag < gtol:
            status = 'gradient magnitude tolerance reached'
            converged = True
    x_hist = np.array(x_hist)
    fx_hist = np.array(fx_hist)
    return {'x_opt': x, 'x_hist': x_hist, 'fx_hist': fx_hist, 'grad': g, 'status': status}

def gradient_descent(f: Callable,
                 x0: np.ndarray,
                 alpha: float = 0.5,
                 maxiter: int = 1000,
                 finite_difference_stepsize: float = 1e-6,
                 gtol: float = 1e-2,
                difference_type: str = 'forward'):
    x = x0.copy()
    t = 0
    fx_hist = [f(x)]
    x_hist = [x]
    converged = False

    while not converged:
        t += 1
        g = finite_difference_gradient(f, x, finite_difference_stepsize, difference_type)
        x = x - alpha * g
        x_hist.append(x)
        fx_hist.append(f(x))
        
        grad_mag = np.linalg.norm(g)
        if t > maxiter:
            status = 'maximum iterations reached'
            converged = True
        if grad_mag < gtol:
            status = 'gradient magnitude tolerance reached'
            converged = True
    x_hist = np.vstack(x_hist)
    fx_hist = np.vstack(fx_hist)
    return {'x_opt': x, 'x_hist': x_hist, 'fx_hist': fx_hist, 'grad': g, 'status': status}

def barzilai_borwein(f: Callable,
                    x0: np.ndarray,
                    alpha0: float = 0.5,
                    maxiter: int = 1000,
                    finite_difference_stepsize: float = 1e-6,
                    gtol: float = 1e-2,
                    step_type: Literal['short', 'long'] = 'short',
                    difference_type: str = 'forward') -> dict:
    x = x0.copy()
    t = 0
    fx_hist = [f(x)]
    x_hist = [x]
    converged = False

    while not converged:
        t += 1
        g = finite_difference_gradient(f, x, finite_difference_stepsize, difference_type)

        if t == 1:
            delta_x = -alpha0 * g
            g_prev = g
        else:
            delta_g = g - g_prev
            if step_type.lower() == 'short':
                alpha = delta_x * delta_g / (delta_g * delta_g)
            else:
                alpha = delta_x * delta_x / (delta_x * delta_g)
            delta_x = -alpha * g
            g_prev = g

        x = x + delta_x
        
        x_hist.append(x)
        fx_hist.append(f(x))
        
        grad_mag = np.linalg.norm(g)
        if t > maxiter:
            status = 'maximum iterations reached'
            converged = True
        if grad_mag < gtol:
            status = 'gradient magnitude tolerance reached'
            converged = True
    x_hist = np.vstack(x_hist)
    fx_hist = np.vstack(fx_hist)
    return {'x_opt': x, 'x_hist': x_hist, 'fx_hist': fx_hist, 'grad': g, 'status': status}

    
    
def nelder_mead(f: Callable,
                 x0: np.ndarray,
                 alpha: float = 0.5,
                 maxiter: int = 1000,
                 finite_difference_stepsize: float = 1e-6,
                 gtol: float = 1e-2,
                 difference_type: str = 'forward') -> dict:
    pass


def rosen(X):
    time.sleep(0.001)
    return (1.0 - X[0])**2 + 100.0 * (X[1] - X[0]**2)**2

x0 = np.array([-1.0,1.0])

# sol_adam = adam(rosen, x0)
# sol_grad_descent = gradient_descent(rosen, x0, alpha=0.002)
# sol_bb = barzilai_borwein(rosen, x0, alpha0=0.002, step_type='long')

methods = {'BFGS': None, 'Nelder-Mead': None, 'Powell': None, 'cg': None}
for method in methods.keys():
    xhist = []
    def callback(xi):
        global xhist
        xhist.append(xi)
    res = minimize(rosen, x0, method=method, callback=callback)
    methods[method] = xhist
    t1 = time.time()
    for _ in range(1):
        res = minimize(rosen, x0, method=method)
        assert res.success
    print(f'{method:15}: {time.time()-t1:10.2e}, nfev: {res["nfev"]:4}, nit: {res["nit"]:4}')


print(f'Adam converged in {sol_adam["x_hist"].shape[0]} iterations!')

# for sol,method in [(sol_adam, 'adam'), (sol_grad_descent, 'grad'), (sol_bb, 'bb')]:
#     plt.plot(sol['x_hist'][:,0], sol['x_hist'][:,1], linewidth=1, label=method)

for method,xhist in methods.items():
    xhist = np.array(xhist)
    plt.plot(xhist[:,0], xhist[:,1], linewidth=1, label=method)
plt.legend()
plt.show()