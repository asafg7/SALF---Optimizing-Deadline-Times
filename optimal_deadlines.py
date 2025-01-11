import numpy as np
from scipy import special
from matplotlib import pyplot as plt
from scipy.optimize import SR1
import scipy.optimize as opt


def opt_function_deadlines(t, etta, g, u, l, rho_c, b, d1):

    """ optimization function for layered federated learning
    Inputs:
    t - deadline times (1xT)
    etta - step sizes (1xT)
    g - gradient bound from AS3 (1x1)
    u - number of users (1x1)
    l - number of layers
    rho_c - strong convexity constant from AS1 (1x1)
    """

    num_iter = np.size(t)
    ex_mult = np.zeros(num_iter)
    p_val = np.zeros([l, num_iter])

    for i in range(1, l+1):
        p_val[i-1, :] = (1+special.gammaincc(i, t)**u)/(1-special.gammaincc(i, t)**u)
    p_sum = np.sum(p_val, 0)
    for i in range(num_iter):
        ex_mult[i] = (etta[i] ** 2)*np.prod(1-rho_c*etta[i+1:])
    c_t = (g**2)*(4*u)/(u-1)*p_sum
    A = np.prod(1-rho_c*etta)*d1
    B = np.sum(ex_mult*(b+c_t))
    f = np.prod(1-rho_c*etta)*d1 + np.sum(ex_mult*(b+c_t))
    return f


def get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta):

    sigma_u = 1 * np.random.rand(u)
    b = 1 / (u ** 2) * np.sum(sigma_u ** 2) + 6 * rho_s * gamma

    t0 = np.ones(num_iter) * (t_max / num_iter)
    d1 = np.sum(t0 ** 2) / 1000
    bounds = opt.Bounds(lb=t_min, ub=np.inf)
    lin_const = opt.LinearConstraint(np.ones([1, num_iter]), lb=0, ub=t_max)

    trivial_val = opt_function_deadlines(t0, etta, g, u, l, rho_c, b, d1)
    res = opt.minimize(opt_function_deadlines, t0, method='trust-constr', jac="2-point", hess=SR1(),
                       constraints=lin_const, options={'verbose': 1}, bounds=bounds, args=(etta, g, u, l, rho_c, b, d1))
    x = res.x
    t_opt = x
    optimal_val = opt_function_deadlines(x, etta, g, u, l, rho_c, b, d1)

    print('Trivial Value - ', trivial_val, ', Optimal Value', optimal_val)

    plt.plot(range(num_iter), t0, range(num_iter), t_opt)
    plt.legend(['Trivial Allocation', 'Optimal Allocation'])
    plt.title('Iteration Time Allocation')
    plt.show()

    return t_opt

def main():
    u = 15
    l = 6
    num_iter = 100
    t_max = 300
    g = 1
    rho_s = 1e-2
    rho_c = 1e-2
    gamma = 1
    t_min = 0.5
    iters = np.arange(1, num_iter + 1)
    kappa = rho_s / rho_c
    l_gamma = np.max((8 * kappa, 1)) - 1

    # decaying 1 / t + l_gamma
    etta = 1 / (rho_c * (iters + l_gamma))
    iteration_times_decay_lin1 = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying 1 / t + l_gamma
    etta = 1 / (rho_c * (iters + l_gamma))
    iteration_times_decay_lin1 = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying 1 / t + l_gamma
    etta = 1 / (rho_c * (iters + l_gamma))
    iteration_times_decay_lin1 = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying 1 / t + l_gamma
    etta = 1 / (rho_c * (iters + l_gamma))
    iteration_times_decay_lin1 = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying 1 / 2(t* + l_gamma)
    etta = 2 / (rho_c * (iters + l_gamma))
    iteration_times_decay_lin2 = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying 1 / 2(t* + l_gamma)
    etta = 2 / (rho_c * (0.1*iters + l_gamma))
    iteration_times_decay_lin3 = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying 1 / 2(t^0.5 * + l_gamma)
    etta = 1 / (2*rho_c * (np.sqrt(iters) + l_gamma))
    iteration_times_decay_sqrt = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # decaying every 20 epochs
    etta = np.concatenate((np.ones(20), 0.7*np.ones(20), (0.7**2)*np.ones(20),
                          (0.7**3)*np.ones(20), (0.7**4)*np.ones(20))).reshape(-1, 1)
    iteration_times_decay_con = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    # uniform
    etta = 0.1*np.ones(np.size(iters))
    iteration_times_uniform = get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta)

    plt.plot(iters, iteration_times_decay_lin1, iters, iteration_times_decay_lin2, iters, iteration_times_decay_lin3,
             iters, iteration_times_decay_sqrt, iters, iteration_times_decay_con, iters, iteration_times_uniform)
    plt.legend([r"$\frac{1}{\rho_c (t+\gamma)}$", r"$\frac{2}{\rho_c (t+\gamma)}$", r"$\frac{2}{\rho_c (0.1*t+\gamma)}$",
                r"$\frac{2}{\rho_c (\sqrt{t}+\gamma)}$", "decaying every 20 epochs", r"$0.1$"])
    plt.title('Iteration Time Allocation')
    plt.show()
if __name__ == '__main__':
    main()