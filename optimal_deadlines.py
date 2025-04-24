import numpy as np
from scipy import special
from matplotlib import pyplot as plt
from scipy.optimize import SR1
import scipy.optimize as opt
from configurations import args_parser


def opt_function_deadlines(t, etta, g, u, l, rho_c, b, d1):

    """ optimization function for layered federated learning
    Inputs:
    t - deadline times (1xT)
    etta - step sizes (1xT)
    g - gradient bound from AS3 (1x1)
    u - number of users (1x1)
    l - number of layers (1x1)
    rho_c - strong convexity constant from AS1 (1x1)
    b - gradient variable (1x1)
    d1 - initial distance bound (1x1)
    """

    num_iter = np.size(t)
    ex_mult = np.zeros(num_iter)
    p_val = np.zeros([l, num_iter])

    for i in range(1, l+1):
        p_val[i-1, :] = (1+special.gammaincc(i, t)**u)/(1-2*special.gammaincc(i, t)**u)
    p_sum = np.sum(p_val, 0)
    for i in range(num_iter):
        ex_mult[i] = (etta[i] ** 2)*np.prod(1-rho_c*etta[i+1:])
    c_t = (g**2)*(4*u)/(u-1)*p_sum
    A = np.prod(1-rho_c*etta)*d1
    B = np.sum(ex_mult*(b+c_t))
    f = np.prod(1-rho_c*etta)*d1 + np.sum(ex_mult*(b+c_t))
    return f

def opt_function_deadlines_batchsize(x, etta, g, u, l, rho_c, sigma_u,
                                     gamma, rho_s, d1, alpha, N_samples, orig_batch_size, R_u):

    """ optimization function for layered federated learning
    Inputs:
    x - input (deadlines + batch size) (1xT+1)
    etta - step sizes (1xT)
    g - gradient bound from AS3 (1x1)
    u - number of users (1x1)
    l - number of layers (1x1)
    rho_c - strong convexity constant from AS1 (1x1)
    sigma_u - variance of the gradients (1x1)
    gamma - heterogeneity gap (1x1)
    rho_c - smoothness constant from AS1 (1x1)
    d1 - initial distance bound (1x1)
    alpha - weight of the sgd variance (1x1)
    N_samples - training samples count (1x1)
    orig_batch_size - original batch size (1x1)
    R_u computational capabilities (1xU)
    """

    num_iter = np.size(x) - 1
    t = x[:-1]
    m = x[-1]
    ex_mult = np.zeros(num_iter)
    var_val = np.zeros([l, num_iter])
    for i in range(1, l+1):
        single_prob = special.gammaincc(l + 1 - i, t/m)
        p_val = single_prob**u
        var_val[i-1, :] = (1+p_val)/(1-2*p_val)
    p_sum = np.sum(var_val, 0)
    c_t = (g**2)*(4*u)/(u-1)*p_sum
    b_t = (1 / (u**2 * m)) * np.sum(np.divide(sigma_u ** 2, R_u)) + 6 * rho_s * gamma
    for i in range(num_iter):
        ex_mult[i] = (etta[i] ** 2)*np.prod(1-rho_c*etta[i+1:])
    A = np.prod(1-rho_c*etta)*d1
    B = np.sum(ex_mult*(b_t+c_t))
    f = 1e3 * (np.prod(1-rho_c*etta)*d1 + np.sum(ex_mult*(alpha*b_t+c_t)))
    return f


def get_optimal_deadlines(u, l, num_iter, t_max, g, rho_s, rho_c, gamma, t_min, etta):

    sigma_u = 1 * np.random.rand(u)
    b = 1 / (u ** 2) * np.sum(sigma_u ** 2) + 6 * rho_s * gamma

    t0 = np.ones(num_iter) * (t_max / num_iter)
    d1 = np.sum(t0 ** 2) / 1e6
    bounds = opt.Bounds(lb=t_min, ub=np.inf)
    lin_const = opt.LinearConstraint(np.ones([1, num_iter]), lb=0, ub=t_max)

    trivial_val = opt_function_deadlines(t0, etta, g, u, l, rho_c, b, d1)
    res = opt.minimize(opt_function_deadlines, t0, method='trust-constr', jac="2-point", hess=SR1(),
                       constraints=lin_const, options={'verbose': 1}, bounds=bounds, args=(etta, g, u, l, rho_c, b, d1))
    x = res.x
    t_opt = x
    optimal_val = opt_function_deadlines(x, etta, g, u, l, rho_c, b, d1)

    print('Trivial Value - ', trivial_val, ', Optimal Value', optimal_val)
    
    #plt.ion()
    plt.plot(range(num_iter), t0, range(num_iter), t_opt)
    plt.legend(['Trivial Allocation', 'Optimal Allocation'])
    plt.title('Iteration Time Allocation')
    plt.show()

    return t_opt

def get_optimal_deadlines_batchsize(u, l, num_iter, t_max, g, rho_s, rho_c,
                                    gamma, t_min, mean_std, etta, alpha, N_samples, orig_batch_size, R_u):

    t0 = np.ones(num_iter) * (t_max / num_iter)
    d1 = np.sum(t0 ** 2) / 1e6
    lb_arr = np.append(t_min*np.ones([1, num_iter]), 0.5)
    ub_arr = np.append(np.inf*np.ones([1, num_iter]), 1)
    bounds = opt.Bounds(lb=lb_arr, ub=ub_arr)
    lin_const = opt.LinearConstraint(np.append(np.ones([1, num_iter]), 0), lb=0, ub=t_max)
    m0 = 1
    sigma_u = mean_std * np.random.rand(1, u)

    x0 = np.append(t0, m0)

    trivial_val = opt_function_deadlines_batchsize(x0, etta, g, u, l, rho_c, sigma_u, gamma,
                                                   rho_s, d1, alpha, N_samples, orig_batch_size, R_u)
    res = opt.minimize(opt_function_deadlines_batchsize, x0, method='trust-constr', jac="2-point", hess=SR1(),
                       constraints=lin_const, options={'verbose': 1}, bounds=bounds,
                       args=(etta, g, u, l, rho_c, sigma_u, gamma, rho_s, d1, alpha, N_samples, orig_batch_size, R_u))
    x = res.x
    optimal_val = opt_function_deadlines_batchsize(x, etta, g, u, l, rho_c, sigma_u,
                                                   gamma, rho_s, d1, alpha, N_samples, orig_batch_size, R_u)

    t_opt = x[:-1]
    m_opt = x[-1]

    print('Trivial Value - ', trivial_val, ', Optimal Value', optimal_val)
    print('m value - ', m_opt)
   
    #plt.ion()
    plt.plot(range(num_iter), t0, range(num_iter), t_opt)
    plt.legend(['Trivial Allocation', 'Optimal Allocation'])
    plt.title('Iteration Time Allocation')
    plt.show()

    return t_opt, m_opt


def main():

    fig, ax = plt.subplots()

    args = args_parser()
    iters = np.arange(1, args.global_epochs + 1)
    kappa = args.rho_s / args.rho_c
    l_gamma = np.max((8 * kappa, 1)) - 1
    num_layers = 8
    args.t_max = 200*8
    sigma_u = 100*np.ones((1, args.num_users))
    #alpha_arr = np.linspace(0.2, 2, 10)
    alpha_arr = [1]
    for alpha in alpha_arr:
        etta = 1 / (args.rho_c * (np.power(iters, alpha) + l_gamma))
        iteration_times = get_optimal_deadlines_batchsize(args.num_users, num_layers, args.global_epochs, args.t_max,
                                                          args.g, args.rho_s, args.rho_c, args.gamma,
                                                          args.t_min, sigma_u, etta, alpha, 100, 100)
        ax.plot(iteration_times, label="alpha: " + "{:.1f}".format(alpha))

    ax.set(xlabel='iteration', ylabel='time allocation',
           title='Deadline Time Allocation by alpha')
    ax.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
