import matplotlib.pyplot as plt
import math
import numpy as np

def forward_euler(N, S, I, beta, gamma, t):
    """
    forward_euler computes one step of forward Euler 
    :param N: Total number of people in population
    :param S: Total number of susceptibles in population
    :param I: Total number of infected people in population
    :param R: Total number of recovered people in population
    :param beta: Transmission rate constant
    :param gamma: Recovery rate
    :param t: time step value
    :returns: new values for S, I, R (in that order)
    """
    # We are assuming that every interaction occurs at the exact same time
    # i.e. people who just got infected this time step are not included in people who can recover
    S_dot = -((beta * S * I)/float(N)) + gamma * I
    I_dot = ((beta * S * I)/float(N)) - gamma * I
    return S + (S_dot * t), I + (I_dot * t)

def analytical(t, beta, gamma):
    # Analytical calculation
    x_0 = 0.01
    r = beta - gamma
    R_0 = beta/gamma
    K = 1 - (1/R_0)

    # yikes
    denom = 1 + (((K - x_0)/x_0) * math.e**(-r * t))
    return K/denom

# params to change
t_arr = [2, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
max_error_arr = []
t_final = 25
beta = 3
gamma = 2

for t in t_arr:
    N = 1000
    S = 990
    I = 10

    S_arr = [S]
    I_arr = [I]
    y_arr = [analytical(0, beta, gamma)]
    iterations = int(t_final/t)
    for i in range(iterations - 1):
        S, I = forward_euler(N, S, I, beta, gamma, t)
        S_arr.append(S)
        I_arr.append(I)
        y_arr.append(analytical(t*i, beta, gamma))


    # max error for each solution
    Euler = np.divide(I_arr, N)
    diff = np.abs(Euler - y_arr)
    max_error_arr.append(max(diff))
    print("max absolute error for t=" + str(t) + ": " + str(max(diff)))

plt.loglog(t_arr, max_error_arr, marker='*', color='r')
plt.title('Max Error for dt (Kevin)')
plt.xlabel('dt', fontsize=12)
plt.ylabel('E(dt)', fontsize=12)

plt.savefig('MaxError.png')