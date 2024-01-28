# code written together with GPT4
import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

my_path = os.path.dirname(__file__)

# Problem parameters
p1 = 0.5
p2 = 1-p1
e1 = np.array([1, 0])
e2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
eta_lst = [0.01, 0.1, 1, 10, 100]
colors = ['black', 'gray', 'lightcoral', 'red', 'darkred']
T = 100

# Define the reverse-sigmoid function
def phi(x):
    return 1 / (1 + np.exp(x))

# Update equations
def update(p1, p2, e1, e2):
    delta_x = phi(x) * p1 * LA.norm(e1)**2 - phi(y) * p2 * np.dot(e1, e2)
    delta_y = phi(y) * p2 * LA.norm(e2)**2 - phi(x) * p1 * np.dot(e1, e2)
    return delta_x, delta_y

def loss(p1, p2, m1, m2):
    return p1 * np.log(1+np.exp(-m1)) + p2 * np.log(1+np.exp(-m2))


for eta_itr in range(len(eta_lst)):
    eta = eta_lst[eta_itr]

    # Initial conditions
    x0, y0 = 0, 0  # Starting values of x and y at t=0

    x_array = np.zeros(T)
    y_array = np.zeros(T)
    loss_array = np.zeros(T)
    x, y = x0, y0
    for itr in range(T):
        x_array[itr] = x
        y_array[itr] = y
        loss_array[itr] = loss(p1, p2, x, y)
        next_val = update(p1, p2, e1, e2)
        x_next = x + eta * next_val[0]
        y_next = y + eta * next_val[1]
        x = x_next
        y = y_next

    # Plotting
    if(eta_itr==0):
        plt.figure()
        plt.title('$p_1=${:.02f} $p_2=${:.02f}'.format(p1, p2))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        #plt.ylabel('Evolution of Margins')
    plt.plot(loss_array, color=colors[eta_itr], label='lr={:.2f}'.format(eta))
    # plt.plot(x_array, color=colors[eta_itr], linestyle='dashed')
    # plt.plot(y_array, label='lr={:.2f}'.format(eta), color=colors[eta_itr], linestyle='solid')

plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(my_path + '/figs/loss_with_LRs_same_prob.png', dpi=500)
