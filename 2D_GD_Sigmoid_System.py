# code written together with GPT4
import numpy as np
from numpy import linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Problem parameters
p1 = 0.95
p2 = 1-p1
e1 = np.array([1, 0])
e2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
eta = 20
T = 1000

# Define the reverse-sigmoid function
def phi(x):
    return 1 / (1 + np.exp(x))

# Update equations
def update(p1, p2, e1, e2):
    delta_x = phi(x) * p1 * LA.norm(e1)**2 - phi(y) * p2 * np.dot(e1, e2)
    delta_y = phi(y) * p2 * LA.norm(e2)**2 - phi(x) * p1 * np.dot(e1, e2)
    return delta_x, delta_y

# Initial conditions
x0, y0 = 0, 0  # Starting values of x and y at t=0

x_array = np.zeros(T)
y_array = np.zeros(T)
x, y = x0, y0
for itr in range(T):
    x_array[itr] = x
    y_array[itr] = y
    next_val = update(p1, p2, e1, e2)
    x_next = x + eta * next_val[0]
    y_next = y + eta * next_val[1]
    x = x_next
    y = y_next

# Plotting
plt.figure()
plt.plot(x_array, label='$m_1(t)$')
plt.plot(y_array, label='$m_2(t)$')
plt.title('Solution of GD $p_1=${} $p_2=${}, $\eta=${}'.format(p1, p2, eta))
plt.xlabel('Time (t)')
plt.ylabel('Evolution of Margins')
plt.legend()
plt.grid(True)

plt.show()
