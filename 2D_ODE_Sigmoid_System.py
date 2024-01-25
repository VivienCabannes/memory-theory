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

# Define the reverse-sigmoid function
def phi(x):
    return 1 / (1 + np.exp(x))

# System of ODEs
def system(t, z, p1, p2, e1, e2):
    x, y = z
    dxdt = phi(x) * p1 * LA.norm(e1)**2 - phi(y) * p2 * np.dot(e1, e2)
    dydt = phi(y) * p2 * LA.norm(e2)**2 - phi(x) * p1 * np.dot(e1, e2)
    return [dxdt, dydt]

# Initial conditions
x0, y0 = 0, 0  # Starting values of x and y at t=0

# Time span for the solution
t_span = [0, 1000]
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where the solution is computed

# Solve the system of ODEs
sol1 = solve_ivp(system, t_span, [x0, y0], args=(p1, p2, e1, e2), t_eval=t_eval)

# Plotting
plt.figure()
plt.plot(sol1.t, sol1.y[0], label='$m_1(t)$')
plt.plot(sol1.t, sol1.y[1], label='$m_2(t)$')
plt.title('Solution of the 2D ODE $p_1=${} $p_2=${}'.format(p1, p2))
plt.xlabel('Time (t)')
plt.ylabel('Evolution of Margins')
plt.legend()
plt.grid(True)

plt.show()
