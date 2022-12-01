import numpy as np
import matplotlib.pyplot as plt
from functions import rosenbrock2d, rosenbrock2d_grad
from methods import steepest_descend

x1 = np.linspace(- 0.5, + 0.5, 30)
x2 = np.linspace(- 0.5, + 0.5, 30)
X1, X2 = np.meshgrid(x1, x2)
Z = rosenbrock2d([X1, X2])
plt.figure()
plt.title('Rosenbrock')
plt.contourf(X1, X2, Z, 30, cmap='jet')
plt.colorbar()
plt.xlabel('$x_1$');
plt.ylabel('$x_2$')
plt.show()

x0 = np.array([-1.2,1])
history = steepest_descend(rosenbrock2d, rosenbrock2d_grad, x0, eps=1e-3, K=10000)
#plt.plot(x_store[:,0],x_store[:,1],c='w')
print(f"El mínimo de la función es: {history[-1]}")