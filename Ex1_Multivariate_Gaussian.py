import tri_d_gausian as tdg
import numpy as np
import matplotlib.pyplot as plt

a = tdg.x_y_z

# q_23
tdg.plot_3d(tdg.x_y_z)

# q_24
s = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
tdg.plot_3d(s @ a)
print(np.cov(s @ a))

# q_25
ort = tdg.get_orthogonal_matrix(3)
tdg.plot_3d(ort @ s @ a)
print(np.cov(ort @ s @ a))

# q_26
projection_mat = np.array([[1, 0, 0], [0, 1, 0]])
tdg.plot_2d(projection_mat @ a)

# q_27
z_ax = a[2, :]
remove_index = np.where(np.logical_and(z_ax > -0.4, z_ax < 0.1))
remove_index = remove_index[0]
conditional_a = a[:, remove_index]
tdg.plot_2d(projection_mat @ conditional_a)

plt.show()
