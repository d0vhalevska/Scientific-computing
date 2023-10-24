import numpy as np
A = np.array([[-4, 2], [6, -3]])
U, Sigma, VT = np.linalg.svd(A)
print(Sigma)