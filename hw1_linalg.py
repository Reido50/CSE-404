import numpy as np
from numpy import linalg as LA

A = np.mat("2 1 3; 1 1 2; 3 2 5")
print("A:", A)

print("Eigenvalues: ", LA.eigvals(A))

eigenvalue, eigenvector = LA.eig(A)

print("First tuple of eig: \n", eigenvalue)
print("Second tuple of eig: \n", eigenvector)
