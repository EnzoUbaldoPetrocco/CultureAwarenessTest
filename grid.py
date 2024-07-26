import numpy as np

A = np.logspace(-4, -1, 11)
B = np.logspace(-4, -1, 6)

print(f"A is {A}")
print(f"B is {B}")

A = [i for i in A if i not in B]

print(f"Result: {A}")