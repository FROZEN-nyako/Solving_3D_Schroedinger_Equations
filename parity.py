import numpy as np

def P(v, n):
    u = np.empty(n ** 3)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u[n ** 2 * i + n * j + k] = v[n ** 2 * (n - 1 - i) + n * (n - 1 - j) + (n - 1 - k)].real
    return u