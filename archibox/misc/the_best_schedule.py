import numpy as np

T = 25
H = 1.0
S = 1.0

tt = 0.5 * np.ones(T)  # or any point in [0,1]^N
for i in range(999):
    total = tt.sum()
    suffix_sums = np.cumsum(tt[::-1])[::-1]
    weights = np.exp(-2 * S * suffix_sums)
    B = np.exp(-2 * h * s * M)
    C = prefix_sum(A * B)  # C[k] = sum_{i<k} A[i]*B[i]
    # 2) gradient
    grad = -h * np.exp(-h * S) + h * B - 2 * h * h * s * C
    # 3) take a small step and project back into [0,1]
    A = np.clip(A - η * grad, 0.0, 1.0)
