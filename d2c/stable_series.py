import random
import math

# Helper function to compute mean over given indices
def mean_over_indices(Y_t, indices):
    return sum(Y_t[i] for i in indices) / len(indices)

# Indicator function
def I(condition):
    return 1 if condition else 0

# Sign function
def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


# Time series update function for the first formula
def time_series_update_1(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = -0.4 * (3 - Y_bar_t**2) / (1 + Y_bar_t**2)
    term2 = 0.6 * (3 - (Y_bar_t_minus_1 - 0.5)**3) / (1 + (Y_bar_t_minus_1 - 0.5)**4)

    return term1 + term2 + W[t][j]

# Time series update function for the second formula
def time_series_update_2(Y, t, j, N_j, W):
    Y_bar_t_minus_5 = mean_over_indices(Y[t-5], N_j)
    Y_bar_t_minus_9 = mean_over_indices(Y[t-9], N_j)

    term1 = (0.4 - 2 * math.exp(-50 * Y_bar_t_minus_5**2)) * Y_bar_t_minus_5
    term2 = (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_9**2)) * Y_bar_t_minus_9

    return term1 + term2 + W[t][j]

# Time series update function for the third formula
def time_series_update_3(Y, t, j, N_j, W):
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = 1.5 * math.sin(math.pi / 2 * Y_bar_t_minus_1)
    term2 = -math.sin(math.pi / 2 * Y_bar_t_minus_2)

    return term1 + term2 + W[t][j]

# Time series update function for the fourth formula you provided
def time_series_update_4(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = 2 * math.exp(-0.1 * Y_bar_t**2) * Y_bar_t
    term2 = -math.exp(-0.1 * Y_bar_t_minus_1**2) * Y_bar_t_minus_1

    return term1 + term2 + W[t][j]


# Time series update function for the fifth formula you provided
def time_series_update_5(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    term1 = -2 * Y_bar_t * I(Y_bar_t < 0)
    term2 = 0.4 * Y_bar_t * I(Y_bar_t < 0)

    return term1 + term2 + W[t][j]

# Time series update function for the sixth formula you provided
def time_series_update_6(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = 0.8 * math.log(1 + 3 * Y_bar_t**2)
    term2 = -0.6 * math.log(1 + 3 * Y_bar_t_minus_2**2)

    return term1 + term2 + W[t][j]

def time_series_update_7(Y, t, j, N_j, W):
    Y_bar_t_minus_5 = mean_over_indices(Y[t-5], N_j)
    Y_bar_t_minus_9 = mean_over_indices(Y[t-9], N_j)

    term1_part1 = 0.4 - 2 * math.cos(40 * Y_bar_t_minus_5) * math.exp(-30 * Y_bar_t_minus_5**2)
    term1 = term1_part1 * Y_bar_t_minus_5
    term2 = (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_9**2)) * Y_bar_t_minus_9

    return term1 + term2 + W[t][j]

def time_series_update_8(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = (0.5 - 1.1 * math.exp(-50 * Y_bar_t**2)) * Y_bar_t
    term2 = (0.3 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2

    return term1 + term2 + W[t][j]

# Time series update function for the ninth formula you provided
def time_series_update_9(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = 0.3 * Y_bar_t
    term2 = 0.6 * Y_bar_t_minus_1
    term3_numerator = 0.1 - 0.9 * Y_bar_t + 0.8 * Y_bar_t_minus_1
    term3_denominator = 1 + math.exp(-10 * Y_bar_t)
    term3 = term3_numerator / term3_denominator

    return term1 + term2 + term3 + W[t][j]

# Time series update function for the tenth formula you provided
def time_series_update_10(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    return sign(Y_bar_t) + W[t][j]

# Time series update function for the eleventh formula you provided
def time_series_update_11(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    term1 = 0.8 * Y_bar_t
    term2_denominator = 1 + math.exp(-10 * Y_bar_t)
    term2 = -0.8 * Y_bar_t / term2_denominator

    return term1 + term2 + W[t][j]

# Time series update function for the twelfth formula you provided
def time_series_update_12(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = 0.3 * Y_bar_t
    term2 = 0.6 * Y_bar_t_minus_1
    term3_numerator = 0.1 - 0.9 * Y_bar_t + 0.8 * Y_bar_t_minus_1
    term3_denominator = 1 + math.exp(-10 * Y_bar_t)
    term3 = term3_numerator / term3_denominator

    return term1 + term2 + term3 + W[t][j]

def time_series_update_13(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = 0.38 * Y_bar_t * (1 - Y_bar_t_minus_1)

    return term1 + W[t][j]

def time_series_update_14(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    if Y_bar_t < 1:
        return -0.5 * Y_bar_t + W[t][j]
    else:
        return 0.4 * Y_bar_t + W[t][j]

def time_series_update_15(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    if abs(Y_bar_t) < 1:
        return 0.9 * Y_bar_t + W[t][j]
    else:
        return -0.3 * Y_bar_t + W[t][j]

def update_x(x_t):
    return 1 - x_t

# Time series update function for the sixteenth formula you provided
def time_series_update_16(Y, t, j, N_j, W, x_t):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    if x_t == 1:
        return -0.5 * Y_bar_t + W[t][j]
    else:
        return 0.4 * Y_bar_t + W[t][j]

def time_series_update_17(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)
    Y_bar_t_minus_3 = mean_over_indices(Y[t-3], N_j)

    squared_sum = (
        Y_bar_t**2 + 
        0.3 * Y_bar_t_minus_1**2 + 
        0.2 * Y_bar_t_minus_2**2 + 
        0.1 * Y_bar_t_minus_3**2
    )

    coefficient = math.sqrt(0.000019 + 0.846 * squared_sum)

    return coefficient * W[t][j]

# Simple simulation

# Initialize parameters
T = 20  # Time steps
N = 5   # number of j indices
Y1 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 1
Y2 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 2
Y3 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 3
Y4 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 4
Y5 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 5
Y6 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 6
Y7 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 7
Y8 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 8
Y9 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 9
Y10 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 10
Y11 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 11
Y12 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 12
Y13 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 13
Y14 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 14
Y15 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 15

x = [1] + [0 for _ in range(T)]
Y16 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 16
Y17 = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(T+1)]  # For time series 17



W = [[random.uniform(-0.1, 0.1) for _ in range(N)] for _ in range(T+1)]  # Some noise
N_j = [0, 2, 3]  # Sample set of indices for N_j

# Time series iteration for the first formula
for t in range(T):
    for j in range(N):
        if t > 0:
            Y1[t+1][j] = time_series_update_1(Y1, t, j, N_j, W)

# Time series iteration for the second formula
for t in range(9, T):  # Start from t=9 as the formula requires t-9
    for j in range(N):
        Y2[t+1][j] = time_series_update_2(Y2, t, j, N_j, W)

# Time series iteration for the third formula
for t in range(2, T):  # Start from t=2 as the formula requires t-2
    for j in range(N):
        Y3[t+1][j] = time_series_update_3(Y3, t, j, N_j, W)

# Time series iteration for the fourth formula
for t in range(1, T):  # Start from t=1 as the formula requires t-1
    for j in range(N):
        Y4[t+1][j] = time_series_update_4(Y4, t, j, N_j, W)

for t in range(T):
    for j in range(N):
        Y5[t+1][j] = time_series_update_5(Y5, t, j, N_j, W)

for t in range(2, T):  # Start from t=2 as the formula requires t-2
    for j in range(N):
        Y6[t+1][j] = time_series_update_6(Y6, t, j, N_j, W)

for t in range(9, T):  # Start from t=9 as the formula requires t-9
    for j in range(N):
        Y7[t+1][j] = time_series_update_7(Y7, t, j, N_j, W)

for t in range(2, T):  # Start from t=2 as the formula requires t-2
    for j in range(N):
        Y8[t+1][j] = time_series_update_8(Y8, t, j, N_j, W)

for t in range(1, T):  # Start from t=1 as the formula requires t-1
    for j in range(N):
        Y9[t+1][j] = time_series_update_9(Y9, t, j, N_j, W)

for t in range(T):
    for j in range(N):
        Y10[t+1][j] = time_series_update_10(Y10, t, j, N_j, W)

for t in range(T):
    for j in range(N):
        Y11[t+1][j] = time_series_update_11(Y11, t, j, N_j, W)

for t in range(1, T):  # Start from t=1 as the formula requires t-1
    for j in range(N):
        Y12[t+1][j] = time_series_update_12(Y12, t, j, N_j, W)

for t in range(1, T):  # Start from t=1 as the formula requires t-1
    for j in range(N):
        Y13[t+1][j] = time_series_update_13(Y13, t, j, N_j, W)

for t in range(T):
    for j in range(N):
        Y14[t+1][j] = time_series_update_14(Y14, t, j, N_j, W)

for t in range(T):
    for j in range(N):
        Y15[t+1][j] = time_series_update_15(Y15, t, j, N_j, W)

for t in range(T):
    for j in range(N):
        Y16[t+1][j] = time_series_update_16(Y16, t, j, N_j, W, x[t])
    x[t+1] = update_x(x[t])

for t in range(3, T):  # Start from t=3 as the formula requires t-3
    for j in range(N):
        Y17[t+1][j] = time_series_update_17(Y17, t, j, N_j, W)




print("Time Series 1:")
print(Y1)
print("\nTime Series 2:")
print(Y2)
print("\nTime Series 3:")
print(Y3)
print("\nTime Series 4:")
print(Y4)
print("\nTime Series 5:")
print(Y5)
print("\nTime Series 6:")
print(Y6)
print("\nTime Series 7:")
print(Y7)
print("\nTime Series 8:")
print(Y8)
print("\nTime Series 9:")
print(Y9)
print("\nTime Series 10:")
print(Y10)
print("\nTime Series 11:")
print(Y11)
print("\nTime Series 12:")
print(Y12)
print("\nTime Series 13:")
print(Y13)
print("\nTime Series 14:")
print(Y14)
print("\nTime Series 15:")
print(Y15)
print("\nTime Series 16:")
print(Y16)
print("\nTime Series 17:")
print(Y17)