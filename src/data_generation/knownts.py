# %%
import random
import math
import networkx as nx

# %%
# Helper function to compute mean over given indices
def mean_over_indices(Y_t, indices):
    return sum(Y_t[i] for i in indices) / len(indices)

# Indicator function
def I(condition):
    return 1 if condition else 0

# Sign function
def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


# %% [markdown]
# $Y_{t+1}[j]=-0.4 \frac{\left(3-\bar{Y}_t\left[\mathcal{N}_j\right]^2\right)}{\left(1+\bar{Y}_t\left[\mathcal{N}_j\right]^2\right)}+0.6 \frac{3-\left(\bar{Y}_{t-1}\left[\mathcal{N}_j\right]-0.5\right)^3}{1+\left(\bar{Y}_{t-1}\left[\mathcal{N}_j\right]-0.5\right)^4}+W_{t+1}[j]$

# %%
# Time series update function for the first formula
def time_series_update_1(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = -0.4 * (3 - Y_bar_t**2) / (1 + Y_bar_t**2)
    term2 = 0.6 * (3 - (Y_bar_t_minus_1 - 0.5)**3) / (1 + (Y_bar_t_minus_1 - 0.5)**4)

    return term1 + term2 + W[t][j]

# %%

def build_DAG_time_series_1(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx
    G = nx.DiGraph()

    # Initialize
    for t in range(T+1):
        for j in range(N):
            G.add_node(f"Y[{t}][{j}]")

    # Add nodes and edges to the DAG based on the dependencies
    for t in range(T):
        for j in range(N):
            # if t > 0:
                # Add edges from the current and previous Y to the next Y
                for nj in N_j[j]:
                    G.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                    if t != T-1:
                        G.add_edge(f"Y[{t}][{nj}]", f"Y[{t+2}][{j}]")
                
                
    return G

# %% [markdown]
# $Y_{t+1}[j]=\left(0.4-2 \cos \left(40 \bar{Y}_{t-5}\left[\mathcal{N}_j\right]\right) \exp \left(-30 \bar{Y}_{t-5}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-5}\left[\mathcal{N}_j\right]+\left(0.5-0.5 \exp \left(-50 \bar{Y}_{t-9}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-9}\left[\mathcal{N}_j\right]+W_{t+1}[j]$
# We replace the above with the following 
# $Y_{t+1}[j]=\left(0.4-2 \cos \left(40 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]\right) \exp \left(-30 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-2}\left[\mathcal{N}_j\right]+\left(0.5-0.5 \exp \left(-50 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-1}\left[\mathcal{N}_j\right]+W_{t+1}[j]$

# %%


# Time series update function for the second formula
def time_series_update_2(Y, t, j, N_j, W):
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = (0.4 - 2 * math.exp(-50 * Y_bar_t_minus_1**2)) * Y_bar_t_minus_1
    term2 = (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2

    return term1 + term2 + W[t][j]


# %%

def build_DAG_time_series_2(T, N_j, N):
    # Create a new Directed Acyclic Graph (DAG) using networkx for the second formula
    G2 = nx.DiGraph()

    # Initialize
    for t in range(T+1):
        for j in range(N):
            G2.add_node(f"Y[{t}][{j}]")

    # Add nodes and edges to the DAG based on the dependencies for the second formula
    for t in range(T):
        for j in range(N):
            
            for nj in N_j[j]: 
                if t >= 1:
                    G2.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 2:
                    G2.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t+1}][{j}]")
    return G2

# %% [markdown]
# $Y_{t+1}[j]=1.5 \sin \left(\pi / 2 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]\right)-\sin \left(\pi / 2 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]\right)+W_{t+1}[j]$

# %%
# Time series update function for the third formula
def time_series_update_3(Y, t, j, N_j, W):
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = 1.5 * math.sin(math.pi / 2 * Y_bar_t_minus_1)
    term2 = -math.sin(math.pi / 2 * Y_bar_t_minus_2)

    return term1 + term2 + W[t][j]


# %%
def build_DAG_time_series_3(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the third formula
    G3 = nx.DiGraph()
    
    # Initialize
    for t in range(T+1):
        for j in range(N):
            G3.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the third formula
    for t in range(T):  # Start from t=2 since the formula depends on t-1 and t-2
        for j in range(N):

            # Add edges from the previous and the one before last Y to the next Y
            for nj in N_j[j]:
                if t >= 1:
                    G3.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 2:
                    G3.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G3


# %% [markdown]
# $Y_{t+1}[j]=2 \exp \left(-0.1 \bar{Y}_t\left[\mathcal{N}_j\right]^2\right) \bar{Y}_t\left[\mathcal{N}_j\right]-\exp \left(-0.1 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]^2\right) \bar{Y}_{t-1}\left[\mathcal{N}_j\right]+W_{t+1}[j]$

# %%
# Time series update function for the fourth formula you provided
def time_series_update_4(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = 2 * math.exp(-0.1 * Y_bar_t**2) * Y_bar_t
    term2 = -math.exp(-0.1 * Y_bar_t_minus_1**2) * Y_bar_t_minus_1

    return term1 + term2 + W[t][j]



# %%
def build_DAG_time_series_4(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the fourth formula
    G4 = nx.DiGraph()
    
    # Initialize
    for t in range(T+1):
        for j in range(N):
            G4.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the fourth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):
            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G4.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 1:
                    G4.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G4

# %% [markdown]
# $Y_{t+1}[j]=-2 \bar{Y}_t\left[\mathcal{N}_j\right] I\left(\bar{Y}_t\left[\mathcal{N}_j\right]<0\right)+0.4 \bar{Y}_t\left[\mathcal{N}_j\right] I\left(\bar{Y}_t\left[\mathcal{N}_j\right]<0\right)+W_{t+1}[j]$

# %%
# Time series update function for the fifth formula you provided
def time_series_update_5(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    term1 = -2 * Y_bar_t * I(Y_bar_t < 0)
    term2 = 0.4 * Y_bar_t * I(Y_bar_t < 0)

    return term1 + term2 + W[t][j]


# %%
def build_DAG_time_series_5(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the fifth formula
    G5 = nx.DiGraph()
    
    # Initialize
    for t in range(T+1):
        for j in range(N):
            G5.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the fifth formula
    for t in range(T):  
        for j in range(N):
            # Add edges from the current Y to the next Y
            for nj in N_j[j]:
                G5.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G5


# %% [markdown]
# $Y_{t+1}[j]=0.8 \log \left(1+3 \bar{Y}_t\left[\mathcal{N}_j\right]^2\right)-0.6 \log \left(1+3 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]^2\right)+W_{t+1}[j]$

# %%
# Time series update function for the sixth formula you provided
def time_series_update_6(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = 0.8 * math.log(1 + 3 * Y_bar_t**2)
    term2 = -0.6 * math.log(1 + 3 * Y_bar_t_minus_2**2)

    return term1 + term2 + W[t][j]


# %%
def build_DAG_time_series_6(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the sixth formula
    G6 = nx.DiGraph()

    # Initialize
    for t in range(T+1):
        for j in range(N):
            G6.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the sixth formula
    for t in range(T):
        for j in range(N):
            for nj in N_j[j]:
                G6.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 2:
                    G6.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t+1}][{j}]")

    return G6

# %% [markdown]
# $Y_{t+1}[j]=\left(0.4-2 \cos \left(40 \bar{Y}_{t-5}\left[\mathcal{N}_j\right]\right) \exp \left(-30 \bar{Y}_{t-5}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-5}\left[\mathcal{N}_j\right]+\left(0.5-0.5 \exp \left(-50 \bar{Y}_{t-9}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-9}\left[\mathcal{N}_j\right]+W_{t+1}[j]$
# We replace the above with the following
# $Y_{t+1}[j]=\left(0.4-2 \cos \left(40 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]\right) \exp \left(-30 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-2}\left[\mathcal{N}_j\right]+\left(0.5-0.5 \exp \left(-50 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-1}\left[\mathcal{N}_j\right]+W_{t+1}[j]$
# %%
def time_series_update_7(Y, t, j, N_j, W):
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1_part1 = 0.4 - 2 * math.cos(40 * Y_bar_t_minus_2) * math.exp(-30 * Y_bar_t_minus_2**2)
    term1 = term1_part1 * Y_bar_t_minus_2
    term2 = (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_1**2)) * Y_bar_t_minus_1

    return term1 + term2 + W[t][j]


# %%
def build_DAG_time_series_7(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the sixth formula
    G7 = nx.DiGraph()
    
    # Initialize
    for t in range(T+1):
        for j in range(N):
            G7.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the seventh formula
    for t in range(T):
        for j in range(N):
            for nj in N_j[j]:
                if t >= 2:
                    G7.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 1:
                    G7.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")

    return G7


# %% [markdown]
# $Y_{t+1}[j]=\left(0.5-1.1 \exp \left(-50 \bar{Y}_t\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_t\left[\mathcal{N}_j\right]+\left(0.3-0.5 \exp \left(-50 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]^2\right)\right) \bar{Y}_{t-2}\left[\mathcal{N}_j\right]+W_{t+1}[j]$

# %%
def time_series_update_8(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

    term1 = (0.5 - 1.1 * math.exp(-50 * Y_bar_t**2)) * Y_bar_t
    term2 = (0.3 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2

    return term1 + term2 + W[t][j]


# %%
def build_DAG_time_series_8(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the sixth formula
    G8 = nx.DiGraph()
    
    # Initialize
    for t in range(T+1):
        for j in range(N):
            G8.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges for the eighth formula
    for t in range(T):  # Start from t=2 since the formula depends on t and t-2
        for j in range(N):
            for nj in N_j[j]:
                G8.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 2:
                    G8.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t+1}][{j}]")

    return G8


# %% [markdown]
# $Y_{t+1}[j]=0.3 \bar{Y}_t\left[\mathcal{N}_j\right]+0.6 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]+\frac{\left(0.1-0.9 \bar{Y}_t\left[\mathcal{N}_j\right]+0.8 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]\right)}{\left(1+\exp \left(-10 \bar{Y}_t\left[\mathcal{N}_j\right]\right)\right)}+W_{t+1}[j]$

# %%

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


# %%
def build_DAG_time_series_9(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the ninth formula
    G9 = nx.DiGraph()

    for t in range(T+1):
        for j in range(N):
            G9.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the ninth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):
            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G9.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 1:
                    G9.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G9


# %% [markdown]
# $Y_{t+1}[j]=\operatorname{sign}\left(\bar{Y}_t\left[\mathcal{N}_j\right]\right)+W_{t+1}[j]$

# %%
# Time series update function for the tenth formula you provided
def time_series_update_10(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    return sign(Y_bar_t) + W[t][j]


# %%
def build_DAG_time_series_10(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the tenth formula
    G10 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G10.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the tenth formula
    for t in range(T):  # Start from t=1 since the formula depends on t
        for j in range(N):
            # Add edges from the current Y to the next Y
            for nj in N_j[j]:
                G10.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G10
                

# %% [markdown]
# $Y_{t+1}[j]=0.8 \bar{Y}_t\left[\mathcal{N}_j\right]-\frac{0.8 \bar{Y}_t\left[\mathcal{N}_j\right]}{\left(1+\exp \left(-10 \bar{Y}_t\left[\mathcal{N}_j\right]\right)\right)}+W_{t+1}[j]$

# %%
# Time series update function for the eleventh formula you provided
def time_series_update_11(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    term1 = 0.8 * Y_bar_t
    term2_denominator = 1 + math.exp(-10 * Y_bar_t)
    term2 = -0.8 * Y_bar_t / term2_denominator

    return term1 + term2 + W[t][j]


# %%
def build_DAG_time_series_11(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the eleventh formula
    G11 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G11.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the eleventh formula
    for t in range(T):  # Start from t=1 since the formula depends on t
        for j in range(N):

            # Add edges from the current Y to the next Y
            for nj in N_j[j]:
                G11.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G11

# %% [markdown]
# $Y_{t+1}[j]=0.3 \bar{Y}_t\left[\mathcal{N}_j\right]+0.6 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]+\frac{\left(0.1-0.9 \bar{Y}_t\left[\mathcal{N}_j\right]+0.8 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]\right)}{\left(1+\exp \left(-10 \bar{Y}_t\left[\mathcal{N}_j\right]\right)\right)}+W_{t+1}[j]$

# %%
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


# %%
def build_DAG_time_series_12(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the twelfth formula
    G12 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G12.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the twelfth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):

            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G12.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 1:
                    G12.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
            

    return G12

# %% [markdown]
# $Y_{t+1}[j]=0.38 \bar{Y}_t\left[\mathcal{N}_j\right]\left(1-\bar{Y}_{t-1}\left[\mathcal{N}_j\right]\right)+W_{t+1}[j]$

# %%
def time_series_update_13(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

    term1 = 0.38 * Y_bar_t * (1 - Y_bar_t_minus_1)

    return term1 + W[t][j]


# %%
def build_DAG_time_series_13(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the thirteenth formula
    G13 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G13.add_node(f"Y[{t}][{j}]")
    
    # Add nodes and edges to the DAG based on the dependencies for the thirteenth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):
            # Add the current and previous Y nodes if not already present

            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G13.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 1:
                    G13.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
            
    return G13

# %% [markdown]
# $Y_{t+1}[j]=\left\{\begin{array}{l}-0.5 \bar{Y}_t\left[\mathcal{N}_j\right] \quad \text { if } \quad \bar{Y}_t\left[\mathcal{N}_j\right]<1 \\ 0.4 \bar{Y}_t\left[\mathcal{N}_j\right]\end{array}\right.$

# %%
def time_series_update_14(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    if Y_bar_t < 1:
        return -0.5 * Y_bar_t + W[t][j]
    else:
        return 0.4 * Y_bar_t + W[t][j]


# %%
def build_DAG_time_series_14(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the thirteenth formula
    G14 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G14.add_node(f"Y[{t}][{j}]")
    

    # Add nodes and edges to the DAG based on the dependencies for the thirteenth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):
            # Add the current and previous Y nodes if not already present

            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G14.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
            
    return G14

# %% [markdown]
# $Y_{t+1}[j]=\left\{\begin{array}{l}0.9 \bar{Y}_t\left[\mathcal{N}_j\right]+W_{t+1}[j] \text { if } \quad\left|\bar{Y}_t\left[\mathcal{N}_j\right]\right|<1 \\ -0.3 \bar{Y}_t\left[\mathcal{N}_j\right]+W_{t+1}[j]\end{array}\right.$

# %%
def time_series_update_15(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    if abs(Y_bar_t) < 1:
        return 0.9 * Y_bar_t + W[t][j]
    else:
        return -0.3 * Y_bar_t + W[t][j]


# %%
def build_DAG_time_series_15(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the thirteenth formula
    G15 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G15.add_node(f"Y[{t}][{j}]")
    
    
    # Add nodes and edges to the DAG based on the dependencies for the thirteenth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):
            # Add the current and previous Y nodes if not already present

            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G15.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
            
    return G15

# %% [markdown]
# $\begin{aligned} Y_{t+1}[j] & =\left\{\begin{array}{l}-0.5 \bar{Y}_t\left[\mathcal{N}_j\right]+W_{t+1}[j] \quad \text { if } \quad x_t=1 \\ 0.4 \bar{Y}_t\left[\mathcal{N}_j\right]+W_{t+1}[j]\end{array}\right. \\ x_{t+1} & =1-x_t, x_0=1\end{aligned}$

# %%
def update_x(x_t):
    return 1 - x_t

# Time series update function for the sixteenth formula you provided
def time_series_update_16(Y, t, j, N_j, W, x_t):
    Y_bar_t = mean_over_indices(Y[t], N_j)

    if x_t == 1:
        return -0.5 * Y_bar_t + W[t][j]
    else:
        return 0.4 * Y_bar_t + W[t][j]


# %%
def build_DAG_time_series_16(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the thirteenth formula
    G16 = nx.DiGraph()
    
    for t in range(T+1):
        for j in range(N):
            G16.add_node(f"Y[{t}][{j}]")
    
    
    # Add nodes and edges to the DAG based on the dependencies for the thirteenth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):

            # Add edges from the current and previous Y to the next Y
            for nj in N_j[j]:
                G16.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
            
    return G16

# %% [markdown]
# $Y_{t+1}[j]=\sqrt{0.000019+0.846 *\left(\bar{Y}_t\left[\mathcal{N}_j\right]^2+0.3 \bar{Y}_{t-1}\left[\mathcal{N}_j\right]^2+0.2 \bar{Y}_{t-2}\left[\mathcal{N}_j\right]^2+0.1 \bar{Y}_{t-3}\left[\mathcal{N}_j\right]^2\right)} W_{t+1}[j]$

# %%
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


# %%
def build_DAG_time_series_17(T, N_j, N):
    # Create a Directed Acyclic Graph (DAG) using networkx for the thirteenth formula
    G17 = nx.DiGraph()

    for t in range(T+1):
        for j in range(N):
            G17.add_node(f"Y[{t}][{j}]")
    
    
    # Add nodes and edges to the DAG based on the dependencies for the thirteenth formula
    for t in range(T):  # Start from t=1 since the formula depends on t and t-1
        for j in range(N):

            # Add edges from the current and previous three Y to the next Y
            for nj in N_j[j]:
                G17.add_edge(f"Y[{t}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 1:
                    G17.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 2:
                    G17.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t+1}][{j}]")
                if t >= 3:
                    G17.add_edge(f"Y[{t-3}][{nj}]", f"Y[{t+1}][{j}]")
            
                
    return G17

# Linear cases \[ Y_{t+1}[j] = 0.9 \cdot \bar{Y}_t[\mathcal{N}_j] + W_{t+1}[j] \]
def time_series_update_18(Y, t, j, N_j, W):
    Y_bar_t = mean_over_indices(Y[t], N_j)
    return 0.9 * Y_bar_t + W[t][j]

def build_DAG_time_series_18(T, N_j, N):
    G_linear_1 = nx.DiGraph()
    for t in range(T+1):
        for j in range(N):
            G_linear_1.add_node(f"Y[{t}][{j}]")
            if t > 0:
                for nj in N_j[j]:
                    G_linear_1.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t}][{j}]")
    return G_linear_1

# \[ Y_{t+1}[j] = 0.4 \cdot \bar{Y}_{t-1}[\mathcal{N}_j] + 0.6 \cdot \bar{Y}_{t-2}[\mathcal{N}_j] + W_{t+1}[j] \]
def time_series_update_19(Y, t, j, N_j, W):
    Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
    Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)
    return 0.4 * Y_bar_t_minus_1 + 0.6 * Y_bar_t_minus_2 + W[t][j]

def build_DAG_time_series_19(T, N_j, N):
    G_linear_2 = nx.DiGraph()
    for t in range(T+1):
        for j in range(N):
            G_linear_2.add_node(f"Y[{t}][{j}]")
            if t > 1:  # Since it depends on t-1
                for nj in N_j[j]:
                    G_linear_2.add_edge(f"Y[{t-1}][{nj}]", f"Y[{t}][{j}]")
                    G_linear_2.add_edge(f"Y[{t-2}][{nj}]", f"Y[{t}][{j}]")
    return G_linear_2

# \[ Y_{t+1}[j] = 0.5 \cdot \bar{Y}_{t-3}[\mathcal{N}_j] + W_{t+1}[j] \]
def time_series_update_20(Y, t, j, N_j, W):
    Y_bar_t_minus_3 = mean_over_indices(Y[t-3], N_j)
    return 0.5 * Y_bar_t_minus_3 + W[t][j]

def build_DAG_time_series_20(T, N_j, N):
    G_linear_2 = nx.DiGraph()
    for t in range(T+1):
        for j in range(N):
            G_linear_2.add_node(f"Y[{t}][{j}]")
            if t > 2:  # Since it depends on t-1
                for nj in N_j[j]:
                    G_linear_2.add_edge(f"Y[{t-3}][{nj}]", f"Y[{t}][{j}]")
    return G_linear_2




def get_utils(T):
    
    utils = {}
    utils[1] = (build_DAG_time_series_1,time_series_update_1,range(1, T))
    utils[2] = (build_DAG_time_series_2,time_series_update_2,range(2, T))
    utils[3] = (build_DAG_time_series_3,time_series_update_3,range(2, T))
    utils[4] = (build_DAG_time_series_4,time_series_update_4,range(1, T))
    utils[5] = (build_DAG_time_series_5,time_series_update_5,range(T))
    utils[6] = (build_DAG_time_series_6,time_series_update_6,range(2, T))
    utils[7] = (build_DAG_time_series_7,time_series_update_7,range(2, T))
    utils[8] = (build_DAG_time_series_8,time_series_update_8,range(2, T))
    utils[9] = (build_DAG_time_series_9,time_series_update_9,range(1, T))
    utils[10] = (build_DAG_time_series_10,time_series_update_10,range(T))
    utils[11] = (build_DAG_time_series_11,time_series_update_11,range(T))
    utils[12] = (build_DAG_time_series_12,time_series_update_12,range(1, T))
    utils[13] = (build_DAG_time_series_13,time_series_update_13,range(1, T))
    utils[14] = (build_DAG_time_series_14,time_series_update_14,range(T))
    utils[15] = (build_DAG_time_series_15,time_series_update_15,range(T))
    utils[16] = (build_DAG_time_series_16,time_series_update_16,range(T))
    utils[17] = (build_DAG_time_series_17,time_series_update_17,range(3, T))
    utils[18] = (build_DAG_time_series_18,time_series_update_18,range(T))
    utils[19] = (build_DAG_time_series_19,time_series_update_19,range(2, T))
    utils[20] = (build_DAG_time_series_20,time_series_update_20,range(3, T))

    return utils
