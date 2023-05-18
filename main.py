#!/usr/bin/env python
# coding: utf-8

# In[1]:


from d2c.simulatedDAGs import SimulatedDAGs
from d2c.D2C import D2C


# In[2]:


N_JOBS = 1
ndag = 10
n = 10
no_nodes = 10
function_types = ["linear", "quadratic", "sigmoid"]
quantize = True
seed = 1
sdn = 0.1
verbose = True


# In[3]:


simulated_dags = SimulatedDAGs(ndag, n, no_nodes, function_types, quantize, seed, sdn, verbose, n_jobs=N_JOBS)
simulated_dags.generate_dags()
simulated_dags.simulate_observations()


# In[4]:


d2c = D2C(simulated_dags, n_jobs=N_JOBS)
d2c.initialize()


# In[7]:


dataframe = d2c.get_df()
dataframe


# In[9]:


score = d2c.get_score(metric="f1")
score

