from d2c.simulatedDAGs import SimulatedDAGs
from d2c.D2C import D2C

N_JOBS = 4

n_dags = 4
n_observations = 50
n_nodes = 5

if __name__ == "__main__":
    simulated_dags = SimulatedDAGs(n_dags, n_observations, n_nodes, n_jobs=N_JOBS)
    simulated_dags.generate_dags()

    simulated_dags.simulate_observations()

    d2c = D2C(simulated_dags, n_jobs=N_JOBS)
    d2c.initialize()


    dataframe = d2c.get_decsriptors_df()
    dataframe.to_csv('dataframe.csv', index=False)