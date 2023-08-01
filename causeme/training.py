from d2c.simulatedTimeSeries import SimulatedTimeSeries
from d2c.D2C import D2C

n_series = 100
n_observations = 150
n_variables = 3

simulated_timeseries = SimulatedTimeSeries(n_series, n_observations, n_variables)
simulated_timeseries.generate_time_series()

d2c = D2C(simulated_timeseries.get_dags(),simulated_timeseries.get_observations())
d2c.initialize()
d2c.save_descriptors_df('timeseries.csv')
