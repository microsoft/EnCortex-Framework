# Data Loading

Data Loading can occur in two ways:

1. Load from a file.
2. Access from a database

Data is always indexed by the `np.datetime64` timestep. In training/simulaton, we load the training/simulation data to a `sqlite3 in-memory database` for faster querying.

In production, we fetch the latest available forecasts for all the entities through the central database that hosts the data.
