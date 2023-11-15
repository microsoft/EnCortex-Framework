# EnCortex Configuration

We use [Hydra](https://hydra.cc/), an elegant framework for building and configuring the various participants in our framework. All the configs can be found in the `conf` folder in the root directory.

We already define how the framework can be run, so by only specifying the `config.yaml` file, Hydra automatically creates contracts, assigns it to the `Producer` object. The `Producer object` then parses the contracts,
creates a graph of all the connected components, run a maximum connected subgraph algorithm and divides the graph into multiple decision units which run independently, all of them accessing the same database.

This process works slightly differently based on the kind of mode the framework is being executed in:

1. Offline - time of the runner is inferred from the data, and time doesn't flow linearly
2. Production - time of the environment is the system time, data is fetched based on the database tables registered for data fetching to each of the entities.

## Producer

The `Producer` class is an encapsulation of a higher entity that owns the sources and all the entities the "producer" has contracts with. It is mainly for encapsulating common functions needed as the "owner".

## Decision Unit

The `Decision Unit` class is an **independent** set of sources, markets and consumers that have contracts with each other within the decision unit, but are unaffected by any decisions/data of entities outside the decision unit.

