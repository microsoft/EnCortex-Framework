# EnCortex: Tutorial

## Entities

The class `Entity` is a general encapsulation of all the entities(sometimes referred to as participants) is a general abstract class to represent any entity. To create a new category of entity, inherit the class and implement all the required methods. To associate an action with an entity, the action attribute an entity must inherit and implement the `Action` class.

Initializing an entity:

```python
from encortex.grid import Grid
from encortex.data import MarketData
from encortex.backend import Backend

grid_data = MarketData(
    ...
)

grid = Market(..., ...)

grid.data.<any parameter you want to access>
```

## Decision Unit

Decision unit registers everything via a contract. Currently, based on the broad categories of entities, these are parsed and can be accessed from the decision unit.

```python
from encortex.decision_unit import DecisionUnit
from encortex.contract import Contract

contracts = [Contract(...), Contract(...)]
decision_unit = DecisionUnit(contracts)
```

## Environment

The Environment handles the dynamics, data handling and state setting of all the entities in the decision unit.

To implement a new environment, inherit the `encortex.environment.Environment` class and implement the required methods.

## Optimizer

The optimizers observes the environment and solves for a reward objective.

To implement a new optimizer, inherit the `encortex.optimizer.Optimizer` class and implement the required methods.


# Common Code

```python
import os
import numpy as np
import pandas as pd

from encortex.contract import Contract
from encortex.decision_unit import DecisionUnit
from encortex import Market
from encortex.sources.solar import Solar
from encortex.visualize import plot_decision_unit_graph

from encortex.market import Market
from encortex.data import MarketData, SourceData
from encortex.backend import DFBackend



def test_decision_unit():
    forecast_data = pd.read_csv('data/Ayana_solar.csv')

    solar_data = SourceData.parse_backend(
        entity_id=1,
        in_memory=True,
        source_id=1,
        entity_forecasting_id=1,
        timestep=np.timedelta64(15, "m"),
        connect_to_db=False,
        generation_forecast=DFBackend(forecast_data["generation"], forecast_data["timestamps"]),
    )
    source = Solar(
        timestep=np.timedelta64(15, "m"),
        name="Solar",
        id=1,
        description="Solar",
        max_capacity=100,
        data=solar_data,
        disable_action=True
    )

    forecast_df = pd.read_csv("data/forecast_grid_data.csv")
    actual_df = pd.read_csv("data/forecast_grid_data.csv")

    market_data = MarketData.parse_backend(
        entity_id=0,
        in_memory=True,
        market_id=0,
        entity_forecasting_id=0,
        timestep=np.timedelta64(15, "m"),
        connect_to_db=False,
        price_forecast=DFBackend(forecast_df["forecast_prices"], forecast_df["timestamps"]),
        volume_forecast=DFBackend(forecast_df["forecast_prices"], forecast_df["timestamps"]),
        price_actual=DFBackend(actual_df["forecast_prices"], actual_df["timestamps"]),
        volume_actual=DFBackend(actual_df["forecast_prices"], actual_df["timestamps"]),
    )

    market = Market(
        timestep=np.timedelta64(15, "m"),
        name="DAM",
        id=0,
        description="DAM",
        bid_start_time_schedule="0 12 * * *",
        bid_window=np.timedelta64(3, "h"),
        commit_start_schedule=np.timedelta64(12, "h"),
        commit_end_schedule=np.timedelta64(36, "h"),
        data=market_data,
        disable_bidding=False
    )

    contract = Contract(
        source,
        market,
        "0/15 10 * * *",
        np.timedelta64("15", "m"),
        "0/45 * * * *",
        np.timedelta64("30", "m"),
        {"penalty": lambda expected, actual: ((actual - expected) ** 2).sum()},
    )

    decision_unit = DecisionUnit([contract])

    assert decision_unit.sources == [source], "Mismatch in sources"
    assert decision_unit.markets == [market], "Mismatch in markets"
    assert decision_unit.contracts == [contract], "Mismatch in contracts"

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    plot_decision_unit_graph(decision_unit)
    decision_unit.generate_schedule(np.datetime64("2020-01-01T00:00:00"))

    assert decision_unit.timestep == np.timedelta64(15, "m"), f"Mismatch in timestep: {decision_unit.timestep}"
    assert decision_unit.horizon == np.timedelta64(1, "D"), f"Mismatch in horizon: {decision_unit.horizon}"

    assert decision_unit.schedule is not None, f"Schedule is None"

```