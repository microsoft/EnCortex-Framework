## Energy Arbitrage

### Advanced: Running on [AML Compute](run/azureml-compute)

In order to run the Energy Arbitrage experiments, place your scenario code `scenario_code.py` and run the script titled `run_encortex_on_aml.py` in an empty folder(let's name it `encortex_on_aml`). The directory structure should look something like the following tree.

```
encortex_on_aml/
├── datasets
└── scenario_code.py
```

````{eval-rst}
.. important::

   It is important the folder containing `scenario_code.py` contains only the files required for the experiment since AML saves a copy of all the files in that directory as a part of code versioning for every experiment.
````

````{eval-rst}
.. note::

   Don't forget to add your wandb API Key when using Wandb.
````

````{eval-rst}
.. tabs::

   .. code-tab:: py scenario_code.py

          import argparse
          import logging
          import numpy as np
          from rsome import ro
          from rsome.lp import Affine

          from encortex.contract import Contract
          from encortex.decision_unit import DecisionUnit
          from encortex.environments.battery_arbitrage import BatteryArbitrageMILPEnv
          from encortex.environments.battery_arbitrage import BatteryArbitrageRLEnv
          from encortex.grid import Grid
          from encortex.sources import Battery
          from encortex.optimizers.milp import MILPOptimizer
          from encortex.optimizers.rl import EnCortexRLOptimizer
          from encortex.data import MarketData
          from encortex.backend import DFBackend
          from encortex.callbacks.env_callback import EnvCallback
          from encortex.logger import get_experiment_logger

          import pandas as pd

          logging.basicConfig(level="DEBUG", filename="battery_arbitrage_example.log")
          logger = logging.getLogger(__name__)

          parser = argparse.ArgumentParser()
          parser.add_argument("--milp", action="store_true")

          args = parser.parse_args()
          milp = args.milp

          battery = Battery(
              (60, "m"),
              "Battery",
              2,
              "Battery",
              10,
              1.0,
              1.0,
              1.0,
              90,
              0.1,
              False,
              0.8,
              20.0,
              0.0,
              10,
              False,
              schedule="0 * * * *",
          )

          forecast_df = pd.read_csv("data/UK_data_2020_meanprev_shortened.csv")
          actual_df = pd.read_csv("data/UK_data_2020_actuals_shortened.csv")

          forecast_df[["emissions", "prices"]] = forecast_df[
              ["emissions", "prices"]
          ].apply(lambda x: np.float32(x))
          actual_df[["emissions", "prices"]] = actual_df[["emissions", "prices"]].apply(
              lambda x: np.float32(x)
          )

          grid_data = MarketData.parse_backend(
              3,
              True,
              3,
              3,
              np.timedelta64(5, "m"),
              price_forecast=DFBackend(forecast_df["prices"], forecast_df["timestamps"]),
              price_actual=DFBackend(actual_df["prices"], actual_df["timestamps"]),
              carbon_emissions_forecast=DFBackend(
                  forecast_df["emissions"], forecast_df["timestamps"]
              ),
              carbon_emissions_actual=DFBackend(
                  actual_df["emissions"], actual_df["timestamps"]
              ),
              carbon_prices_forecast=DFBackend(
                  forecast_df["prices"], forecast_df["timestamps"]
              ),
              carbon_prices_actual=DFBackend(
                  actual_df["prices"], actual_df["timestamps"]
              ),
              volume_forecast=DFBackend(None, None),
              volume_actual=DFBackend(None, None),
          )
          grid = Grid(
              (60, "m"),
              "Grid",
              3,
              "Grid",
              "0 * * * *",
              np.timedelta64(0, "h"),
              (0, "h"),
              (0, "h"),
              schedule="0 * * * *",
              data=grid_data,
          )

          contract = Contract(battery, grid, True)

          du = DecisionUnit([contract])

          exp_logger = get_experiment_logger("wandb")
          if milp:
              env = BatteryArbitrageMILPEnv(
                  du,
                  np.datetime64("2020-01-01T00:00"),
                  42,
                  callbacks=[EnvCallback()],
                  exp_logger=exp_logger,
              )

              opt = MILPOptimizer(env, obj="min")
              done = False

              objective_function_values = []
              reward_values = []

              time = np.datetime64("2020-01-01T00:00")
              while not done:
                  obj, reward, time, done = opt.run(time)
                  print(f"Time: {time} | Obj: {obj} | Rew: {reward}")
                  objective_function_values.append(obj)
                  reward_values.append(reward)

          else:
              env = BatteryArbitrageRLEnv(
                  du,
                  np.datetime64("2020-01-01T00:00"),
                  42,
                  callbacks=[],
                  action_window=np.timedelta64(1, "h"),
                  future_window=np.timedelta64(24, "h"),
                  exp_logger=exp_logger,
              )
              env.log_experiment_hyperparameters(env.export_config())
              opt = EnCortexRLOptimizer(
                  env,
                  "dqn",
                  "MlpPolicy",
                  40,
                  target_update_interval=1,
                  verbose=2,
                  batch_size=8,
              )
              opt.train(10000, log_interval=1)
````

Now,

1. Create a folder `data` and add your `train.csv` and `test.csv` files
2. Run `python exp_aml.py` on the terminal or `!python exp_aml.py` inside a jupyter-notebook cell to submit a job to AML.
3. You'll be prompted to login via your Microsoft account. Only after successful authentication is the job submitted.

> Results can be viewed on the AzureML studio. Follow the [instructions here](setup/aml-job-results) to visualize the runs. Detailed visualizations are missing(Charts will be added as a part of AzureML Image support in the future).
