import argparse
import logging
import numpy as np
from omegaconf import OmegaConf
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

config = OmegaConf.load("examples/battery_arbitrage_config.yml")

args = parser.parse_args()
milp = args.milp

battery = Battery(
    config["time_unit"],
    "Battery",
    2,
    "Battery",
    config["storage_capacity"],
    config["efficiency"],
    config["efficiency"],
    config["soc_initial"],
    config["depth_of_discharge"],
    config["soc_minimum"],
    config["degradation_flag"],
    config["min_battery_capacity_factor"],
    config["battery_cost_per_kWh"],
    config["reduction_coefficient"],
    config["degradation_period"],
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
