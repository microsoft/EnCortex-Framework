import argparse
import logging
import numpy as np
from omegaconf import OmegaConf
from rsome import ro
from rsome.lp import Affine

from encortex.contract import Contract
from encortex.decision_unit import DecisionUnit
from encortex.environments.microgrid import MicrogridMILPEnv
from encortex.environments.microgrid.rl_env import MicrogridRLEnv
from encortex.grid import Grid
from encortex.sources import Battery, Solar
from encortex.consumer import Consumer
from encortex.microgrid import Microgrid
from encortex.optimizers.milp import MILPOptimizer
from encortex.data import MarketData, SourceData, ConsumerData, UtilityGridData
from encortex.backend import DFBackend
from encortex.optimizers.rl import EnCortexRLOptimizer
from encortex.logger import get_experiment_logger
from encortex.callbacks.offline_rl_callback import (
    OfflineCollectTrajectoryCallback,
)


import pandas as pd
from ast import literal_eval

logging.basicConfig(level="DEBUG", filename="microgrid_example.log")
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--milp", action="store_true")

config = OmegaConf.load("examples/microgrid.yml")

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

forecast_df = pd.read_csv("data/71shortened/France_price_forecast_test.csv")
actual_df = pd.read_csv("data/71shortened/France_price_actual_test.csv")
forecast_df.prices_buy = forecast_df.prices_buy.apply(lambda x: literal_eval(x))
forecast_df.prices_sell = forecast_df.prices_sell.apply(
    lambda x: literal_eval(x)
)

grid_data = UtilityGridData.parse_backend(
    3,
    True,
    3,
    3,
    np.timedelta64("15", "m"),
    price_buy_forecast=DFBackend(
        forecast_df["prices_buy"],
        forecast_df["timestamps"],
        is_static=False,
        timestep=np.timedelta64(15, "m") / np.timedelta64(15, "m"),
    ),
    price_buy_actual=DFBackend(
        actual_df["prices_buy"], actual_df["timestamps"]
    ),  # TODO: @VB Change attributes to support buy and sell price
    price_sell_forecast=DFBackend(
        forecast_df["prices_sell"],
        forecast_df["timestamps"],
        is_static=False,
        timestep=np.timedelta64(15, "m") / np.timedelta64(15, "m"),
    ),
    price_sell_actual=DFBackend(
        actual_df["prices_sell"], actual_df["timestamps"]
    ),
)
grid = Grid(
    (15, "m"),
    "Grid",
    3,
    "Grid",
    "*/15 * * * *",
    np.timedelta64(0, "h"),
    (0, "h"),
    (0, "h"),
    schedule="*/15 * * * *",
    data=grid_data,
)

forecast_df_solar = pd.read_csv("data/71shortened/France_pv_forecast_test.csv")
actual_df_solar = pd.read_csv("data/71shortened/France_pv_actual_test.csv")
forecast_df_solar.generation = forecast_df_solar.generation.apply(
    lambda x: literal_eval(x)
)


solar_data = SourceData.parse_backend(
    5,
    True,
    5,
    5,
    5,
    np.timedelta64("15", "m"),
    generation_forecast=DFBackend(
        forecast_df_solar["generation"],
        forecast_df_solar["timestamps"],
        is_static=False,
        timestep=np.timedelta64(15, "m") / np.timedelta64(15, "m"),
    ),
    generation_actual=DFBackend(
        actual_df_solar["generation"], actual_df_solar["timestamps"]
    ),
)
solar = Solar(
    (15, "m"),
    "Solar",
    5,
    "Solar",
    30000,
    schedule="*/15 * * * *",
    data=solar_data,
)

forecast_df_load = pd.read_csv("data/71shortened/France_load_forecast_test.csv")
actual_df_load = pd.read_csv("data/71shortened/France_load_actual_test.csv")
forecast_df_load.consumption = forecast_df_load.consumption.apply(
    lambda x: literal_eval(x)
)

consumer_data = ConsumerData.parse_backend(
    6,
    True,
    6,
    6,
    np.timedelta64("15", "m"),
    demand_forecast=DFBackend(
        forecast_df_load["consumption"],
        forecast_df_load["timestamps"],
        is_static=False,
        timestep=np.timedelta64(15, "m") / np.timedelta64(15, "m"),
    ),
    demand_actual=DFBackend(
        actual_df_load["consumption"], actual_df_load["timestamps"]
    ),
)
consumer = Consumer(
    (15, "m"),
    "Consumer",
    6,
    "Consumer",
    data=consumer_data,
    schedule="*/15 * * * *",
)
microgrid = Microgrid(
    (15, "m"),
    "Microgrid",
    101,
    "Microgrid",
    schedule="*/15 * * * *",
    sources=[solar],
    consumers=[consumer],
    storage_devices=[battery],
)
contract1 = Contract(microgrid, grid, True)

du = DecisionUnit([contract1])

exp_logger = get_experiment_logger("wandb")

if milp:
    env = MicrogridMILPEnv(
        du,
        np.datetime64("2014-04-12T00:00:00"),
        42,
        callbacks=[],
        action_window=np.timedelta64(1, "h"),
        exp_logger=exp_logger,
    )

    opt = MILPOptimizer(env, solver="ort", obj="max")

    time = np.datetime64("2014-04-12T00:00:00")
    done = False

    i = 0
    while not done:
        obj, rew, time, done = opt.run(time)
        logger.info(f"Obj: {obj} || Rew: {rew}")
        i += 1

        if i == 100:
            break
else:
    env = MicrogridRLEnv(
        du,
        np.datetime64(forecast_df["timestamps"][0]),
        42,
        callbacks=[],
        action_window=np.timedelta64(15, "m"),
        future_window=np.timedelta64(24, "h"),
        exp_logger=exp_logger,
    )
    opt = EnCortexRLOptimizer(
        env, "dqn", "MlpPolicy", 40, target_update_interval=100, verbose=True
    )

    opt.train(100000)
