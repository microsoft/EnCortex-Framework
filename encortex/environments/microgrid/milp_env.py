import logging
import typing as t
from collections import OrderedDict

import numpy as np
from rsome import ro  # isort:skip
from rsome.lp import Vars, Affine  # isort:skip
from pytorch_lightning.loggers import LightningLoggerBase  # isort:skip

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import repeat

import encortex
from encortex.callbacks import EnvCallback
from encortex.decision_unit import DecisionUnit
from encortex.env import EnCortexEnv
from encortex.environments.milp_env import MILPEnvironment
from encortex.grid import Grid
from encortex.microgrid import Microgrid
from encortex.utils.time_utils import tuple_to_np_timedelta
from encortex.optimizers.milp import get_solver

__all__ = ["MicrogridMILPEnv"]

logger = logging.getLogger(__name__)


@EnCortexEnv.register
class MicrogridMILPEnv(MILPEnvironment):
    """Microgrid MILP Environment"""

    def __init__(
        self,
        decision_unit: DecisionUnit,
        start_time: np.datetime64,
        seed: int = None,
        exp_logger: LightningLoggerBase = None,
        callbacks: t.List[EnvCallback] = ...,
        mode: str = "train",
        action_window: np.timedelta64 = np.timedelta64(24, "h"),
        future_window: np.timedelta64 = np.timedelta64(24, "h"),
    ) -> None:
        super().__init__(
            decision_unit, start_time, seed, exp_logger, callbacks, mode
        )

        self.action_window = tuple_to_np_timedelta(action_window)
        self.future_window = tuple_to_np_timedelta(future_window)

        self.microgrids = decision_unit.microgrids
        self.grid = decision_unit.markets[0]

    def _check_decision_unit_constraints(self, decision_unit: DecisionUnit):
        assert (
            len(decision_unit.storage_entities) == 1
        ), "Multiple storage entities not supported"
        assert (
            len(decision_unit.sources) == 1
        ), "Multiple source entities not supported"
        assert (
            len(decision_unit.consumers) == 1
        ), "Multiple source entities not supported"
        assert len(decision_unit.markets) == 1, "Multiple markets not supported"

        assert isinstance(
            decision_unit.markets[0], Grid
        ), "Market of type Grid only supported"

    def _step(
        self, action: t.Dict
    ) -> t.Tuple[t.Any, float, bool, t.Dict[str, t.Any]]:
        # Hack the action - extract battery action, get new variables under new model, equate battery actions and then act.
        model = ro.Model("Objective")
        forecast_state = self.get_state()
        objective, (
            price_buys_forecast,
            price_sells_forecast,
            generations_forecast,
            demands_forecast,
            objectives_forecast,
            grid_vs_forecast,
        ) = self.get_objective_function(
            forecast_state, action, model=model, return_info=True
        )
        self.log_experiment_metrics(f"{self.mode}/objective", objective)

        actual_state = self.get_state(type="actual")
        actions = self.get_action_space(
            self.time, model=model, state=actual_state
        )

        self._equate_storage_actions(action, actions, model)
        actions = self._prune(actions)
        reward = self.get_objective_function(actual_state, actions, model=model)
        model.max(reward.sum())
        model.solve(
            get_solver(encortex.solver),
            display=str(logger.level).upper() in ["DEBUG", "ERROR", "CRITICAL"],
        )

        actions = self.modify(actions)
        reward, (
            price_buys,
            price_sells,
            generations,
            demands,
            objectives,
            grid_vs,
        ) = self.get_objective_function(
            actual_state, actions, model=model, return_info=True
        )

        for idx in range(int(self.action_window / self.timestep)):
            self.log("Price_buy", price_buys[idx])
            self.log("Price_buy_forecast", price_buys_forecast[idx])
            self.log("Price_sell", price_sells[idx])
            self.log("Price_sell_forecast", price_sells_forecast[idx])
            self.log("Generation", generations[idx])
            self.log("Generation Forecast", generations_forecast[idx])
            self.log("Demand", demands[idx])
            self.log("Demand Forecast", demands_forecast[idx])
            self.log("Reward", objectives[idx][0])
            self.log("Reward Forecast", objectives_forecast[idx][0])
            self.log("Grid", grid_vs[idx])
            self.log("Grid Forecast", grid_vs_forecast[idx])

        logger.info(f"Reward: {model.get()}")
        self.log_experiment_metrics(f"{self.mode}/reward", model.get())

        results = self.decision_unit.act(self.time, actions, True)
        penalty = self._get_penalties(results)
        logger.info(f"Penalty: {penalty}")
        self.log_experiment_metrics(f"{self.mode}/penalty", penalty)

        reward = model.get() - penalty
        self.time += self.get_schedule_timestep()
        self.decision_unit.set_time(self.time)
        done = self.is_done(self.time, self.time + self.get_schedule_timestep())
        return reward, self.time, done

    def _prune(self, variables):
        time_idx_max = self.action_window / self.timestep
        for cid in variables.keys():
            time_idx = 0
            times = list(variables[cid].keys())
            for time in times:
                if time_idx >= time_idx_max:
                    variables[cid].pop(time)
                time_idx += 1

        return variables

    def _get_penalties(self, action_results: t.Dict):
        penalty = 0
        for contract_id, penalty_list_by_time in action_results.items():
            for contract_penalty in penalty_list_by_time:
                contractor_penalty, contractee_penalty = contract_penalty
                penalty += sum(contractor_penalty.values()) + sum(
                    contractee_penalty
                )

        return penalty

    def _equate_storage_actions(
        self, decided_actions, new_actions, model: ro.Model
    ):
        for contract in self.decision_unit.contracts:
            for time, entity_action in decided_actions[contract.id].items():
                mg = (
                    contract.contractee
                    if isinstance(contract.contractee, Microgrid)
                    else contract.contractor
                )

                for storage_id, storage_variables in entity_action[mg.id][
                    "all"
                ]["volume"][1]["storage_devices"].items():
                    model.st(
                        storage_variables["volume"][0]
                        == new_actions[contract.id][time][mg.id]["all"][
                            "volume"
                        ][1]["storage_devices"][storage_id]["volume"][0]
                    )
                    for k, v in storage_variables["volume"][1].items():
                        model.st(
                            new_actions[contract.id][time][mg.id]["all"][
                                "volume"
                            ][1]["storage_devices"][storage_id]["volume"][1][k]
                            == v
                        )

    def _apply_constraints(self, variables: t.Dict, model: ro.Model):
        return variables

    def modify(self, variables: t.Dict):
        for cid in variables.keys():
            for time, actions in variables[cid].items():
                storage_id = None
                for entity_id in actions.keys():
                    for contract_id, contract_actions in actions[
                        entity_id
                    ].items():
                        for (
                            contract_action_variable_name,
                            contract_action_variables,
                        ) in contract_actions.items():
                            if isinstance(contract_action_variables, t.List):
                                contract_actions[
                                    contract_action_variable_name
                                ] = self._parse_mod_list(
                                    contract_action_variables
                                )
                            elif isinstance(contract_action_variables, t.Dict):
                                contract_actions[
                                    contract_action_variable_name
                                ] = self._parse_mod_dict(
                                    contract_action_variables
                                )
                            elif isinstance(contract_action_variables, Vars):
                                contract_actions[
                                    contract_action_variable_name
                                ] = self._parse_mod_vars(
                                    contract_action_variables
                                )
                            else:
                                raise NotImplementedError(
                                    "Modify: not supported: ",
                                    contract_action_variables,
                                )

        return variables

    def _parse_mod_list(self, variables: t.List):
        for idx, i in enumerate(variables):
            value = i
            if isinstance(i, t.List):
                value = self._parse_mod_list(i)
            elif isinstance(i, t.Dict):
                value = self._parse_mod_dict(i)
            elif isinstance(i, Vars):
                value = self._parse_mod_vars(i)
            variables[idx] = value
        return variables

    def _parse_mod_dict(self, variables: t.Dict):
        for key, value in variables.items():
            i = value
            if isinstance(i, t.List):
                value = self._parse_mod_list(i)
            elif isinstance(i, t.Dict):
                value = self._parse_mod_dict(i)
            elif isinstance(i, Vars):
                value = self._parse_mod_vars(i)
            variables[key] = value
        return variables

    def _parse_mod_vars(self, variable: Vars):
        return variable.get()

    def _transform(self, state: t.Dict, variables: t.Dict, model: ro.Model):
        eq_variables = OrderedDict()

        for cid, c_actions in variables.items():
            for time, e_actions in c_actions.items():
                if time not in eq_variables.keys():
                    eq_variables[time] = OrderedDict()
                contract = self.decision_unit.get_contract(cid)
                microgrid = (
                    contract.contractor
                    if isinstance(contract.contractor, Microgrid)
                    else contract.contractee
                )
                eq_variables[time][microgrid] = {
                    "volume": e_actions[microgrid.id][cid]["volume"][0]
                }

        return eq_variables

    def get_objective_function(
        self,
        state: t.Dict,
        variables: t.Dict,
        model=None,
        return_info: bool = False,
    ):
        """Objective function as a function of state and variables

        Args:
            state (Dict): Dictionary with attribute as key, numpy array as value
            variables (Dicr): Dictionary with attribute as entity, variable as value

        Returns:
            Affine: Affine expression
        """
        # Reorganize the variables as per time-slot
        time_variables = OrderedDict()

        # variables = self._apply_constraints(state, time_variables, model)
        time_variables = self._transform(state, variables, model)

        objective = np.zeros(1)
        # Merge state and variables into objective
        generations = []
        demands = []
        price_buys = []
        price_sells = []
        objectives = []
        grid_vs = []
        idx = 0
        for t in time_variables.keys():
            microgrid_volumes = sum(
                time_variables[t][microgrid]["volume"]
                for microgrid in self.decision_unit.microgrids
            )
            objective = objective - (
                state[self.grid]["price_buy"][idx]
                * float(self.decision_unit.timestep / np.timedelta64(60, "m"))
                * microgrid_volumes
            )

            if not isinstance(objective, (Affine, Vars)):
                objectives.append(
                    -state[self.grid]["price_buy"][idx]
                    * microgrid_volumes
                    * float(
                        self.decision_unit.timestep / np.timedelta64(60, "m")
                    )
                )
                price_buys.append(state[self.grid]["price_buy"][idx])
                price_sells.append(state[self.grid]["price_sell"][idx])
                generations.append(
                    sum(
                        [
                            list(state[microgrid]["sources"].values())[0][idx]
                            for microgrid in self.microgrids
                        ]
                    )
                )
                grid_vs.append(sum(microgrid_volumes))
                demands.append(
                    sum(
                        [
                            list(state[microgrid]["consumers"].values())[0][
                                "demand"
                            ][idx]
                            for microgrid in self.microgrids
                        ]
                    )
                )

            idx += 1
        # Return the objective and the variables
        if return_info:
            return objective.sum(), (
                price_buys,
                price_sells,
                generations,
                demands,
                objectives,
                grid_vs,
            )
        return objective.sum()  # , variables

    def get_schedule_timestep(self):
        return self.action_window

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        return any(
            grid.is_done(start_time, end_time)
            for grid in self.decision_unit.markets
        )

    def visualize(self, results_folder, optimizers):
        return MGVisualize(results_folder, optimizers)

    def export_config(self):
        config = super().export_config()
        config[f"{self.mode}/action_window"] = self.action_window
        return config

    def _reset(self):
        self.time = self.start_time
        self.decision_unit.set_time(self.start_time)
        return self.get_state()

    def get_action_space(
        self,
        time: np.datetime64,
        model: ro.Model = None,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        if model is None:
            model = ro.Model()
            logger.warn("Creating new model, beware")
        return self.decision_unit.get_schedule(
            time,
            time + self.future_window,
            model,
            apply_constraints,
            state,
        )

    def get_state(
        self, start_time: np.datetime64 = None, type: str = "forecast"
    ):
        if start_time is None:
            start_time = self.time
        return self.decision_unit.get_state(
            start_time, start_time + self.future_window, type=type
        )


class MGVisualize:
    def __init__(self, results_folder: str, optimizers: t.List) -> None:
        self.te_files = OrderedDict()
        for optimizer in optimizers:
            self.te_files[optimizer] = pd.read_csv(
                f"{results_folder}/testdf_{optimizer}.csv"
            )
            for c in self.te_files[optimizer].columns:
                logger.info("Column: ", c)
                self.te_files[optimizer][c] = self.te_files[optimizer][c].apply(
                    lambda x: np.float32(x)
                )

        length_te = 1000000000000
        for data in self.te_files.values():
            if data.shape[0] < length_te:
                length_te = data.shape[0]

        for data in self.te_files.values():
            data = data.iloc[:length_te]

    def show_values(self, axs, orient="v", space=0.01):
        def _single(ax):
            if orient == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                    value = "{:.1f}".format(p.get_height())
                    ax.text(_x, _y, value, ha="center")
            elif orient == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                    value = "{:.1f}".format(p.get_width())
                    ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _single(ax)
        else:
            _single(axs)

    def initial_plots(self, approach):
        approach = list(approach)
        s_price = []

        actual_grid_power = np.asarray(
            self.te_files[approach[0]]["Load_Power_A"]
            - self.te_files[approach[0]]["Solar_Power_A"]
        )
        price_savings_actual_list = np.where(
            np.asarray(actual_grid_power) > 0,
            -1
            * np.asarray(actual_grid_power)
            * np.asarray(self.te_files[approach[0]]["Prices_Buy_A"]),
            -1
            * np.asarray(actual_grid_power)
            * np.asarray(self.te_files[approach[0]]["Prices_Sell_A"]),
        )
        for optimizer in approach:
            s_price.append(self.te_files[optimizer]["Price_savings_A"].sum())
            s_price.append(np.sum(price_savings_actual_list))

        if len(approach) == 1:
            if approach[0] == "MILP":
                s_price = s_price[:2]
            elif approach[0] == "DRL":
                s_price = s_price[2:4]
            elif approach[0] == "SA":
                s_price = s_price[4:6]
            else:
                pass

        savingsdf_price = pd.DataFrame(
            {
                "type": ["batt", "wbatt"] * len(approach),
                "approach": [x for item in approach for x in repeat(item, 2)],
                "Savings": s_price,
            }
        )

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
        z = sns.barplot(
            data=savingsdf_price, x="type", y="Savings", hue="approach", ax=ax1
        )
        self.show_values(z)
        ax1.set_xlabel("Dataset Type")
        fig.tight_layout(pad=5.0)
        fig.suptitle("Total Savings")
        plt.show()

    def plot_results(self, df, start, length, title, approach):
        approach = list(approach)

        if len(approach) > 1:
            approach = ["MILP", "DRL"]

        price_buy_list = list(
            df[0]["Prices_Buy_A"][start * length : (start + 1) * length]
        )
        price_sell_list = list(
            df[0]["Prices_Buy_A"][start * length : (start + 1) * length]
        )

        price_savings = []
        for i, optimizer in enumerate(approach):
            price_savings.append(
                df[i]["Price_savings_A"][start * length : (start + 1) * length]
            )

        fig, axis = plt.subplots(1, figsize=(15, 5))
        title_txt = " \n Savings done during the day: "
        for i, optimizer in enumerate(approach):
            title_txt += f"\n {optimizer} :- PriceSavings = {round(price_savings[i].sum(),2)} EUR"
        axis.set_title(title + title_txt)

        axes = [axis, axis.twinx(), axis.twinx()]
        fig.subplots_adjust(right=0.75)
        axes[-1].spines["right"].set_position(("axes", 1.1))
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)
        soc_colors = ("lime", "darkgreen")
        colors = ("Green", "y", "Red")
        testing_results = [
            "SoC",
            "Actual_Price_Buy (EUR/kWh)",
            "Actual_Price_Sell (EUR/kWh)",
        ]
        lists = [df, price_buy_list, price_sell_list]
        for i, ax, color, testing_result in zip(
            list(range(len(testing_results))), axes, colors, testing_results
        ):
            data = lists[i]
            if i == 0:
                for j in range(len(data)):
                    ax.scatter(
                        list(range(0, length)),
                        data[j]["Current_SOC"][
                            start * length : (start + 1) * length
                        ],
                        color=soc_colors[j],
                        edgecolor="k",
                        s=10,
                    )
                    ax.plot(
                        list(range(0, length)),
                        data[j]["Current_SOC"][
                            start * length : (start + 1) * length
                        ],
                        color=soc_colors[j],
                        label=approach[j],
                    )
                ax.legend()
            else:
                ax.plot(data, color=color)
            ax.set_ylabel("%s" % testing_result, color=color)
            ax.tick_params(axis="y", colors=color)
            ax.tick_params(axis="x")

        axis.set_xlabel("Time")

        plt.show()

    def plot_grid_results(
        self, dfs, start, length, title, approach, power_type
    ):
        df = dfs.copy()
        approach = list(approach)

        if len(approach) > 1:
            approach = ["MILP", "DRL"]

        actual_grid_power = np.asarray(
            self.te_files[approach[0]][f"Load_Power_{power_type[-1]}"]
            - self.te_files[approach[0]][f"Solar_Power_{power_type[-1]}"]
        )

        datafram = pd.DataFrame(columns=[power_type])
        datafram[power_type] = actual_grid_power
        df.append(datafram)

        if len(approach) == 1:
            soc_colors = ("lime", "b")
        else:
            soc_colors = ("lime", "darkgreen", "b")
        approach.append("NoOpt-wbatt")

        price_buy_list = list(
            df[0][f"Prices_Buy_{power_type[-1]}"][
                start * length : (start + 1) * length
            ]
        )
        price_sell_list = list(
            df[0][f"Prices_Buy_{power_type[-1]}"][
                start * length : (start + 1) * length
            ]
        )

        # price_savings = []
        # for i,optimizer in enumerate(approach):
        #     price_savings.append(df[i]["Price_savings_A"][start*length:(start+1)*length])

        fig, axis = plt.subplots(1, figsize=(15, 5))
        axis.set_title(title)

        axes = [axis, axis.twinx(), axis.twinx()]
        fig.subplots_adjust(right=0.75)
        axes[-1].spines["right"].set_position(("axes", 1.1))
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)

        colors = ("Green", "y", "Red")
        testing_results = [
            "Power (kW)",
            "Actual_Price_Buy (EUR/kWh)",
            "Actual_Price_Sell (EUR/kWh)",
        ]
        lists = [df, price_buy_list, price_sell_list]
        for i, ax, color, testing_result in zip(
            list(range(len(testing_results))), axes, colors, testing_results
        ):
            data = lists[i]
            if i == 0:
                for j in range(len(data)):
                    ax.scatter(
                        list(range(0, length)),
                        data[j][power_type][
                            start * length : (start + 1) * length
                        ],
                        color=soc_colors[j],
                        edgecolor="k",
                        s=10,
                    )
                    ax.plot(
                        list(range(0, length)),
                        data[j][power_type][
                            start * length : (start + 1) * length
                        ],
                        color=soc_colors[j],
                        label=approach[j],
                    )
                ax.legend()
            else:
                ax.plot(data, color=color)
            ax.set_ylabel("%s" % testing_result, color=color)
            ax.tick_params(axis="y", colors=color)
            ax.tick_params(axis="x")

        axis.set_xlabel("Time")

        plt.show()

    def plotting_stacked_bars(self, data):
        # Take negative and positive data apart and cumulate
        def get_cumulated_array(datax, **kwargs):
            cum = datax.clip(**kwargs)
            cum = np.cumsum(cum, axis=0)
            d = np.zeros(np.shape(datax))
            d[1:] = cum[:-1]
            return d

        cumulated_data = get_cumulated_array(data, min=0)
        cumulated_data_neg = get_cumulated_array(data, max=0)

        # Re-merge negative and positive data.
        row_mask = data < 0
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data
        return data_stack

    def powermatching(self, df, start, length, title):
        solarpf = list(
            df["Solar_Power_F"][start * length : (start + 1) * length]
        )
        batterypf = list(
            df["Battery_Power_F"][start * length : (start + 1) * length]
        )
        gridpf = list(
            -df["Grid_Power_F"][start * length : (start + 1) * length]
        )
        consumerpf = list(
            df["Load_Power_F"][start * length : (start + 1) * length]
        )
        data_f = np.asarray([solarpf, batterypf, gridpf])
        data_f_shape = np.shape(data_f)
        data_f_stack = self.plotting_stacked_bars(data_f)

        solarpa = list(
            df["Solar_Power_A"][start * length : (start + 1) * length]
        )
        batterypa = list(
            df["Battery_Power_A"][start * length : (start + 1) * length]
        )
        gridpa = list(df["Grid_Power_A"][start * length : (start + 1) * length])
        consumerpa = list(
            df["Load_Power_A"][start * length : (start + 1) * length]
        )
        data_a = np.asarray([solarpa, batterypa, gridpa])
        data_a_shape = np.shape(data_a)
        data_a_stack = self.plotting_stacked_bars(data_a)

        cols = ["orange", "g", "grey"]
        labels = ["Solar_P", "Battery_P", "Grid_P"]
        sns.set(font_scale=0.8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for i in np.arange(0, data_f_shape[0]):
            ax1.bar(
                np.arange(data_f_shape[1]),
                data_f[i],
                bottom=data_f_stack[i],
                color=cols[i],
                label=labels[i],
            )
        ax1.plot(range(len(consumerpf)), consumerpf, label="Consumer_P")
        ax1.set_title("Forecasted Power Matching")

        for i in np.arange(0, data_a_shape[0]):
            ax2.bar(
                np.arange(data_a_shape[1]),
                data_a[i],
                bottom=data_a_stack[i],
                color=cols[i],
                label=labels[i],
            )
        ax2.plot(range(len(consumerpa)), consumerpa, label="Consumer_P")
        ax2.set_title("Actual Power Matching")

        ax1.set_xlabel("Decision Slots")
        ax1.set_ylabel("Power (in kW)")
        ax2.set_xlabel("Decision Slots")
        ax2.set_ylabel("Power (in kW)")

        fig.suptitle(title)

        plt.legend()
        plt.show()
