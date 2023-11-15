# This Early-Access License Agreement (“Agreement”) is an agreement between the party receiving access to the Software Code, as defined below (such party, “you” or “Customer”) and Microsoft Corporation (“Microsoft”). It applies to the Hourly-Matching Solution Accelerator software code, which is Microsoft pre-release beta code or pre-release technology and provided to you at no charge (“Software Code”). IF YOU COMPLY WITH THE TERMS IN THIS AGREEMENT, YOU HAVE THE RIGHTS BELOW. BY USING OR ACCESSING THE SOFTWARE CODE, YOU ACCEPT THIS AGREEMENT AND AGREE TO COMPLY WITH THESE TERMS. IF YOU DO NOT AGREE, DO NOT USE THE SOFTWARE CODE.
#
# 1.	INSTALLATION AND USE RIGHTS.
#    a)	General. Microsoft grants you a nonexclusive, perpetual, royalty-free right to use, copy, and modify the Software Code. You may not redistribute or sublicense the Software Code or any use of it (except to your affiliates and to vendors to perform work on your behalf) through network access, service agreement, lease, rental, or otherwise. Unless applicable law gives you more rights, Microsoft reserves all other rights not expressly granted herein, whether by implication, estoppel or otherwise.
#    b)	Third Party Components. The Software Code may include or reference third party components with separate legal notices or governed by other agreements, as may be described in third party notice file(s) accompanying the Software Code.
#
# 2.	USE RESTRICTIONS. You will not use the Software Code: (i) in a way prohibited by law, regulation, governmental order or decree; (ii) to violate the rights of others; (iii) to try to gain unauthorized access to or disrupt any service, device, data, account or network; (iv) to spam or distribute malware; (v) in a way that could harm Microsoft’s IT systems or impair anyone else’s use of them; (vi) in any application or situation where use of the Software Code could lead to the death or serious bodily injury of any person, or to severe physical or environmental damage; or (vii) to assist or encourage anyone to do any of the above.
#
# 3.	PRE-RELEASE TECHNOLOGY. The Software Code is pre-release technology. It may not operate correctly or at all. Microsoft makes no guarantees that the Software Code will be made into a commercially available product or offering. Customer will exercise its sole discretion in determining whether to use Software Code and is responsible for all controls, quality assurance, legal, regulatory or standards compliance, and other practices associated with its use of the Software Code.
#
# 4.	AZURE SERVICES.  Microsoft Azure Services (“Azure Services”) that the Software Code is deployed to (but not the Software Code itself) shall continue to be governed by the agreement and privacy policies associated with your Microsoft Azure subscription.
#
# 5.	TECHNICAL RESOURCES.  Microsoft may provide you with limited scope, no-cost technical human resources to enable your use and evaluation of the Software Code in connection with its deployment to Azure Services, which will be considered “Professional Services” governed by the Professional Services Terms in the “Notices” section of the Microsoft Product Terms (available at: https://www.microsoft.com/licensing/terms/product/Notices/all) (“Professional Services Terms”). Microsoft is not obligated under this Agreement to provide Professional Services. For the avoidance of doubt, this Agreement applies solely to no-cost technical resources provided in connection with the Software Code and does not apply to any other Microsoft consulting and support services (including paid-for services), which may be provided under separate agreement.
#
# 6.	FEEDBACK. Customer may voluntarily provide Microsoft with suggestions, comments, input and other feedback regarding the Software Code, including with respect to other Microsoft pre-release and commercially available products, services, solutions and technologies that may be used in conjunction with the Software Code (“Feedback”). Feedback may be used, disclosed, and exploited by Microsoft for any purpose without restriction and without obligation of any kind to Customer. Microsoft is not required to implement Feedback.
#
# 7.	REGULATIONS. Customer is responsible for ensuring that its use of the Software Code complies with all applicable laws.
#
# 8.	TERMINATION. Either party may terminate this Agreement for any reason upon (5) business days written notice. The following sections of the Agreement will survive termination: 1-4 and 6-12.
#
# 9.	ENTIRE AGREEMENT. This Agreement is the entire agreement between the parties with respect to the Software Code.
#
# 10.	GOVERNING LAW. Washington state law governs the interpretation of this Agreement. If U.S. federal jurisdiction exists, you and Microsoft consent to exclusive jurisdiction and venue in the federal court in King County, Washington for all disputes heard in court. If not, you and Microsoft consent to exclusive jurisdiction and venue in the Superior Court of King County, Washington for all disputes heard in court.
#
# 11.	DISCLAIMER OF WARRANTY. THE SOFTWARE CODE IS PROVIDED “AS IS” AND CUSTOMER BEARS THE RISK OF USING IT. MICROSOFT GIVES NO EXPRESS WARRANTIES, GUARANTEES, OR CONDITIONS. TO THE EXTENT PERMITTED BY APPLICABLE LAW, MICROSOFT EXCLUDES ALL IMPLIED WARRANTIES, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
#
# 12.	LIMITATION ON AND EXCLUSION OF DAMAGES. IN NO EVENT SHALL MICROSOFT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE BY CUSTOMER, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. TO THE EXTENT PERMITTED BY APPLICABLE LAW, IF YOU HAVE ANY BASIS FOR RECOVERING DAMAGES UNDER THIS AGREEMENT, YOU CAN RECOVER FROM MICROSOFT ONLY DIRECT DAMAGES UP TO U.S. $5,000.
#
# This limitation applies even if Microsoft knew or should have known about the possibility of the damages.

import logging
import typing as t
from collections import OrderedDict
from itertools import repeat

from rsome import ro  # isort:skip
from rsome.lp import Vars, Affine  # isort:skip
import numpy as np  ## isort:skip
from pytorch_lightning.loggers import LightningLoggerBase  ## isort:skip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from encortex.callbacks import EnvCallback
from encortex.decision_unit import DecisionUnit
from encortex.environments.milp_env import MILPEnvironment
from encortex.grid import Grid
from encortex.utils.time_utils import tuple_to_np_timedelta

__all__ = ["BatteryArbitrageMILPEnv"]

logger = logging.getLogger(__name__)


class BatteryArbitrageMILPEnv(MILPEnvironment):
    """Battery Arbitrage Environment"""

    def __init__(
        self,
        decision_unit: DecisionUnit,
        start_time: np.datetime64,
        seed: int = None,
        exp_logger: LightningLoggerBase = None,
        callbacks: t.List[EnvCallback] = ...,
        mode: str = "train",
        action_window: np.timedelta64 = np.timedelta64(24, "h"),
        omega: float = 1.0,
    ) -> None:
        super().__init__(
            decision_unit, start_time, seed, exp_logger, callbacks, mode
        )

        self.action_window = tuple_to_np_timedelta(action_window)
        self.state = None

        assert (
            omega >= 0.0 and omega <= 1.0
        ), f"Omega must be between 0 and 1; Received {omega}"
        self.omega = omega

    def _check_decision_unit_constraints(self, decision_unit: DecisionUnit):
        assert (
            len(decision_unit.storage_entities) >= 1
        ), "Must contain at least 1 storage entity"
        assert (
            len(decision_unit.markets) == 1
        ), "Must only contain a single entity of instance Market"

        assert isinstance(
            decision_unit.markets[0], Grid
        ), "Only market of instance Grid is supported"

    def _get_penalties(self, action_results: t.Dict):
        penalty = 0
        for contract_id, penalty_list_by_time in action_results.items():
            for contract_penalty in penalty_list_by_time:
                contractor_penalty, contractee_penalty = contract_penalty
                penalty += sum(contractor_penalty.values()) + sum(
                    contractee_penalty
                )

        return penalty

    def _step(
        self, action: t.Dict
    ) -> t.Tuple[t.Any, float, bool, t.Dict[str, t.Any]]:
        results = self.decision_unit.act(self.time, action, True)
        penalty = self._get_penalties(results)
        forecasted_state = self.get_state()
        actual_state = self.get_state(type="actual")

        logger.info(f"Penalty: {penalty}")
        self.log_experiment_metrics(f"{self.mode}/penalty", penalty)

        objective, (
            reward_logs_forecast,
            reward_log_carbons_forecast,
            reward_log_costs_forecast,
            grid_carbon_price_forecast,
            grid_price_forecast,
        ) = self.get_objective_function(
            forecasted_state, action, return_info=True, normalize=False
        )

        logger.info(f"Forecast Objective: {objective}")

        reward, (
            reward_logs,
            reward_log_carbons,
            reward_log_costs,
            grid_carbon_price,
            grid_price,
        ) = self.get_objective_function(
            actual_state, action, return_info=True, normalize=False
        )
        reward -= penalty
        for idx in range(len(reward_logs)):
            self.log("Reward", reward_logs[idx])
            self.log_experiment_metrics(f"{self.mode}/Reward", reward_logs[idx])
            self.log("Objective", reward_logs_forecast[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Objective", reward_logs_forecast[idx]
            )
            self.log("Reward Carbon", reward_log_carbons[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Reward Carbon", reward_log_carbons[idx]
            )
            self.log("Reward Carbon Forecast", reward_log_carbons_forecast[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Reward Carbon Forecast",
                reward_log_carbons_forecast[idx],
            )
            self.log("Reward Cost", reward_log_costs[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Reward Cost", reward_log_costs[idx]
            )
            self.log("Reward Cost Forecast", reward_log_costs_forecast[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Reward Cost Forecast",
                reward_log_costs_forecast[idx],
            )
            self.log("Carbon Emission", grid_carbon_price[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Carbon Emission", grid_carbon_price[idx]
            )
            self.log(
                "Carbon Emission Forecast", grid_carbon_price_forecast[idx]
            )
            self.log_experiment_metrics(
                f"{self.mode}/Carbon Emission Forecast",
                grid_carbon_price_forecast[idx],
            )
            self.log("Price", grid_price[idx])
            self.log_experiment_metrics(f"{self.mode}/Price", grid_price[idx])
            self.log("Price Forecast", grid_price_forecast[idx])
            self.log_experiment_metrics(
                f"{self.mode}/Price Forecast", grid_price_forecast[idx]
            )

        self.time += self.get_schedule_timestep()
        self.decision_unit.set_time(self.time)
        done = self.is_done(self.time, self.time + self.get_schedule_timestep())
        logger.info(f"DONE: {done}")
        return reward, self.time, done

    def get_objective_function(
        self,
        state: t.Dict,
        variables: t.Dict,
        model: ro.Model = None,
        return_info: bool = False,
        normalize: bool = True,
    ):
        """Objective function as a function of state and variables

        Args:
            state (Dict): Dictionary with attribute as key, numpy array as value
            variables (Dicr): Dictionary with attribute as entity, variable as value

        Returns:
            Affine: Affine expression
        """
        # Use state and variables to construct the objective function
        grid = self.decision_unit.markets[0]
        grid_price = state[grid]["price"]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(grid_price.reshape(-1, 1))
        grid_price_scaled = scaler.transform(grid_price.reshape(-1, 1)).reshape(
            -1
        )

        scaler = MinMaxScaler(feature_range=(-1, 1))
        grid_carbon_price = state[grid]["emission"]
        scaler = scaler.fit(grid_carbon_price.reshape(-1, 1))
        grid_carbon_price_scaled = scaler.transform(
            grid_carbon_price.reshape(-1, 1)
        ).reshape(-1)

        if normalize:
            price = (
                (self.omega * grid_price_scaled)
                + ((1 - self.omega) * grid_carbon_price_scaled)
            ) * float(
                self.timestep / np.timedelta64(60, "m")
            )  # TODO: Normalize this / Fetch normalized values
        else:
            price = (
                (self.omega * grid_price)
                + ((1 - self.omega) * grid_carbon_price)
            ) * float(self.timestep / np.timedelta64(60, "m"))
        assert len(variables[list(variables.keys())[0]]) == len(
            price
        ), f"{len(variables[list(variables.keys())[0]])} || {len(price)}"

        objective = 0.0
        model = None

        variables_as_times = OrderedDict()

        for cid in variables.keys():
            for time, actions in variables[cid].items():
                storage_id = None
                for k in actions.keys():
                    for sd in self.decision_unit.storage_entities:
                        if k == sd.id:
                            storage_id = k
                            break
                    if storage_id is not None:
                        break

                if time in variables_as_times.keys():
                    variables_as_times[time].append(
                        actions[storage_id]["all"]["volume"][0]
                    )
                else:
                    variables_as_times[time] = [
                        actions[storage_id]["all"]["volume"][0]
                    ]

        i = 0
        objective_carbons = []
        objective_costs = []
        objectives = []
        for time in variables_as_times.keys():
            volume = sum(variables_as_times[time])
            objective += volume * price[i]
            objective_cost = volume * grid_price[i]
            objective_carbon = volume * grid_carbon_price[i]

            if not isinstance(objective_cost, (Affine, Vars)):
                objective_costs.append(sum(objective_cost))
                objective_carbons.append(sum(objective_carbon))
                objectives.append(sum(volume * price[i]))
            i += 1

        assert len(price) == i, f"{len(price)} | {i}"
        assert len(variables_as_times.keys()) == len(
            price
        ), f"{len(price)} | {list(variables_as_times.keys())}"

        for callback in self.callbacks:
            callback.on_step(locals())

        if return_info:
            return objective.sum(), (
                objectives,
                objective_carbons,
                objective_costs,
                grid_carbon_price,
                grid_price,
            )
        return objective.sum()

    def get_schedule_timestep(self):
        return self.action_window

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        return any(
            grid.is_done(start_time, end_time)
            for grid in self.decision_unit.markets
        )

    def export_config(self):
        config = super().export_config()
        config[f"{self.mode}/omega"] = self.omega
        config[f"{self.mode}/action_window"] = self.action_window
        return config

    def visualize(self, results_folder, optimizers):
        return EAVisualize(results_folder, optimizers)


class EAVisualize:
    def __init__(self, results_folder: str, optimizers: t.List) -> None:
        self.tr_files = OrderedDict()
        self.te_files = OrderedDict()
        for optimizer in optimizers:
            self.tr_files[optimizer] = self.rename_columns(
                pd.read_csv(f"{results_folder}/traindf_{optimizer}.csv")
            )
            self.te_files[optimizer] = self.rename_columns(
                pd.read_csv(f"{results_folder}/testdf_{optimizer}.csv")
            )

        length_tr = 1000000000000
        for data in self.tr_files.values():
            if data.shape[0] < length_tr:
                length_tr = data.shape[0]
        length_te = 1000000000000
        for data in self.te_files.values():
            if data.shape[0] < length_te:
                length_te = data.shape[0]

        for data in self.tr_files.values():
            data = data.iloc[:length_tr]

        for data in self.te_files.values():
            data = data.iloc[:length_te]

    def rename_columns(self, data):
        data["Actual Price"] = data["Price_emissions"]
        data["Actual Emission"] = data["Carbon_emissions"]
        data["SOC"] = data["Current_SOC"]
        data["Forecast_Carbon_Savings"] = data["Forecast_Carbon_savings"]
        data["Forecast_Price_Savings"] = data["Forecast_Price_savings"]
        data["Actual_Price_Savings"] = data["Price_savings"]
        data["Actual_Carbon_Savings"] = data["Carbon_savings"]
        return data

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
        s_emissions = []

        for optimizer in approach:
            s_price.append(
                self.te_files[optimizer]["Actual_Price_Savings"].sum()
            )
            s_price.append(
                self.tr_files[optimizer]["Actual_Price_Savings"].sum()
            )
            s_emissions.append(
                self.te_files[optimizer]["Actual_Carbon_Savings"].sum()
            )
            s_emissions.append(
                self.tr_files[optimizer]["Actual_Carbon_Savings"].sum()
            )

        if len(approach) == 1:
            if approach[0] == "MILP":
                s_price = s_price[:2]
                s_emissions = s_emissions[:2]
            elif approach[0] == "DRL":
                s_price = s_price[2:4]
                s_emissions = s_emissions[2:4]
            elif approach[0] == "SA":
                s_price = s_price[4:6]
                s_emissions = s_emissions[4:6]
            else:
                pass

        savingsdf_price = pd.DataFrame(
            {
                "type": ["test", "train"] * len(approach),
                "approach": [x for item in approach for x in repeat(item, 2)],
                "Savings": s_price,
            }
        )

        savingsdf_emission = pd.DataFrame(
            {
                "type": ["test", "train"] * len(approach),
                "approach": [x for item in approach for x in repeat(item, 2)],
                "Savings": s_emissions,
            }
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        z = sns.barplot(
            data=savingsdf_price, x="type", y="Savings", hue="approach", ax=ax1
        )
        self.show_values(z)
        ax1.set_xlabel("Dataset Type")
        ax1.set_ylabel("Price Savings (in $)")
        z = sns.barplot(
            data=savingsdf_emission,
            x="type",
            y="Savings",
            hue="approach",
            ax=ax2,
        )
        self.show_values(z)
        ax2.set_xlabel("Dataset Type")
        ax2.set_ylabel("Carbon Savings (in mTCO2eq)")
        fig.tight_layout(pad=5.0)
        fig.suptitle("Total Savings")
        plt.show()

    def plot_results(self, df, start, length, title, approach):
        approach = list(approach)

        if len(approach) > 1:
            approach = ["MILP", "DRL"]

        # battery_soc_list=list(df["SOC"][start*length:(start+1)*length])
        price_intensity_list = list(
            df[0]["Actual Price"][start * length : (start + 1) * length]
        )
        carbon_intensity_list = list(
            df[0]["Actual Emission"][start * length : (start + 1) * length]
        )

        if len(approach) > 1:
            price_savings_MILP = df[0]["Actual_Price_Savings"][
                start * length : (start + 1) * length
            ]
            carbon_savings_MILP = df[0]["Actual_Carbon_Savings"][
                start * length : (start + 1) * length
            ]
            price_savings_RL = df[1]["Actual_Price_Savings"][
                start * length : (start + 1) * length
            ]
            carbon_savings_RL = df[1]["Actual_Carbon_Savings"][
                start * length : (start + 1) * length
            ]
        else:
            price_savings = df[0]["Actual_Price_Savings"][
                start * length : (start + 1) * length
            ]
            carbon_savings = df[0]["Actual_Carbon_Savings"][
                start * length : (start + 1) * length
            ]

        fig, axis = plt.subplots(1, figsize=(15, 5))
        if len(approach) > 1:
            axis.set_title(
                title
                + f" \n Savings done during the day: \n MILP :- PriceSavings = {round(price_savings_MILP.sum(),2)} EUR, CarbonSavings = {round(carbon_savings_MILP.sum(),2)} gCO2eq \n DRL :- PriceSavings = {round(price_savings_RL.sum(),2)} EUR, CarbonSavings = {round(carbon_savings_RL.sum(),2)} gCO2eq"
            )
        else:
            axis.set_title(
                title
                + f" \n Savings done during the day: \n {approach[0]} :- PriceSavings = {round(price_savings.sum(),2)} EUR, CarbonSavings = {round(carbon_savings.sum(),2)} gCO2eq "
            )

        axes = [axis, axis.twinx(), axis.twinx()]
        fig.subplots_adjust(right=0.75)
        axes[-1].spines["right"].set_position(("axes", 1.1))
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)
        soc_colors = ("lime", "darkgreen")
        colors = ("Green", "y", "Red")
        testing_results = [
            "SoC",
            "Actual_Price($/MWh)",
            "Actual_Carbon(mTCO2eq/MWh)",
        ]
        lists = [df, price_intensity_list, carbon_intensity_list]
        for i, ax, color, testing_result in zip(
            list(range(len(testing_results))), axes, colors, testing_results
        ):
            data = lists[i]
            if i == 0:
                for j in range(len(data)):
                    ax.scatter(
                        list(range(0, length)),
                        data[j]["SOC"][start * length : (start + 1) * length],
                        color=soc_colors[j],
                        edgecolor="k",
                        s=10,
                    )
                    ax.plot(
                        list(range(0, length)),
                        data[j]["SOC"][start * length : (start + 1) * length],
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
        fig.tight_layout()
