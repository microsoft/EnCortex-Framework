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
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
from rsome import ro
from rsome.lp import Affine

from encortex.action import Action
from encortex.config import EntityConfig
from encortex.data import SourceData
from encortex.source import Source
from encortex.sources.storage import Storage
from encortex.utils.transform import vectorize_dict

Data = SourceData

import copy
from collections import deque

logger = logging.getLogger(__name__)


class BatteryAction(Action):
    """Battery Action Behaviour"""

    def __init__(
        self,
        name: str,
        description: str,
        action: gym.Space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        ),
        timestep: np.timedelta64 = None,
        disable_degradation=True,
        # action_mask: t.List = [-3.3, 0, 3.3],
    ):
        super().__init__(name, description, action, timestep)
        self.disable_degradation = disable_degradation
        # self.action_mask = action_mask

    def _checks(self, action: np.ndarray, *args, **kwargs):
        super()._checks(action, *args, **kwargs)
        # assert action >= -1.0 and action <= 1.0

    def __call__(
        self,
        time: np.datetime64,
        action: t.Dict,
        entity,
        train_flag,
        *args,
        **kwargs: np.ndarray,
    ):
        entity: Battery
        times = [time]

        try:
            action = action["all"]
        except:
            pass
        actions = (
            action["volume"][1]["Dt"] - action["volume"][1]["Ct"]
        ) * self.entity.max_discharging_power
        action_battery = {
            "Ct": action["volume"][1]["Ct"],
            "Dt": action["volume"][1]["Dt"],
        }

        super().__call__(time, actions, entity, *args, **kwargs)

        (
            action_value,
            actions,
            new_soc,
            change_in_soc,
            power,
            penalty,
        ) = self._update_state_of_charge(entity, actions, action_battery)

        if not self.disable_degradation and not isinstance(actions, Affine):
            if entity.num_steps % entity.degradation_period == 0:
                self._degradation_model(entity)

            if actions != 0:
                self._capacity_update(entity)

            entity.battery_capacity_variation.append(
                [entity.current_storage_capacity, actions]
            )
            entity.history.append(entity.battery_capacity_variation)

        action_log = {"action": action_value}
        action_log["current_soc"] = entity.current_soc
        action_log["power"] = action_value * entity.max_discharging_power

        entity.current_soc -= change_in_soc
        logger.info(
            f"Battery: Time: {time} Action: {action} New SOC: {entity.current_soc}"
        )
        entity.current_reference_timestep += entity.timestep

        self._update_action_mask(entity)

        assert (entity.current_soc <= entity.soc_maximum) and (
            entity.current_soc >= entity.soc_minimum
        ), f"Boundary crossed: {entity.current_soc}"

        action_log["penalty"] = penalty
        action_log["new_soc"] = entity.current_soc
        self.log_actions(time, action_log)

        return action_value, penalty, None

    def _charging_action_power_invalid(self, entity, new_soc):
        if new_soc > entity.soc_maximum:
            # logging.warn("Invalid Charging Action")
            return True
        else:
            return False

    def _discharging_action_power_invalid(self, entity, new_soc):
        if new_soc < entity.soc_minimum:
            # logging.warn("Invalid Discharging Action")
            return True
        else:
            return False

    def _update_state_of_charge(self, entity, action_value, action: t.Dict):
        if action_value < 0:
            change_in_soc = self._charge(entity, action_value)
        elif action_value > 0:
            change_in_soc = self._discharge(entity, action_value)
        else:
            change_in_soc = 0

        new_soc = entity.current_soc - change_in_soc
        if new_soc < entity.soc_minimum:
            logger.warn(
                "The power is not valid, discharging power (↓) is high."
            )
            new_soc1 = entity.current_soc
            power = 0
            action_value = 0
            penalty = {"soc_upper_violation": 0, "soc_lower_violation": 1}
        elif new_soc > entity.soc_maximum:
            logger.warn("The power is not valid, charging power (↑) is high.")
            new_soc1 = entity.current_soc
            power = 0
            action_value = 0
            penalty = {"soc_upper_violation": 1, "soc_lower_violation": 0}
        else:
            new_soc1 = new_soc
            power = action_value
            penalty = {"soc_upper_violation": 0, "soc_lower_violation": 0}

        change_in_soc = entity.current_soc - new_soc1

        entity.current_cycle_discharge += (
            0 if change_in_soc > 0 else -change_in_soc
        )
        entity.net_power_discharge += 0 if change_in_soc > 0 else -change_in_soc

        return action_value, action, new_soc1, change_in_soc, power, penalty

    def _charge(self, entity, action_value, current_storage_capacity=None):
        if current_storage_capacity is None:
            current_storage_capacity = entity.current_storage_capacity
        return (
            entity.charging_efficiency
            * float(entity.timestep / np.timedelta64("60", "m"))
            * float(action_value)
            * float(1.0 / current_storage_capacity)
        )

    def _discharge(self, entity, action_value, current_storage_capacity=None):
        if current_storage_capacity is None:
            current_storage_capacity = entity.current_storage_capacity
        return (
            float(entity.timestep / np.timedelta64("60", "m"))
            * float(action_value)
            * float(
                1.0 / (current_storage_capacity * entity.discharging_efficiency)
            )
        )

    def _capacity_update(self, entity):
        while (
            entity.current_cycle_discharge >= entity.depth_of_discharge * 0.01
        ):
            entity.cycles += 1
            entity.previous_storage_capacity = entity.current_storage_capacity
            entity.current_storage_capacity *= entity.reduction_coefficient
            entity.current_cycle_discharge -= entity.depth_of_discharge * 0.01

    def _degradation_model(self, entity):
        capacity_difference = entity.history[0][0] - entity.history[-1][0]
        net_discharge = sum(
            abs(entity.history[timestep][1])
            * (entity.timestep / np.timedelta64("60", "m"))
            for timestep in range(len(entity.history))
        )
        if net_discharge != 0:
            entity.degradation_coefficient = (
                capacity_difference / net_discharge
            ) * entity.battery_cost_per_kWh

    def _update_action_mask(self, entity):
        entity.max_charging_power = self._max_power(
            entity, -1, entity.charging_efficiency
        )
        entity.max_discharging_power = self._max_power(
            entity, 1, entity.discharging_efficiency
        )

    def _max_power(self, entity, power, efficiency):
        if power < 0:
            power_grid = entity.max_charging_power
        else:
            power_grid = entity.max_discharging_power
        return power_grid

    def get_action_variable(
        self,
        model: ro.Model,
        time: np.datetime64,
        apply_constraints: bool = True,
        cid: int = 1000,
        **kwargs,
    ) -> t.Dict:
        Ct = model.dvar(
            (1,), "B", name=f"battery_ct_{time}"
        )  # TODO: Action variable Ct and Dt
        Dt = model.dvar((1,), "B", name=f"battery_dt_{time}")
        volume = model.dvar((1,), "C", f"volume_{time}_{self.entity.id}")
        if apply_constraints:
            model.st(Ct + Dt <= 1)

        return {"volume": [(volume), {"Ct": Ct, "Dt": Dt}]}

    def batch_apply_constraints(
        self,
        variables: t.Dict,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        soc = deepcopy(self.entity.current_soc)
        storage_capacity = deepcopy(self.entity.current_storage_capacity)
        action_times = list(sorted(list(variables.keys())))

        for action_time in action_times:
            logger.info(
                f"Action time: {action_time} | Variable: {variables[action_time]} | SOC: {soc}"
            )
            action = variables[action_time]
            Ct = action["volume"][1]["Ct"]
            Dt = action["volume"][1]["Dt"]
            volume_t = action["volume"][0]

            change_in_soc = Ct * self._charge(
                self.entity, self.entity.max_charging_power, storage_capacity
            ) + Dt * self._discharge(
                self.entity, self.entity.max_discharging_power, storage_capacity
            )
            model.st(
                volume_t
                == (
                    Ct * self.entity.max_charging_power
                    + Dt * self.entity.max_discharging_power
                )
            )

            new_soc = soc - change_in_soc

            if apply_constraints:
                model.st(new_soc >= self.entity.soc_minimum)
                model.st(new_soc <= self.entity.soc_maximum)

                if (
                    pd.Timestamp(action_time + self.entity.timestep).day
                    - pd.Timestamp(action_time).day
                ) == 1:
                    model.st(new_soc >= self.entity.soc_minimum)

            soc = new_soc

    def transform(self, action_type: str, action: int):
        if action_type == "rl":
            if action == 0:
                Ct = 1
                Dt = 0
            elif action == 1:
                Ct = 0
                Dt = 0
            elif action == 2:
                Ct = 0
                Dt = 1
            else:
                raise NotImplementedError(f"Unknown action: {action} passed")
            return {"Ct": Ct, "Dt": Dt}
        else:
            raise NotImplementedError(f"Type: {action_type} not implemented")

    def transform_variables(
        self, action_type: str, action, model: ro.Model, variables: t.Dict
    ):
        if action_type == "rl":
            if action == 0:
                model.st(variables["Ct"] == 1)
                model.st(variables["Dt"] == 0)
            elif action == 1:
                model.st(variables["Ct"] == 0)
                model.st(variables["Dt"] == 0)
            elif action == 2:
                model.st(variables["Ct"] == 0)
                model.st(variables["Dt"] == 1)
            else:
                raise NotImplementedError(f"Unknown action: {action} passed")
            return variables
        else:
            raise NotImplementedError

    def get_idle_action(self, action_type: str = "rl", **kwargs):
        if action_type == "rl":
            return 1
        elif action_type == "milp":
            model = kwargs["model"]
            Ct = kwargs["Ct"]
            Dt = kwargs["Dt"]
            model.st(Ct == Dt)
            model.st(Ct == 0)
            return kwargs

    def get_action_mask(self):
        change_in_soc = self._charge(
            self.entity,
            self.entity.max_charging_power,
            self.entity.storage_capacity,
        ) + self._discharge(
            self.entity,
            self.entity.max_discharging_power,
            self.entity.storage_capacity,
        )
        mask = np.asarray([0, 0, 0])
        if self.entity.current_soc - change_in_soc >= self.entity.soc_maximum:
            mask[0] = 1
        elif self.entity.current_soc - change_in_soc <= self.entity.soc_minimum:
            mask[2] = 0
        return mask


@Source.register
class Battery(Storage):
    """Battery entity as a storage device."""

    def __init__(
        self,
        timestep: np.timedelta64,
        name: str,
        id: int,
        description: str,
        storage_capacity: float,
        charging_efficiency: float,
        discharging_efficiency: float,
        soc_initial: float,
        depth_of_discharge: float,
        soc_minimum: float,
        degradation_flag: bool,
        min_battery_capacity_factor: float,
        battery_cost_per_kWh: float,
        reduction_coefficient: float,
        degradation_period: int,
        test_flag: bool = False,
        action: BatteryAction = BatteryAction("Battery", "Description"),
        config: EntityConfig = None,
        data: SourceData = None,
        schedule: t.Dict = None,
    ) -> None:
        super().__init__(
            timestep, name, id, description, action, config, data, schedule
        )

        self.storage_capacity = storage_capacity
        self.current_storage_capacity = storage_capacity
        self.previous_storage_capacity = storage_capacity
        self.charging_efficiency = charging_efficiency
        self.discharging_efficiency = discharging_efficiency
        self.current_charging_efficiency = charging_efficiency
        self.current_discharging_efficiency = discharging_efficiency
        self.soc_initial = soc_initial
        self.depth_of_discharge = depth_of_discharge
        self.soc_minimum = soc_minimum

        self.degradation_flag = degradation_flag
        self.degradation_period = degradation_period
        self.reduction_coefficient = reduction_coefficient
        self.min_battery_capacity_factor = min_battery_capacity_factor
        self.battery_cost_per_kWh = battery_cost_per_kWh / (
            1.0 - self.min_battery_capacity_factor
        )

        self._set_max_charging_power()  # negative value
        self._set_max_discharging_power()  # positive value

        self.soc_maximum = 1.0
        self.degradation_coefficient = 0
        self.current_cycle_discharge = 0
        self.net_power_discharge = 0
        self.cycles = 0

        self.num_steps = 0
        self.start_time = None

        self.test = test_flag
        if self.test:
            self.current_soc = copy.deepcopy(self.soc_initial)
        else:
            self.current_soc = self._initial_soc()

        self.inf_soc = self.current_soc

        if action is not None:
            self.action = action

        self.config["storage_capacity"] = storage_capacity
        self.config["current_storage_capacity"] = storage_capacity
        self.config["previous_storage_capacity"] = storage_capacity
        self.config["charging_efficiency"] = charging_efficiency
        self.config["discharging_efficiency"] = discharging_efficiency
        self.config["current_charging_efficiency"] = charging_efficiency
        self.config["current_discharging_efficiency"] = discharging_efficiency
        self.config["soc_initial"] = soc_initial
        self.config["depth_of_discharge"] = depth_of_discharge
        self.config["soc_minimum"] = soc_minimum
        self.config["degradation_flag"] = self.degradation_flag
        self.config["degradation_period"] = self.degradation_period
        self.config["reduction_coefficient"] = self.reduction_coefficient
        self.config[
            "min_battery_capacity_factor"
        ] = self.min_battery_capacity_factor
        self.config["battery_cost_per_kWh"] = self.battery_cost_per_kWh

    def reset(self):
        self.num_steps = 0
        self.cycles = 0
        self.current_cycle_discharge = 0
        self.degradation_coefficient = 0
        self.net_power_discharge = 0
        self.current_storage_capacity = self.storage_capacity

        # in this formulation efficiencies are fixed but framework has the capability to modify them if required
        self.current_charging_efficiency = self.charging_efficiency
        self.current_discharging_efficiency = self.discharging_efficiency

        if self.test:
            self.current_soc = copy.deepcopy(self.soc_initial)
        else:
            self.current_soc = self._initial_soc()

        self.battery_capacity_variation = []

        # history maintained for calculating degradation rate
        # TODO: Centralize history maintenance with a DataBackend that uses a fast tensor storage format.
        self.history = deque(
            maxlen=(
                int(
                    self.degradation_period
                    * int(np.timedelta64("60", "m") / self.timestep)
                )
            )
        )

        # maps action to power
        self.action._update_action_mask(self)
        self._set_max_charging_power()  # negative value of self.max_charging_power
        self._set_max_discharging_power()  # positive value of self.max_discharging_power

        self.time = self.start_time

        return self.current_soc

    def _set_max_charging_power(self):
        self.max_charging_power = np.float32(
            -1.0 * np.round(np.array(self.storage_capacity) / 3, 2)
        )

    def _set_max_discharging_power(self):
        self.max_discharging_power = np.float32(
            np.round(np.array(self.storage_capacity) / 3, 2)
        )

    def _initial_soc(self):
        # initial_randomized_soc = self.soc_minimum + np.random.rand() * (
        #     self.soc_maximum - self.soc_minimum
        # )
        initial_randomized_soc = copy.deepcopy(self.soc_initial)

        return initial_randomized_soc

    def get_state(
        self,
        start_time,
        end_time,
        vectorize: bool = False,
        type: str = "forecast",
    ):
        state = {"current_soc": self.current_soc}
        if vectorize:
            state = vectorize_dict(state)
        return state

    def get_action_space(self):
        return self.action.action_space()

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        if (
            self.current_storage_capacity
            < self.min_battery_capacity_factor * self.storage_capacity
        ):
            logging.info(
                f"Battery capacity is reduced to 80%, will have to reset the battery {self.id} soon, since not at its optimal state"
            )
            return True
        else:
            return False

    def act(
        self,
        times: t.List[np.datetime64],
        actions: t.List[t.Dict],
        train_flag: bool,
    ):  # action : {"Ct": ... , "Dt": ....}
        # required since its called from decision_unit.step
        self.current_reference_timestep = times
        ret = self.action(times, actions, self, False)
        self.num_steps += 1
        return ret

    def visualize(
        self, type: str, start_time: np.datetime64, end_time: np.datetime64
    ):
        raise NotImplementedError

    def is_schedule_uniform(self):
        return True

    def get_action_mask(self):
        return self.action.get_action_mask()
