from __future__ import annotations
import logging
import typing as t
import numpy as np
import pandas as pd
from rsome import ro
from rsome.lp import Affine

from copy import deepcopy


from encortex.sources.battery import Battery, BatteryAction
from encortex.config import EntityConfig
from encortex.data import SourceData


logger = logging.getLogger(__name__)


class BatteryActionV2(BatteryAction):
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

        (
            action_battery,
            action_value,
            penalty,
            change_in_soc,
            change_in_temperature,
        ) = self._get_valid_actions(entity, action_battery, actions)

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
        action_log["current_temperature"] = entity.cell_temperature
        action_log["power"] = action_value * entity.max_discharging_power

        entity.current_soc -= change_in_soc
        entity.cell_temperature += change_in_temperature
        logger.info(
            f"Battery: Time: {time} Action: {action} New SOC: {entity.current_soc} New Temp: {entity.cell_temperature}"
        )
        entity.current_reference_timestep += entity.timestep

        self._update_action_mask(entity)

        assert (entity.current_soc <= entity.soc_maximum) and (
            entity.current_soc >= entity.soc_minimum
        ), f"Boundary crossed: {entity.current_soc}"

        action_log["penalty"] = penalty
        action_log["new_temperature"] = entity.cell_temperature
        action_log["new_soc"] = entity.current_soc
        self.log_actions(time, action_log)

        return action_value, penalty, None

    def _get_valid_actions(
        self, entity, action_entity: t.Dict, action_value: float
    ) -> t.Tuple[t.Dict, float]:
        # check soc and temperature conditions - idle if either is violating
        if action_value < 0:
            change_in_soc = self._charge(entity, action_value)
        elif action_value > 0:
            change_in_soc = self._discharge(entity, action_value)
        else:
            change_in_soc = 0

        change_in_temperature, _ = self._update_temperature(
            entity, action_entity, action_value, False
        )
        new_soc = entity.current_soc - change_in_soc

        new_temperature = entity.cell_temperature - change_in_temperature
        new_soc = entity.current_soc - change_in_soc

        penalty = {
            "soc_upper_violation": 0,
            "soc_lower_violation": 0,
            "temperature_lower_violation": 0,
            "temperature_upper_violation": 0,
        }

        if (
            new_soc < entity.soc_minimum
            or new_temperature < entity.min_temperature
        ):
            logger.warn(
                "The power is not valid, discharging power (↓) is high."
            )
            new_soc1 = entity.current_soc
            new_temperature1 = entity.cell_temperature
            power = 0
            action_value = 0
            action_entity = {
                "Ct": 0,
                "Dt": 0,
            }
            if new_soc < entity.soc_minimum:
                penalty["soc_lower_violation"] = 1
            else:
                penalty["temperature_lower_violation"] = 1

        elif (
            new_soc > entity.soc_maximum
            or new_temperature > entity.min_temperature
        ):
            logger.warn("The power is not valid, charging power (↑) is high.")
            new_soc1 = entity.current_soc
            new_temperature1 = entity.cell_temperature
            power = 0
            action_entity = {
                "Ct": 0,
                "Dt": 0,
            }
            action_value = 0
            if new_soc > entity.soc_maximum:
                penalty["soc_lower_violation"] = 1
            else:
                penalty["temperature_lower_violation"] = 1
        else:
            new_soc1 = new_soc
            new_temperature1 = new_temperature
            power = action_value

        change_in_soc = entity.current_soc - new_soc1
        change_in_temperature = new_temperature1 - entity.cell_temperature

        return (
            action_entity,
            action_value,
            penalty,
            change_in_soc,
            change_in_temperature,
        )

    def batch_apply_constraints(
        self,
        variables: t.Dict,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        soc = deepcopy(self.entity.current_soc)
        cell_temperature = deepcopy(self.entity.cell_temperature)
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

            change_in_temperature, _ = self._update_temperature(
                self.entity, Dt - Ct, self.entity.max_charging_power, False
            )

            new_soc = soc - change_in_soc

            if apply_constraints:
                model.st(new_soc >= self.entity.soc_minimum)
                model.st(new_soc <= self.entity.soc_maximum)
                model.st(
                    cell_temperature + change_in_temperature
                    <= self.entity.max_temperature
                )
                model.st(
                    cell_temperature + change_in_temperature
                    >= self.entity.min_temperature
                )

                if (
                    pd.Timestamp(action_time + self.entity.timestep).day
                    - pd.Timestamp(action_time).day
                ) == 1:
                    model.st(new_soc >= self.entity.soc_minimum)

            soc = new_soc
            cell_temperature += change_in_temperature

    def _update_temperature(
        self,
        entity: BatteryV2,
        action: float,
        action_value: float,
        update_temperature: bool = True,
    ):
        current = self._get_current(entity, action)
        q_p = (
            (current**2)
            * entity.internal_impedance
            * (
                (np.float32(entity.timestep.astype("timedelta64[m]")) * 60)
                / entity.c_cell
            )
        )
        q_s = (
            (entity.cell_temperature * entity.delta_s * current)
            / entity.faradays_constant
            * (
                (np.float32(entity.timestep.astype("timedelta64[m]")) * 60)
                / entity.c_cell
            )
        )
        q_b = (
            (entity.cell_temperature - entity.ambient_temperature)
            * entity.A
            * entity.h
            * (
                (np.float32(entity.timestep.astype("timedelta64[m]")) * 60)
                / entity.c_cell
            )
        )

        change_in_temperature = q_p + q_s - q_b

        if update_temperature:
            if (
                entity.cell_temperature + change_in_temperature
                > entity.max_temperature
                or entity.cell_temperature + change_in_temperature
                < entity.min_temperature
            ):
                action = 0
            else:
                entity.cell_temperature += change_in_temperature

            return change_in_temperature, action

        return change_in_temperature, None

    def _get_current(self, entity: BatteryV2, action: float) -> float:
        action = action["Dt"] - action["Ct"]
        if action == 0:
            return 0
        elif action < 0:  # charge
            return entity.max_charge_current * action
        else:
            return entity.max_discharge_current * action


class BatteryV2(Battery):
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
        c_cell: float,
        internal_impedance: float,
        ambient_temperature: float,
        cell_temperature: float,
        max_temperature: float,
        min_temperature: float,
        faradays_constant: float,
        A: float,
        h: float,
        delta_s: float,
        max_charge_current: float,
        max_discharge_current: float,
        test_flag: bool = False,
        action: BatteryAction = BatteryActionV2("Bv2", "Bv2"),
        config: EntityConfig = None,
        data: SourceData = None,
        schedule: t.Dict = None,
    ) -> None:
        super().__init__(
            timestep,
            name,
            id,
            description,
            storage_capacity,
            charging_efficiency,
            discharging_efficiency,
            soc_initial,
            depth_of_discharge,
            soc_minimum,
            degradation_flag,
            min_battery_capacity_factor,
            battery_cost_per_kWh,
            reduction_coefficient,
            degradation_period,
            test_flag,
            action,
            config,
            data,
            schedule,
        )

        self.internal_impedance = internal_impedance
        self.ambient_temperature = ambient_temperature
        self.faradays_constant = faradays_constant
        self.delta_s = delta_s
        self.A = A
        self.h = h
        self.max_charge_current = max_charge_current
        self.max_discharge_current = max_discharge_current
        self.cell_temperature = cell_temperature
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.c_cell = c_cell
