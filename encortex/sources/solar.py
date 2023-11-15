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

from __future__ import annotations

import typing as t
from collections import OrderedDict

import gym
import numpy as np
from gym import spaces
from rsome import ro

from encortex.action import Action as EntityAction
from encortex.config import EntityConfig
from encortex.data import Data, SourceData

EntityData = Data
from encortex.source import Source
from encortex.visualize import plot_source_data


class SolarCapacityChange(EntityAction):
    def __init__(
        self,
        name: str = "SolarCapacityChange",
        description: str = "Changes maximum capacity",
        action: gym.Space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        ),
        timestep: np.timedelta64 = None,
    ):
        super().__init__(name, description, action, timestep)

    def _checks(self, action: np.ndarray, *args, **kwargs):
        super()._checks(action, *args, **kwargs)
        assert action >= 0.0 and action <= 1.0

    def __call__(
        self,
        time: np.ndarray,
        action: np.ndarray,
        entity,
        *args,
        **kwargs: np.ndarray,
    ):
        if not entity.disable_action:
            entity.current_max_capacity = action * entity.max_capacity
        self.log_actions(time, {"supplied": action["volume"][0][0]})

        return action["volume"][0][0], {}, None

        # def get_action_variable(
        #     self, model: ro.Model, time: np.datetime64, **kwargs
        # ) -> t.Dict:
        def get_action_variable(
            self,
            model: ro.Model,
            time: np.datetime64,
            apply_constraints: bool = True,
            cid: int = 1000,
            **kwargs,
        ) -> t.Dict:
            kwargs_str = ""
            for k, v in kwargs.items():
                kwargs_str += f"{k}_{v}_"
            return {
                "volume": [
                    model.dvar(
                        (1,), "C", f"{self.id}_{str(time)}_volume_{kwargs}"
                    )
                ],
                # "volume_ground": [
                #     model.dvar((1,), "C", f"{self.id}_{str(time)}_volume_{kwargs}")
                # ],
            }

    def get_action_variables(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        start_time = np.datetime64(start_time).astype("datetime64[m]")
        end_time = np.datetime64(end_time).astype("datetime64[m]")

        current_time = start_time
        variables = OrderedDict()
        contract_variables = OrderedDict()
        while current_time < end_time:

            action_variable = self.get_action_variable(model, current_time)

            variables[current_time] = action_variable
            current_time = current_time + self.entity.timestep

        for contract in self.entity.contracts:

            contract_variables[contract.id] = OrderedDict()
            current_time = start_time
            while current_time < end_time:

                contract_variables[contract.id][
                    current_time
                ] = self.get_action_variable(
                    model, current_time  # , contract_id=str(contract.id)
                )

                current_time += self.entity.timestep

        for t in variables.keys():

            v_sum_t = contract_variables[self.entity.contracts[0].id][t][
                "volume"
            ][0]
            for c in self.entity.contracts[1:]:
                v_sum_t += contract_variables[c.id][t]["volume"][0]

            model.st(
                v_sum_t
                == variables[t]["volume"][
                    0
                ]  # + variables[t]["volume_ground"][0]
            )

        # if len(self.entity.contracts) > 1:
        contract_variables["all"] = variables
        return contract_variables


@Source.register
class Solar(Source):
    data: SourceData
    max_capacity: float
    current_max_capacity: float
    actions: SolarCapacityChange

    def __init__(
        self,
        timestep: np.timedelta64,
        name: str,
        id: int,
        description: str,
        max_capacity: float,
        action: EntityAction = None,
        config: EntityConfig = None,
        data: SourceData = None,
        schedule: t.Dict = None,
        disable_action: bool = True,
    ) -> None:
        self.disable_action = disable_action
        if action is None:
            action = SolarCapacityChange()
        super().__init__(
            timestep, name, id, description, action, config, data, schedule
        )

        self.config["max_capacity"] = max_capacity
        self.current_max_capacity = max_capacity

    def get_action(self):
        return self.actions.action

    def get_state(
        self,
        start_time,
        end_time,
        vectorize: bool = False,
        type: str = "forecast",
    ):
        if type == "forecast":
            data = self.data.generation_forecast[start_time:end_time]
        elif type == "actual":
            data = self.data.generation_actual[start_time:end_time]
        else:
            raise NotImplementedError(f"{type} not supported")
        if vectorize:
            # assert False, f"{np.asarray(data.generation).reshape(-1).shape}"
            data = np.asarray(data).reshape(-1)

        return data

    def step(self, timestamp):
        self.timestamp = timestamp
        self.current_forecast_generation = self.generation_forecast.query(
            timestamp, timestamp
        )

    # def act(self, action):
    #     if self.disable_action:
    #         pass
    #     else:
    #         return self.actions[0](action, self)
    def act(self, times: np.ndarray, actions: np.ndarray, train_flag: bool):
        ret = self.action(times, actions, self)
        return ret

    def visualize(
        self,
        is_forecast: bool,
        start_time: np.datetime64 = None,
        end_time: np.datetime64 = None,
    ):
        data = (
            self.generation_forecast if is_forecast else self.generation_actual
        )
        plot_source_data(list(data), start_time, end_time)

    @staticmethod
    def parse_data(
        timestep: np.timedelta64,
        name: str,
        id: int,
        description: str,
        max_capacity,
        forecast_data_filename,
        actual_data_filename,
        in_memory: bool,
    ) -> Solar:
        generation_forecast = None
        generation_actual = None
        if forecast_data_filename is not None:
            generation_forecast = SourceData.parse_csv(
                forecast_data_filename, id, in_memory=in_memory
            )
            generation_actual = SourceData.parse_csv(
                actual_data_filename, id, in_memory=in_memory
            )

        return Solar(
            timestep,
            name,
            id,
            description,
            max_capacity=max_capacity,
            forecast_data=generation_forecast,
            actual_data=generation_actual,
        )

    def __repr__(self) -> str:
        return super().__repr__() + f"\n id: {self.id}"

    def is_schedule_uniform(self):
        return True

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        if start_time is None or end_time is None:
            return (
                self.time > self.data.generation_forecast.get_end_of_data_time()
            )
        else:
            if start_time is None:
                start_time = end_time
            if end_time is None:
                end_time = start_time
            return (
                start_time
                > self.data.generation_forecast.get_end_of_data_time()
            ) and (
                end_time > self.data.generation_forecast.get_end_of_data_time()
            )
