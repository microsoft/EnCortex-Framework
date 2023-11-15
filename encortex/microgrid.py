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

import logging
import typing as t
from collections import OrderedDict

import gym
import numpy as np
from rsome import ro

from encortex.action import Action as EntityAction
from encortex.config import EntityConfig
from encortex.data import Data as EntityData
from encortex.entity import Entity
from encortex.sources.storage import Storage
from encortex.utils.transform import vectorize_dict


logger = logging.getLogger(__name__)


class MicrogridAction(EntityAction):
    def __init__(
        self,
        name: str = "Microgrid",
        description: str = "Microgrid",
        action: gym.Space = None,
        timestep: np.timedelta64 = None,
    ):
        super().__init__(name, description, action, timestep)

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
        variable = {
            "volume": [
                model.dvar(
                    (1,),
                    "C",
                    f"microgrid_{self.entity.id}_{str(time)}_volume_{kwargs}",
                )
            ],
        }

        microgrid_variables = self._get_subaction_variables(
            model, time, apply_constraints, cid, **kwargs
        )

        source_volumes = sum(
            source_actions["volume"][0]
            for source_actions in list(microgrid_variables["sources"].values())
        )
        consumer_volumes = sum(
            consumer_actions["volume"][0]
            for consumer_actions in list(
                microgrid_variables["consumers"].values()
            )
        )
        storage_device_volumes = sum(
            storage_device_actions["volume"][0]
            for storage_device_actions in list(
                microgrid_variables["storage_devices"].values()
            )
        )

        model.st(
            variable["volume"][0]
            == (source_volumes - consumer_volumes + storage_device_volumes)
        )

        variable["volume"].append(microgrid_variables)
        return variable

    def _get_subaction_variables(
        self,
        model: ro.Model,
        time: np.datetime64,
        apply_constraints: bool = True,
        cid: int = 1000,
        **kwargs,
    ) -> t.Dict:
        variables = OrderedDict()
        variables["sources"] = OrderedDict()
        variables["consumers"] = OrderedDict()
        variables["storage_devices"] = OrderedDict()

        for source in self.entity.sources:
            variables["sources"][source] = source.action.get_action_variable(
                model, time, apply_constraints, cid, **kwargs
            )

        for consumer in self.entity.consumers:
            variables["consumers"][
                consumer
            ] = consumer.action.get_action_variable(
                model, time, apply_constraints, cid, **kwargs
            )

        for storage_device in self.entity.storage_devices:
            variables["storage_devices"][
                storage_device
            ] = storage_device.action.get_action_variable(
                model, time, apply_constraints, cid, **kwargs
            )

        return variables

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

        self.batch_apply_constraints(variables, model, apply_constraints, state)

        for contract in self.entity.contracts:

            contract_variables[contract.id] = OrderedDict()
            current_time = start_time
            while current_time < end_time:

                contract_variables[contract.id][current_time] = {
                    "volume": [
                        model.dvar(
                            (1,),
                            "C",
                            name=f"contract_{contract.id}_volume_time_{current_time}",
                        )
                    ]
                }  # self.get_action_variable(

                current_time += self.entity.timestep

        for t in variables.keys():

            v_sum_t = contract_variables[self.entity.contracts[0].id][t][
                "volume"
            ][0]
            for c in self.entity.contracts[1:]:
                v_sum_t += contract_variables[c.id][t]["volume"][0]

            model.st(v_sum_t == variables[t]["volume"][0])

        # if len(self.entity.contracts) > 1:
        contract_variables["all"] = variables
        return contract_variables

    def batch_apply_constraints(
        self,
        variables: t.Dict,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        storage_actions = OrderedDict()
        for time, action_variable in variables.items():
            for storage_device, storage_device_actions in action_variable[
                "volume"
            ][1]["storage_devices"].items():
                if storage_device not in storage_actions.keys():
                    storage_actions[storage_device] = OrderedDict()
                storage_actions[storage_device][time] = storage_device_actions

        for storage_device in self.entity.storage_devices:
            storage_device.action.batch_apply_constraints(
                storage_actions[storage_device], model, apply_constraints
            )

        if state is not None:
            idx = 0
            for time, action_variable in variables.items():
                for source, source_actions in action_variable["volume"][1][
                    "sources"
                ].items():
                    model.st(
                        state[self.entity]["sources"][source][idx]
                        == source_actions["volume"][0]
                    )
                idx += 1

            idx = 0
            for time, action_variable in variables.items():
                for consumer, consumer_actions in action_variable["volume"][1][
                    "consumers"
                ].items():
                    model.st(
                        state[self.entity]["consumers"][consumer]["demand"][idx]
                        == consumer_actions["volume"][0]
                    )
                idx += 1

    def __call__(
        self,
        time: np.ndarray,
        action: t.Dict,
        entity,
        *args,
        **kwargs: np.ndarray,
    ):
        super().__call__(time, action, entity, *args, **kwargs)

        logger.info(f"Action: {action} @ Time: {time}")
        microgrid_action = action["all"]["volume"]

        penalties = {}
        infos = []
        for source, source_action in microgrid_action[1]["sources"].items():
            # source = self.entity.get_entity(source)
            source_action, source_penalty, source_info = source.act(
                time, source_action, True
            )
            for k, v in source_penalty.items():
                penalties[f"{source.id}/{k}"] = v

            infos.append(source_info)
            microgrid_action[1]["sources"][source] = source_action

        for consumer, consumer_action in microgrid_action[1][
            "consumers"
        ].items():
            # consumer = self.entity.get_entity(consumer)
            consumer_action, consumer_penalty, consumer_info = consumer.act(
                time, consumer_action, True
            )
            infos.append(consumer_info)

            for k, v in consumer_penalty.items():
                penalties[f"{consumer.id}/{k}"] = v

            microgrid_action[1]["consumers"][consumer] = consumer_action

        for storage_device, storage_device_action in microgrid_action[1][
            "storage_devices"
        ].items():
            # storage_device = self.entity.get_entity(storage_device)
            (
                storage_action,
                storage_device_penalty,
                storage_device_info,
            ) = storage_device.act(time, storage_device_action, True)
            for k, v in storage_device_penalty.items():
                penalties[f"{storage_device.id}/{k}"] = v
            microgrid_action[1]["storage_devices"][
                storage_device
            ] = storage_device_action
            infos.append(storage_device_info)

        return microgrid_action, penalties, infos


class Microgrid(Entity):
    def __init__(
        self,
        timestep: np.timedelta64,
        name: str,
        id: int,
        description: str,
        action: MicrogridAction = None,
        config: EntityConfig = None,
        data: EntityData = None,
        schedule: t.Dict = None,
        sources: t.List = None,
        consumers: t.List = None,
        storage_devices: t.List = None,
    ) -> None:
        super().__init__(
            timestep, name, id, description, action, config, data, schedule
        )

        assert len(sources) >= 0, "At least one source should be provided"
        assert len(consumers) >= 0, "At least one consumer should be provided"
        assert (
            len(storage_devices) >= 0
        ), "At least one storage should be provided"

        self.sources = sources
        self.consumers = consumers
        self.storage_devices = storage_devices

        self.entities = sources + consumers + storage_devices

        if action is None:
            self.action = MicrogridAction()
            self.action.set_entity(self)

    def is_schedule_uniform(self):
        return all(entity.is_schedule_uniform() for entity in self.entities)

    def act(self, times: np.ndarray, actions: np.ndarray, train_flag: bool):
        self.current_reference_timestep = times
        ret = self.action(times, actions, self, train_flag)
        self.num_steps += 1
        return ret

    def reset(self):
        for entity in self.entities:
            entity.reset()

    def get_state(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        vectorize: bool,
        type: str,
    ):
        state = OrderedDict()
        state["sources"] = OrderedDict()
        state["consumers"] = OrderedDict()
        state["storage_devices"] = OrderedDict()

        for source in self.sources:
            state["sources"][source] = source.get_state(
                start_time, end_time, vectorize, type
            )
        if vectorize:
            state["sources"] = vectorize_dict(state["sources"])

        for consumer in self.consumers:
            state["consumers"][consumer] = consumer.get_state(
                start_time, end_time, vectorize, type
            )
        if vectorize:
            state["consumers"] = vectorize_dict(state["consumers"])

        for storage_device in self.storage_devices:
            state["storage_devices"][storage_device] = storage_device.get_state(
                start_time, end_time, vectorize, type
            )
        if vectorize:
            state["storage_devices"] = vectorize_dict(state["storage_devices"])

        if vectorize:
            state = vectorize_dict(state)

        return state

    def get_entity(self, id):
        source_ids = [i.id for i in self.sources]
        if id in source_ids:
            return self.sources[source_ids.index(id)]

        consumer_ids = [i.id for i in self.consumers]
        if id in consumer_ids:
            return self.consumers[consumer_ids.index(id)]

        storage_device_ids = [i.id for i in self.storage_devices]
        if id in storage_device_ids:
            return self.storage_devices[storage_device_ids.index(id)]

        raise Exception(f"ID Not found. {id}")

    def set_time(self, time: np.datetime64):
        for entity in self.entities:
            entity.set_time(time)

    def is_done(
        self,
        start_time: np.datetime64 = None,
        end_time: np.datetime64 = None,
        check_data: bool = True,
    ):
        if check_data:
            return any(
                e.is_done() for e in self.entities if not isinstance(e, Storage)
            )
        else:
            return any(
                e.is_done() for e in self.entities if isinstance(e, Storage)
            )
