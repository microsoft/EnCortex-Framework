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

import os
import typing as t
from collections import OrderedDict

import numpy as np
from cron_validator import CronValidator
from cron_validator.util import str_to_datetime
from rsome import ro
from copy import deepcopy

from encortex.action import Action as EntityAction
from encortex.config import EntityConfig
from encortex.data import Data as EntityData
from encortex.volume_log import VolumeLog
from encortex.utils.time_utils import tuple_to_np_timedelta


class Entity:
    """Entity in the encortex network"""

    action: EntityAction
    config: EntityConfig
    data: EntityData
    schedule: t.Dict

    volume_log: VolumeLog
    contracts: t.List

    def __init__(
        self,
        timestep: np.timedelta64,
        name: str,
        id: int,
        description: str,
        action: EntityAction = None,
        config: EntityConfig = None,
        data: EntityData = None,
        schedule: t.Dict = None,
    ) -> None:
        self.timestep = tuple_to_np_timedelta(timestep)
        self.name = str(name)
        self.id = id
        self.description = str(description)

        if action is not None:
            self.action = action
        else:
            self.action = EntityAction(name, description, timestep=timestep)
        self.action.set_timestep(self.timestep)
        self.action.set_id(id)
        self.action.set_entity(self)
        if config is not None:
            self.config = config
        else:
            self.config = EntityConfig(name)
        if data is not None:
            self.data = data
        if schedule is not None:
            self.schedule = schedule

        self.config["name"] = name
        self.config["id"] = id
        self.config["description"] = description
        self.config["timestep"] = self.timestep

        self.num_steps = 0

        self.contracts = []

    def __repr__(self) -> str:
        return f"{self.__class__} {self.name} {self.id}"

    def __hash__(self) -> int:
        return self.id.__hash__()

    def get_action_space(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def set_time(self, time: np.datetime64):
        self.time = time

    def step(self, time: np.datetime64):
        self.time = time

    def act(self, times: np.ndarray, actions: np.ndarray, train_flag: bool):
        raise NotImplementedError

    def get_constrained_model(self, model: ro.Model) -> ro.Model:
        raise NotImplementedError

    def get_schedule(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ) -> t.Dict:
        assert self.schedule is not None, "Schedule is not provided"
        assert isinstance(self.schedule, str), "Schedule is not of type str"

        schedule = OrderedDict()
        times = list(
            CronValidator.get_execution_time(
                self.schedule,
                from_dt=str_to_datetime(str(start_time)),
                to_dt=str_to_datetime(str(end_time)),
            )
        )
        if start_time != end_time:  # Edge case
            times = times[:-1]

        action_variables = self.action.get_action_variables(
            start_time, end_time, model, apply_constraints, state
        )
        for scheduled_time in times:
            schedule[np.datetime64(scheduled_time).astype("datetime64[m]")] = {}

            for cid in action_variables.keys():
                schedule[np.datetime64(scheduled_time).astype("datetime64[m]")][
                    cid
                ] = action_variables[cid][
                    np.datetime64(scheduled_time).astype("datetime64[m]")
                ]

        return schedule

    def get_target_variables(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
        filter_by_contract_id: t.Any = None,
    ) -> t.Dict:
        variables = self.action.get_action_variables(
            start_time, end_time, model
        )  # Assuming these variables exists in the future. #TODO: Add a reverse check and filter non-existing variables. Currently avoided because very niche cases require this.
        if filter_by_contract_id is not None:
            variables = variables[filter_by_contract_id]
        else:
            variables = variables["all"]
        return variables

    def modify_to_timedelta(self, input_timedelta):
        return tuple_to_np_timedelta(input_timedelta)

    def export(self):
        return self.config.export()

    def graph_image(self):
        return os.path.join(
            __package__,
            f"static/images/{str(self.__class__.__name__).lower()}.png",
        )

    @property
    def node(self) -> t.Tuple[t.Any, t.Dict]:
        return (
            self.id,
            {
                "config": self.config.config,
                "entity": self,
                "image": self.graph_image(),
            },
        )

    def get_volume_variable(self, model: ro.Model, time: np.datetime64):
        return self.action.get_action_variable(model, time)["volume"]

    def __hash__(self) -> int:
        return self.id.__hash__()

    def get_target_times(self, scheduled_time: np.datetime64) -> t.Tuple:
        scheduled_time = np.datetime64(scheduled_time).astype("datetime64[m]")
        return scheduled_time, scheduled_time + self.timestep

    def reset(self):
        pass

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        return False

    def get_state(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        vectorize: bool,
        type: str,
    ):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def is_schedule_uniform(self):
        raise NotImplementedError

    def set_start_time(self, time: np.datetime64):
        self.start_time = deepcopy(time)

    def get_config(self):
        return self.config.config
