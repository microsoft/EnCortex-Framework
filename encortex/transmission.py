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
import uuid
from uuid import UUID

import gym
import numpy as np
from cron_validator import CronValidator
from cron_validator.util import str_to_datetime
from gym import spaces
from rsome import ro

from encortex.action import Action
from encortex.data import DeliveryLog, TransmissionData
from encortex.utils.time_utils import get_eod


class TransmissionAction(Action):
    def __init__(
        self,
        name: str = "Transmission",
        description: str = "Transmission",
        action: gym.Space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        ),
    ):
        super().__init__(name, description, action)

    def __call__(
        self,
        time: np.ndarray,
        action: np.ndarray,
        entity,
        *args,
        **kwargs: np.ndarray,
    ):
        super().__call__(time, action, entity, *args, **kwargs)

        self.log_actions(time, action)

        if entity.time in entity.schedule.keys():
            action = self.get_action_log()[entity.time]


class Transmission:
    delivery_commit_schedule: str
    delivery_revise_schedule: str
    data: TransmissionData
    delivery_commit_difference: np.timedelta64
    delivery_revise_difference: np.timedelta64
    action: Action = TransmissionAction()
    delivery: DeliveryLog
    penalties: t.Dict[t.Any, t.Callable]

    def __init__(
        self,
        delivery_commit_schedule: str,
        delivery_revise_schedule: str,
        delivery_commit_difference: np.timedelta64,
        delivery_revise_difference: np.timedelta64,
        contract,
        id: t.Union[int, UUID] = uuid.uuid4(),
        action: Action = TransmissionAction(),
        in_memory: bool = True,
        penalties: t.Dict[t.Any, t.Callable] = None,
    ) -> None:
        self.delivery_commit_schedule = delivery_commit_schedule
        self.delivery_revise_schedule = delivery_revise_schedule
        self.delivery_commit_difference = np.timedelta64(
            delivery_commit_difference
        )
        self.delivery_revise_difference = np.timedelta64(
            delivery_revise_difference
        )

        self.contract = contract
        self.contractor = contract.contractor
        self.contractee = contract.contractee

        self.action = action

        assert (
            self.contractor.timestep == self.contractee.timestep
        ), f"{self.contractor.timestep} != {self.contractee.timestep}"
        self.timestep = self.contractor.timestep

        self.id = id
        self.delivery = DeliveryLog(id, in_memory)

        self.penalties = penalties

    def __repr__(self) -> str:
        return f"{self.__class__} {self.delivery_commit_schedule} {self.delivery_revise_schedule}"

    def set_reference_time(self, reference_time: np.datetime64):
        self.reference_time = reference_time

    def get_commit_times(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> t.List:
        if (
            self.delivery_commit_schedule == ""
            or self.delivery_commit_schedule is None
        ):
            return []
        self.delivery_commit_times = list(
            CronValidator.get_execution_time(
                self.delivery_commit_schedule,
                from_dt=str_to_datetime(str(start_time)),
                to_dt=str_to_datetime(str(end_time)),
            )
        )
        return [np.datetime64(i) for i in self.delivery_commit_times]

    def get_revise_times(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> t.List:
        if (
            self.delivery_revise_schedule == ""
            or self.delivery_revise_schedule is None
        ):
            return []
        self.delivery_revise_times = list(
            CronValidator.get_execution_time(
                self.delivery_revise_schedule,
                from_dt=str_to_datetime(str(start_time)),
                to_dt=str_to_datetime(str(end_time)),
            )
        )
        return [np.datetime64(i) for i in self.delivery_revise_times]

    def act(self):
        pass

    def get_action(self):
        pass

    def get_action_space(self, type: int):
        return self.action.action_space

    def get_schedule(self, start_time, end_time) -> t.Dict:
        transmission_schedule = {}
        for commit_time in self.get_commit_times(start_time, end_time):
            commit_time = np.datetime64(commit_time).astype("datetime64[m]")
            transmission_schedule[commit_time] = []
            commit_time_start = commit_time + self.delivery_commit_difference
            while True:
                end_of_day_of_commit_time_slot = get_eod(commit_time)
                transmission_schedule[commit_time].append(
                    (0, commit_time_start, self.get_action_space(0))
                )
                commit_time_start += self.timestep
                if commit_time_start >= end_of_day_of_commit_time_slot:
                    break

        for revise_time in self.get_revise_times(start_time, end_time):
            revise_time = np.datetime64(revise_time).astype("datetime64[m]")
            revise_time_start = revise_time + self.delivery_revise_difference
            transmission_schedule[revise_time] = []
            while True:
                end_of_day_of_revise_time_slot = get_eod(revise_time)
                transmission_schedule[revise_time].append(
                    (1, revise_time_start, self.get_action_space(1))
                )
                revise_time_start += self.timestep
                if revise_time_start >= end_of_day_of_revise_time_slot:
                    break

        return transmission_schedule

    def set_time(self, time: np.datetime64):
        self.time = time

    def get_target_variables(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
    ):
        return
