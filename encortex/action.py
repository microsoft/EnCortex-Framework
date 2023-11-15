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

import gym
import numpy as np
from rsome import ro


logger = logging.getLogger(__name__)


class Action:
    name: str
    description: str
    action: gym.Space

    def __init__(
        self,
        name: str,
        description: str,
        action: gym.Space = None,
        timestep: np.timedelta64 = None,
    ):
        self.name = name
        self.description = description
        self.action = action
        self.action_log = OrderedDict()

        if timestep:
            self.timestep = timestep

    def _checks(self, action: np.ndarray, *args, **kwargs):
        # assert (
        #     self.action.shape == action.shape
        # ), f"{self.__class__} has a shape mismatch: received {action.shape}, expected {self.action.shape}"
        pass

    def __call__(
        self,
        time: np.ndarray,
        action: np.ndarray,
        entity,
        *args,
        **kwargs: np.ndarray,
    ):
        self._checks(action)

    def sample(self):
        return self.action.sample()

    def action_space(self):
        return self.action

    def log_actions(self, time, action):
        if isinstance(time, t.List):
            for idx, ti in enumerate(time):
                self.action_log[ti] = action[idx]
        else:
            self.action_log[time] = action

    def get_action_log(self):
        return self.action_log

    def get_action_variable(
        self,
        model: ro.Model,
        time: np.datetime64,
        apply_constraints: bool = True,
        cid: int = 1e3,
        **kwargs,
    ) -> t.Dict:
        kwargs_str = ""
        for k, v in kwargs.items():
            kwargs_str += f"{k}_{v}_"
        return {
            "volume": [
                model.dvar(
                    (1,),
                    "C",
                    f"{self.id}_{str(time)}_volume_{kwargs}_cid_{cid}",
                )
            ]
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

        if start_time == end_time:
            action_variable = self.get_action_variable(model, current_time)

            variables[current_time] = action_variable
            current_time = current_time + self.entity.timestep
        else:
            while current_time < end_time:

                action_variable = self.get_action_variable(model, current_time)

                variables[current_time] = action_variable
                current_time = current_time + self.entity.timestep

        self.batch_apply_constraints(variables, model, apply_constraints)

        for contract in self.entity.contracts:

            contract_variables[contract.id] = OrderedDict()
            current_time = start_time
            if start_time == end_time:
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
            else:
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

        for ti in variables.keys():
            v_sum_t = contract_variables[self.entity.contracts[0].id][ti][
                "volume"
            ][0]
            for c in self.entity.contracts[1:]:
                v_sum_t += contract_variables[c.id][ti]["volume"][0]

            model.st(v_sum_t == variables[ti]["volume"][0])

        contract_variables["all"] = variables
        return contract_variables

    def set_timestep(self, timestep: np.timedelta64):
        self.timestep = timestep

    def set_id(self, id):
        self.id = id

    def set_entity(self, entity):
        self.entity = entity

    def batch_apply_constraints(
        self, variables: t.Dict, model: ro.Model, apply_constraints: bool = True
    ):
        pass

    def transform(self, action_type: str, action: int):
        pass

    def transform_variables(
        self, action_type: str, action, model: ro.Model, variables: t.Dict
    ):
        pass
