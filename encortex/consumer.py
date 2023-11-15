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

import typing as t

import gym
import numpy as np
from gym import spaces
from rsome import ro

from encortex.action import Action as EntityAction
from encortex.config import EntityConfig
from encortex.data import ConsumerData
from encortex.data import Data as EntityData
from encortex.entity import Entity


class ConsumerAction(EntityAction):
    """Consumer Action behaviour"""

    def __init__(
        self,
        name: str = "Consumer supply",
        description: str = "Consumer supply",
        action: gym.Space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        ),
        max_value: float = 1.0,
    ):
        """Consumer action initialization

        Args:
            name (str, optional): Name. Defaults to "Consumer supply".
            description (str, optional): Description. Defaults to "Consumer supply".
            action (gym.Space, optional): Action space. Defaults to spaces.Box( low=0, high=1, shape=(1,), dtype=np.float32 ).
            max_value (float, optional): Max value to scale the action space to. Defaults to 1.0.
        """
        super().__init__(name, description, action)
        self.max_value = max_value

    def _checks(self, action: spaces.Box, *args, **kwargs):
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
        self.log_actions(time, {"demand": action["volume"][0][0]})
        return action["volume"][0][0], {}, None

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
                model.dvar((1,), "C", f"{self.id}_{str(time)}_volume_{kwargs}")
            ],
            # "volume_excess": [
            #     model.dvar((1,), "C", f"{self.id}_{str(time)}_volume_{kwargs}")
            # ],
        }


class Consumer(Entity):
    horizon: int
    timeslots: int
    data: ConsumerData

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
        disable_action: bool = True,
    ) -> None:
        if action is None:
            action = ConsumerAction()
        super().__init__(
            timestep, name, id, description, action, config, data, schedule
        )

        self.disable_action = disable_action

    def get_actions(self):
        if self.disable_action:
            return None
        else:
            return self.action.action

    def get_state(
        self,
        start_time,
        end_time,
        vectorize: bool = True,
        type: str = "forecast",
    ):
        data = self.data.get_state(start_time, end_time, type)
        if vectorize:
            data = np.concatenate([np.asarray(d) for d in data.values()])
        return data

    def is_schedule_uniform(self):
        return True

    def act(self, times: np.ndarray, actions: np.ndarray, train_flag: bool):
        ret = self.action(times, actions, self)
        return ret

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        if start_time is None or end_time is None:
            return self.time > self.data.demand_forecast.get_end_of_data_time()
        else:
            if start_time is None:
                start_time = end_time
            if end_time is None:
                end_time = start_time
            return (
                start_time > self.data.demand_forecast.get_end_of_data_time()
            ) and (end_time > self.data.demand_forecast.get_end_of_data_time())
