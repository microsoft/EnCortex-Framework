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

import numpy as np
from class_registry import ClassRegistry

from encortex.action import Action as EntityAction
from encortex.config import EntityConfig
from encortex.data import Data

EntityData = Data
from encortex.source import Source
from encortex.utils.data_loaders import load_data
from encortex.visualize import plot_source_data

ENERGY_SOURCES = ClassRegistry("energy_sources")


class WindCapacityChange(EntityAction):
    def __init__(
        self,
        name: str = "WindCapacityChange",
        description: str = "Changes maximum capacity",
        shape: t.Tuple = (1),
    ):
        super().__init__(name, description, shape)

    def _checks(self, action: np.ndarray, *args, **kwargs):
        super()._checks(action, *args, **kwargs)
        assert action >= 0.0 and action <= 1.0

    def __call__(self, action: np.ndarray, entity, *args, **kwargs: np.ndarray):
        super().__call__(action, entity, *args, **kwargs)
        entity.current_max_capacity = action * entity.max_capacity


@Source.register
class Wind(Source):
    max_capacity: float
    generation_forecast: MarketData
    generation_actual: MarketData
    current_max_capacity: float
    actions: t.Dict

    def __init__(
        self,
        max_capacity: float,
        generation_forecast: MarketData,
        generation_actual: MarketData,
    ) -> None:
        self.max_capacity = max_capacity
        self.generation_actual = generation_actual
        self.generation_forecast = generation_forecast

        self.current_max_capacity = max_capacity

        self._build_actions()

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
        super().__init__(
            timestep, name, id, description, action, config, data, schedule
        )

    def _build_actions(self):
        self.actions = [
            WindCapacityChange()
        ]  # A downside of keeping objects for actions is that they'll always be in memory! Could be huge memory consumption..

    def step(self, action):
        return self.actions[0](action, self)

    def visualize(
        self,
        is_forecast: bool,
        start_time: np.datetime64 = None,
        end_time: np.datetime64 = None,
    ):
        data = (
            self.generation_forecast if is_forecast else self.generation_actual
        )
        plot_source_data(data, start_time, end_time)

    @staticmethod
    def parse_data(
        max_capacity, forecast_data_filename, actual_data_filename
    ) -> Wind:
        return Wind(
            max_capacity=max_capacity,
            generation_forecast=load_data(forecast_data_filename, "source"),
            generation_actual=load_data(actual_data_filename, "source"),
        )
