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

import numpy as np
from cron_descriptor import get_description
from cron_validator import CronValidator
from cron_validator.util import str_to_datetime
from gym import spaces
from rsome import ro

from encortex.action import Action
from encortex.config import EntityConfig
from encortex.data import Data, MarketData
from encortex.entity import Entity


logger = logging.getLogger(__name__)


class MarketAction(Action):
    """Market Action behaviour"""

    def __init__(
        self,
        name: str = "MarketBid",
        description: str = "Market Bidding action",
        action_dim: int = 1,
        max_bid_price: float = 1.0,
        max_bid_volume: float = 1.0,
        bidding: bool = True,
    ):
        """Market Action initialization

        Args:
            name (str, optional): Market Action Name. Defaults to "MarketBid".
            description (str, optional): Description. Defaults to "Market Bidding action".
            action_dim (int, optional): Action dimension. Defaults to 1.
            max_bid_price (float, optional): Max bid price. Defaults to 1.0.
            max_bid_volume (float, optional): Max bid volume. Defaults to 1.0.
        """
        action = spaces.Tuple(
            (
                spaces.Box(
                    low=0, high=1, shape=(int(action_dim),), dtype=np.float32
                ),  # Price
                spaces.Box(
                    low=0, high=1, shape=(int(action_dim),), dtype=np.float32
                ),  # Volume
            )
        )
        super().__init__(name, description, action)
        self.max_bid_price = max_bid_price
        self.max_bid_volume = max_bid_volume

        self.bidding = bidding

    def _checks(self, action: np.ndarray, *args, **kwargs):
        # super()._checks(action, *args, **kwargs)
        # assert isinstance(action, spaces.Tuple), "Action not of type tuple"
        # assert len(action) == 2
        pass

    def __call__(
        self,
        time: np.ndarray,
        action: np.ndarray,
        entity,
        *args,
        **kwargs: np.ndarray,
    ):
        super().__call__(action, entity, *args, **kwargs)
        # entity.executed_bids.put(
        #     Bid(action[0] * self.max_bid_price, action[1] * self.max_bid_volume)
        # )
        self.log_actions(time, {"volume": action["all"][time]["volume"][0]})
        return action["all"][time]["volume"][0], {}, None

    def get_action_variable(
        self,
        model: ro.Model,
        time: np.datetime64,
        apply_constraints: bool = True,
        cid: int = 1e3,
    ) -> t.Dict:
        if self.bidding:
            volume = model.dvar(
                (1,), "C", f"{self.entity.id}_{str(time)}_volume_cid_{cid}"
            )
            price = model.dvar(
                (1,), "C", f"{self.entity.id}_{str(time)}_price_cid_{cid}"
            )
            return {
                "volume": [volume],
                "price": [price],
            }
        else:
            volume = model.dvar(
                (1,), "C", f"{self.entity.id}_{str(time)}_volume_{cid}"
            )
            return {"volume": [volume]}

    def set_bidding(self, enabled: bool = True):
        self.bidding = enabled


class Market(Entity):  # TODO: Model a generic constraint
    """Market as an entity in the encortex network"""

    bid_start_time_schedule: str  # Cron representation of starting of bidding window
    bid_window: np.timedelta64  # Cron representation of starting of bidding window
    commit_start_schedule: np.timedelta64
    commit_end_schedule: np.timedelta64
    bid_start_time: np.datetime64  # Bid start time for a given day
    bid_end_time: np.datetime64  # Bid end time for a given day
    commit_start_time: np.datetime64  # Start of bid's commit time +2days, 12AM - relative to bid_start_time
    commit_end_time: np.datetime64  # End of bid's commit time +10, 11:45PM - relative to bid_end_time
    bid_window_duration: int  # time difference between commit_start_time and commit_end_time #TODO: change variable name to bid_window_duration.
    horizon: int  # time difference between
    start_of_day: str
    end_of_day: str
    disable_bidding: bool
    # TODO: Add consideration for holiday/off day scenarios.

    data: MarketData

    actions: Action

    current_timestep: int = 0

    def __init__(
        self,
        timestep: np.timedelta64,
        name: str,
        id: int,
        description: str,
        bid_start_time_schedule,
        bid_window,
        commit_start_schedule,
        commit_end_schedule,
        action: MarketAction = MarketAction(),
        config: EntityConfig = None,
        data: MarketData = None,
        schedule: t.Dict = None,
        disable_bidding: bool = False,
    ) -> None:
        super().__init__(
            timestep, name, id, description, action, config, data, schedule
        )

        self.bid_start_time_schedule = bid_start_time_schedule
        self.bid_window = self.modify_to_timedelta(bid_window)
        self.commit_start_schedule = self.modify_to_timedelta(
            commit_start_schedule
        )
        self.commit_end_schedule = self.modify_to_timedelta(commit_end_schedule)

        self.data = data

        self.disable_bidding = disable_bidding

        if action is None:
            time_difference = int(
                (self.commit_end_schedule - self.commit_start_schedule)
                .astype("timedelta64[m]")
                .astype(np.int32)
            )
            num_slots = time_difference / (self.timestep).astype(np.int32)
            assert num_slots is not None
            logger.info(f"Num slots: {num_slots}")
            self.action = MarketAction(action_dim=num_slots)
        else:
            self.action = action

        self.action.set_bidding(not disable_bidding)
        self.action.set_entity(self)

    def __repr__(self) -> str:
        msg = ""
        if hasattr(self, "id"):
            msg += str(self.id)
        else:
            logger.warn(f"Missing ID in {self.__class__}")
            return str(self.commit_end_time)
        return msg

    def _calculate_bid_window(self):
        self.bid_window_duration = self.bid_end_time

    def step(self, timestamp):
        self.timestamp = timestamp

    def get_action(self):
        if not self.disable_bidding:
            return (self.get_price_action(), self.get_volume_action())
        else:
            return None

    def get_price_action(self):
        return self.action.action[0]

    def get_volume_action(self):
        return self.action.action[1]

    def get_state(
        self,
        start_time,
        end_time,
        vectorize: bool = False,
        type: str = "forecast",
    ):
        data = self.data.get_state(start_time, end_time, type)
        assert len(list(data.keys())) > 0, "Received no data"
        if vectorize:
            data = np.concatenate([np.asarray(d) for d in data.values()])
        return data

    def _act(self, action):
        raise NotImplementedError

    def reset(self):
        self.set_time(self.start_time)

    def save(self, dirpath: str):
        raise NotImplementedError

    def load(self, dirpath: str):
        raise NotImplementedError

    def get_bid_start_time(
        self, pretty: bool = False
    ) -> str:  # TODO: Add getters for all other times
        if pretty:
            return get_description(self.bid_start_time)
        return self.bid_start_time

    def get_bid_data_requirements(self) -> t.Tuple[int, Data]:
        num_data = self.horizon * self.window
        data_class = MarketData
        return (num_data, data_class)

    def get_bid_times(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> t.List:
        bid_times = list(
            CronValidator.get_execution_time(
                self.bid_start_time_schedule,
                from_dt=str_to_datetime(str(start_time)),
                to_dt=str_to_datetime(str(end_time)),
            )
        )
        if end_time != start_time:
            bid_times = bid_times[:-1]
        return [np.datetime64(t) for t in bid_times]

    def set_reference_time(self, current_reference_time: np.datetime64) -> None:
        self.current_reference_time = current_reference_time

        self.current_bid_start_time = list(
            CronValidator.get_execution_time(
                self.bid_start_time_schedule,
                from_dt=str_to_datetime(str(self.current_reference_time)),
                to_dt=str_to_datetime(
                    str(self.current_reference_time + np.timedelta64(1, "D"))
                ),
            )
        )
        # assert len(self.current_bid_start_time) == 1, f"Wrong format of bid start time | received {self.current_bid_start_time}"
        self.current_bid_start_time = np.datetime64(
            self.current_bid_start_time[0]
        )

        self.current_bid_end_time = (
            self.current_bid_start_time + self.bid_window
        )

        self.current_commit_start_time = (
            self.current_bid_start_time + self.commit_start_schedule
        )
        self.current_commit_end_time = (
            self.current_bid_end_time + self.commit_end_schedule
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Market):
            return False

        if self.id == other.id:
            return True
        return False

    def __lt__(self, other: Market) -> bool:
        assert (
            self.timestep == other.timestep
        ), "Variable timeslots not supported yet"  # TODO: needs discussion on how to handle this
        if self.commit_end_time < other.commit_end_time:
            return True
        return False

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __repr__(self) -> str:
        return super().__repr__() + f"{self.id}"

    def get_schedule(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ) -> t.Dict:
        bid_times = self.get_bid_times(start_time, end_time)

        assert bid_times[0] >= start_time
        assert bid_times[-1] <= end_time
        schedule = {}

        for bid_time in bid_times:
            commit_start_time = bid_time + self.commit_start_schedule
            commit_end_time = bid_time + self.commit_end_schedule
            schedule[bid_time.astype("datetime64[m]")] = {}
            action_variable = self.action.get_action_variables(
                commit_start_time, commit_end_time, model
            )
            for cid, c_actions in action_variable.items():
                schedule[bid_time.astype("datetime64[m]")][cid] = c_actions

        return schedule

    def is_schedule_uniform(self):
        return False

    def act(self, times: np.ndarray, actions: np.ndarray, train_flag: bool):
        self.current_reference_timestep = times
        logger.info(f"Market action: {actions}")
        ret = self.action(times, actions, self, train_flag)
        self.num_steps += 1
        return ret

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        if start_time is None or end_time is None:
            try:
                return (
                    self.time > self.data.price_forecast.get_end_of_data_time()
                )
            except:
                return (
                    self.time
                    > self.data.price_buy_forecast.get_end_of_data_time()
                )

        else:
            if start_time is None:
                start_time = end_time
            if end_time is None:
                end_time = start_time

            try:
                return (
                    start_time
                    >= self.data.price_forecast.get_end_of_data_time()
                ) or (
                    end_time >= self.data.price_forecast.get_end_of_data_time()
                )
            except:
                logger.info(
                    f"TIMESSS: {start_time} {end_time} {self.data.price_buy_forecast.get_end_of_data_time()}"
                )
                return (
                    start_time
                    >= self.data.price_buy_forecast.get_end_of_data_time()
                ) or (
                    end_time
                    >= self.data.price_buy_forecast.get_end_of_data_time()
                )
