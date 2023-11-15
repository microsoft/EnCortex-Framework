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
from uuid import UUID

import numpy as np
import pandas as pd
from dfdb import engine, get_db_as_context, get_in_memory_db, in_memory_engine
from dfdb.models import actions as actions_module
from dfdb.models import (
    consumer_actual_demand,
    consumer_forecast_demand,
    market_actual_clearing_price_and_volume,
    market_actual_marginal_carbon_intensity,
    market_forecast_clearing_price_and_volume,
    market_forecast_marginal_carbon_intensity,
    source_actual_generation,
    source_forecast_generation,
)

from encortex.backend import DataBackend, DBBackend
from encortex.utils.data_loaders import load_data

logger = logging.getLogger(__name__)


def infer_interval(df: pd.DataFrame) -> int:
    interval = df.index[1] - df.index[0]
    # assert np.all((df.timestamps[1:] - df.timestamps[:-1]) == ()), "Uneven interval"
    return interval


class DataStructure:
    """DataStructure to represent data of entities encapsulated in a class"""

    def _check_params(self, *args, **kwargs) -> None:
        for arg in args:
            if isinstance(arg, t.List):
                raise ValueError(f"{arg} should be a scalar")

            if isinstance(arg, np.ndarray):
                if arg.shape[0] > 1 or arg.ndim > 1:
                    raise ValueError(f"{arg} should be a scalar")

        for key, value in kwargs.items():
            if isinstance(value, t.List):
                raise ValueError(f"{key} should be a scalar")

            if isinstance(value, np.ndarray):
                if value.shape[0] > 1 or value.ndim > 1:
                    raise ValueError(f"{key} should be a scalar")


class Data:
    def __init__(
        self, entity_id: t.Union[int, UUID, str], in_memory: bool = True
    ) -> None:
        self.entity_id = entity_id
        self.in_memory = in_memory

    def _register_id_to_backend(self):
        for key, value in self.__dict__.items():
            if isinstance(value, DBBackend):
                value.entity_id = self.entity_id
                setattr(self, key, value)

    @staticmethod
    def read_data(
        source_file: str,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        *args,
        **kwargs,
    ) -> Data:
        df = load_data(source_file)
        return Data.parse_df(df, entity_id, in_memory)

    @staticmethod
    def parse_df(
        df: pd.DataFrame,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        *args,
        **kwargs,
    ) -> Data:
        raise NotImplementedError

    def __getitem__(self, key: t.Union[int, np.datetime64, t.Tuple]) -> t.Any:
        raise NotImplementedError

    def _get_db_as_context(self):
        if self.in_memory:
            return get_in_memory_db, in_memory_engine
        else:
            return get_db_as_context, engine

    def get_state(
        self, start_time: np.datetime64, end_time: np.datetime64, type: str
    ):
        raise NotImplementedError

    def is_valid(self, query_time):
        return False


class Bid(DataStructure):
    """Bid represents a single unit of Market data"""

    price: np.float32
    price_uncertainty: t.Union[np.float32, t.Tuple]
    volume: np.float32
    volume_uncertainty: t.Union[np.float32, t.Tuple]
    carbon_emission: np.float32
    carbon_emission_uncertainty: t.Union[np.float32, t.Tuple]
    carbon_price: np.float32
    carbon_price_uncertainty: t.Union[np.float32, t.Tuple]
    timestamp: np.datetime64

    def __init__(
        self,
        price: np.float32,
        price_uncertainty: t.Union[np.float32, t.Tuple],
        volume: np.float32,
        volume_uncertainty: t.Union[np.float32, t.Tuple],
        carbon_emission: np.float32,
        carbon_emission_uncertainty: t.Union[np.float32, t.Tuple],
        carbon_price: np.float32,
        carbon_price_uncertainty: t.Union[np.float32, t.Tuple],
        timestamp: np.datetime64,
    ) -> None:
        self._check_params(
            price=price,
            price_uncertainty=price_uncertainty,
            volume=volume,
            volume_uncertainty=volume_uncertainty,
            carbon_emission=carbon_emission,
            carbon_emission_uncertainty=carbon_emission_uncertainty,
            carbon_price=carbon_price,
            carbon_price_uncertainty=carbon_price_uncertainty,
            timestamp=timestamp,
        )

        self.price = np.float32(price)
        self.volume = np.float32(volume)
        self.carbon_emission = np.float32(carbon_emission)
        self.carbon_emission_uncertainty = carbon_emission_uncertainty
        self.carbon_price = np.float32(carbon_price)
        self.carbon_price_uncertainty = carbon_price_uncertainty
        self.timestamp = np.datetime64(timestamp).astype("datetime64[s]")

    def __repr__(self) -> str:
        return f"Bid(price={self.price}, uncertainty={self.price_uncertainty}, volume={self.volume}, volume_uncertainty={self.volume_uncertainty}, carbon_emission={self.carbon_emission}, carbon_emission_uncertainty={self.carbon_emission_uncertainty}, carbon_price={self.carbon_price}, carbon_price_uncertainty={self.carbon_price_uncertainty}, timestamp={self.timestamp})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Bid):
            return False

        return (
            self.price == __o.price
            and self.volume == __o.volume
            and self.timestamp == __o.timestamp
        )


class SourceGeneration(DataStructure):
    """SourceGeneration represents a single unit of Source Data"""

    generation: np.float32
    generation_uncertainty: t.Union[np.float32, t.Tuple]
    carbon_emission: np.float32
    carbon_emission_uncertainty: t.Union[np.float32, t.Tuple]
    carbon_price: np.float32
    carbon_price_uncertainty: t.Union[np.float32, t.Tuple]
    timestamp: np.datetime64

    def __init__(
        self,
        generation: np.float32,
        generation_uncertainty: t.Union[np.float32, t.Tuple],
        carbon_emission: np.float32,
        carbon_emission_uncertainty: t.Union[np.float32, t.Tuple],
        carbon_price: np.float32,
        carbon_price_uncertainty: t.Union[np.float32, t.Tuple],
        timestamp: np.datetime64,
    ) -> None:
        self._check_params(
            generation=generation,
            generation_uncertainty=generation_uncertainty,
            carbon_emission=carbon_emission,
            carbon_emission_uncertainty=carbon_emission_uncertainty,
            carbon_price=carbon_price,
            carbon_price_uncertainty=carbon_price_uncertainty,
            timestamp=timestamp,
        )

        self.generation = np.float32(generation)
        self.generation_uncertainty = generation_uncertainty
        self.carbon_emission = np.float32(carbon_emission)
        self.carbon_emission_uncertainty = carbon_emission_uncertainty
        self.carbon_price = np.float32(carbon_price)
        self.carbon_price_uncertainty = carbon_price_uncertainty
        self.timestamp = np.datetime64(timestamp).astype("datetime64[s]")

    def __repr__(self) -> str:
        return f"SourceGeneration(generation={self.generation}, uncertainty={self.generation_uncertainty}, carbon_emission={self.carbon_emission}, carbon_emission_uncertainty={self.carbon_emission_uncertainty}, carbon_price={self.carbon_price}, carbon_price_uncertainty={self.carbon_price_uncertainty}, timestamp={self.timestamp})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SourceGeneration):
            return False

        return (
            self.generation == __o.generation
            and self.carbon_emission == __o.carbon_emission
            and self.timestamp == __o.timestamp
        )


class ConsumerDemand(DataStructure):
    """ConsumerDemand represents a single unit of Consumer Data"""

    demand: np.float32
    demand_uncertainty: t.Union[np.float32, t.Tuple]
    timestamp: np.datetime64

    def __init__(
        self,
        demand: np.float32,
        demand_uncertainty: t.Union[np.float32, t.Tuple],
        timestamp: np.datetime64,
    ) -> None:
        self._check_params(
            demand=demand,
            demand_uncertainty=demand_uncertainty,
            timestamp=timestamp,
        )

        self.demand = np.float32(demand)
        self.demand_uncertainty = demand_uncertainty
        self.timestamp = np.datetime64(timestamp).astype("datetime64[s]")

    def __repr__(self) -> str:
        return f"ConsumerDemand(demand={self.demand}, uncertainty={self.demand_uncertainty}, timestamp={self.timestamp})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ConsumerDemand):
            return False

        return self.demand == __o.demand and self.timestamp == __o.timestamp


class Delivery(DataStructure):
    delivery: np.float32
    timestamp: np.datetime64

    def __init__(self, delivery: np.float32, timestamp: np.datetime64) -> None:
        self._check_params(delivery=delivery, timestamp=timestamp)

        self.delivery = np.float32(delivery)
        self.timestamp = np.datetime64(timestamp).astype("datetime64[s]")

    def __repr__(self) -> str:
        return f"Delivery(delivery={self.delivery}, timestamp={self.timestamp})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Delivery):
            return False

        return self.delivery == __o.delivery and self.timestamp == __o.timestamp


class ActionDataStructure(DataStructure):
    """ActionDataStructure represents a single unit of Action Data"""

    action: t.Dict
    timestamp: np.datetime64

    def __init__(self, action: t.Dict, timestamp: np.datetime64) -> None:
        self._check_params(action=action, timestamp=timestamp)

        self.action = action
        self.timestamp = np.datetime64(timestamp).astype("datetime64[s]")

    def __repr__(self) -> str:
        return f"ActionDataStructure(action={self.action}, timestamp={self.timestamp})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ActionDataStructure):
            return False

        return self.action == __o.action and self.timestamp == __o.timestamp


class ActionLog(Data):

    actions: DBBackend

    def __init__(
        self,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        ACTION_TABLE: str = actions_module.Actions,
        ACTION_COLUMN: str = "action",
        TIMESTAMP_COLUMN: str = "start_timestamp",
    ) -> None:
        super().__init__(entity_id, in_memory)

        module_db, module_engine = self._get_db_as_context()
        self.action_table = ACTION_TABLE
        self.actions = DBBackend(
            module_db,
            module_engine,
            ACTION_TABLE,
            columns=[ACTION_COLUMN, TIMESTAMP_COLUMN],
            id_column="creator_id",
            id=entity_id,
            vectorize=False,
        )

    def __getitem__(self, key: t.Union[int, np.datetime64, t.Tuple]) -> t.Any:
        values = self.actions[key]
        values = (
            [
                ActionDataStructure(
                    action=value, timestamp=value.start_timestamp
                )
                for value in values
            ]
            if isinstance(values, t.Iterable)
            else ActionDataStructure(
                action=values.action, timestamp=values.start_timestamp
            )
        )
        return values

    def insert(self, action, timestamp) -> None:
        self.actions.insert(
            self.action_table(
                creator_id=self.entity_id,
                action=action,
                start_timestamp=pd.Timestamp(timestamp),
            )
        )


class DeliveryLog(Data):

    predicted_delivery: DBBackend
    actual_delivery: DBBackend

    def __init__(
        self, entity_id: t.Union[int, UUID, str], in_memory: bool = True
    ) -> None:
        super().__init__(entity_id, in_memory)


class MarketData(Data):
    """MarketData is for querying, inserting market-related data"""

    price_buy_forecast: DBBackend
    price_sell_forecast: DBBackend
    price_forecast_uncertainties: DBBackend
    price_buy_actual: DBBackend
    price_sell_actual: DBBackend
    volume_forecast: DBBackend
    volume_forecast_uncertainties: DBBackend
    volume_actual: DBBackend
    carbon_emissions_forecast: DBBackend
    carbon_emission_forecast_uncertainties: DBBackend
    carbon_emissions_actual: DBBackend
    carbon_prices_forecast: DBBackend
    carbon_price_forecast_uncertainties: DBBackend
    carbon_prices_actual: DBBackend

    def __init__(
        self,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        market_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: np.timedelta64 = None,
        connect_to_db: bool = True,
        PRICE_FORECAST_TABLE: str = market_forecast_clearing_price_and_volume.MarketForecastClearingPriceAndVolume,
        PRICE_FORECAST_COLUMN: str = "clearing_price",
        PRICE_FORECAST_UNCERTAINTY_TABLE: str = market_forecast_clearing_price_and_volume.MarketForecastClearingPriceAndVolume,
        PRICE_FORECAST_UNCERTAINTY_COLUMN: str = "uncertainty_clearing_price",
        PRICE_ACTUAL_TABLE: str = market_actual_clearing_price_and_volume.MarketActualClearingPriceAndVolume,
        PRICE_ACTUAL_COLUMN: str = "clearing_price",
        VOLUME_FORECAST_TABLE: str = market_forecast_clearing_price_and_volume.MarketForecastClearingPriceAndVolume,
        VOLUME_FORECAST_COLUMN: str = "clearing_volume",
        VOLUME_FORECAST_UNCERTAINTY_TABLE: str = market_forecast_clearing_price_and_volume.MarketForecastClearingPriceAndVolume,
        VOLUME_FORECAST_UNCERTAINTY_COLUMN: str = "uncertainty_clearing_volume",
        CARBON_EMISSIONS_FORECAST_TABLE: str = market_forecast_marginal_carbon_intensity.MarketForecastMarginalCarbonIntensity,
        CARBON_EMISSIONS_FORECAST_COLUMN: str = "marginal_carbon_intensity",
        CARBON_EMISSIONS_FORECAST_UNCERTAINTY_TABLE: str = market_forecast_marginal_carbon_intensity.MarketForecastMarginalCarbonIntensity,
        CARBON_EMISSIONS_FORECAST_UNCERTAINTY_COLUMN: str = "uncertainty",
        CARBON_EMISSIONS_ACTUAL_TABLE: str = market_actual_marginal_carbon_intensity.MarketActualMarginalCarbonIntensity,
        CARBON_EMISSIONS_ACTUAL_COLUMN: str = "marginal_carbon_intensity",
        CARBON_PRICES_FORECAST_TABLE: str = market_forecast_clearing_price_and_volume.MarketForecastClearingPriceAndVolume,
        CARBON_PRICES_FORECAST_COLUMN: str = "clearing_price",
        CARBON_PRICES_FORECAST_UNCERTAINTY_TABLE: str = market_forecast_clearing_price_and_volume.MarketForecastClearingPriceAndVolume,
        CARBON_PRICES_FORECAST_UNCERTAINTY_COLUMN: str = "uncertainty_clearing_price",
        CARBON_PRICES_ACTUAL_TABLE: str = market_actual_clearing_price_and_volume.MarketActualClearingPriceAndVolume,
        CARBON_PRICES_ACTUAL_COLUMN: str = "clearing_price",
    ) -> None:
        super().__init__(entity_id, in_memory)

        assert market_id is not None, "market_id is required"
        assert (
            entity_forecasting_id is not None
        ), "entity_forecasting_id is required"
        assert timestep is not None, "timestep is required"

        self.market_id = market_id
        self.entity_forecasting_id = entity_forecasting_id
        self.timestep = timestep.astype("timedelta64[m]").astype(np.int32)

        if connect_to_db:

            self.PRICE_FORECAST_TABLE = PRICE_FORECAST_TABLE
            self.PRICE_FORECAST_COLUMN = PRICE_FORECAST_COLUMN
            self.PRICE_FORECAST_UNCERTAINTY_TABLE = (
                PRICE_FORECAST_UNCERTAINTY_TABLE
            )
            self.PRICE_FORECAST_UNCERTAINTY_COLUMN = (
                PRICE_FORECAST_UNCERTAINTY_COLUMN
            )
            self.PRICE_ACTUAL_TABLE = PRICE_ACTUAL_TABLE
            self.PRICE_ACTUAL_COLUMN = PRICE_ACTUAL_COLUMN
            self.VOLUME_FORECAST_TABLE = VOLUME_FORECAST_TABLE
            self.VOLUME_FORECAST_COLUMN = VOLUME_FORECAST_COLUMN
            self.VOLUME_FORECAST_UNCERTAINTY_TABLE = (
                VOLUME_FORECAST_UNCERTAINTY_TABLE
            )
            self.VOLUME_FORECAST_UNCERTAINTY_COLUMN = (
                VOLUME_FORECAST_UNCERTAINTY_COLUMN
            )
            self.CARBON_EMISSIONS_FORECAST_TABLE = (
                CARBON_EMISSIONS_FORECAST_TABLE
            )
            self.CARBON_EMISSIONS_FORECAST_COLUMN = (
                CARBON_EMISSIONS_FORECAST_COLUMN
            )
            self.CARBON_EMISSIONS_FORECAST_UNCERTAINTY_TABLE = (
                CARBON_EMISSIONS_FORECAST_UNCERTAINTY_TABLE
            )
            self.CARBON_EMISSIONS_FORECAST_UNCERTAINTY_COLUMN = (
                CARBON_EMISSIONS_FORECAST_UNCERTAINTY_COLUMN
            )
            self.CARBON_EMISSIONS_ACTUAL_TABLE = CARBON_EMISSIONS_ACTUAL_TABLE
            self.CARBON_EMISSIONS_ACTUAL_COLUMN = CARBON_EMISSIONS_ACTUAL_COLUMN
            self.CARBON_PRICES_FORECAST_TABLE = CARBON_PRICES_FORECAST_TABLE
            self.CARBON_PRICES_FORECAST_COLUMN = CARBON_PRICES_FORECAST_COLUMN
            self.CARBON_PRICES_FORECAST_UNCERTAINTY_TABLE = (
                CARBON_PRICES_FORECAST_UNCERTAINTY_TABLE
            )
            self.CARBON_PRICES_FORECAST_UNCERTAINTY_COLUMN = (
                CARBON_PRICES_FORECAST_UNCERTAINTY_COLUMN
            )
            self.CARBON_PRICES_ACTUAL_TABLE = CARBON_PRICES_ACTUAL_TABLE
            self.CARBON_PRICES_ACTUAL_COLUMN = CARBON_PRICES_ACTUAL_COLUMN

            module_db, module_engine = self._get_db_as_context()
            self.price_forecast = DBBackend(
                module_db,
                module_engine,
                PRICE_FORECAST_TABLE,
                columns=[PRICE_FORECAST_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.price_forecast_uncertainties = DBBackend(
                module_db,
                module_engine,
                PRICE_FORECAST_UNCERTAINTY_TABLE,
                columns=[PRICE_FORECAST_UNCERTAINTY_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.price_actual = DBBackend(
                module_db,
                module_engine,
                PRICE_ACTUAL_TABLE,
                columns=[PRICE_ACTUAL_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.volume_forecast = DBBackend(
                module_db,
                module_engine,
                VOLUME_FORECAST_TABLE,
                columns=[
                    VOLUME_FORECAST_COLUMN,
                    VOLUME_FORECAST_UNCERTAINTY_COLUMN,
                ],
                id_column="market_id",
                id=entity_id,
            )
            self.volume_forecast_uncertainties = DBBackend(
                module_db,
                module_engine,
                VOLUME_FORECAST_UNCERTAINTY_TABLE,
                columns=[VOLUME_FORECAST_UNCERTAINTY_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.carbon_emissions_forecast = DBBackend(
                module_db,
                module_engine,
                CARBON_EMISSIONS_FORECAST_TABLE,
                columns=[
                    CARBON_EMISSIONS_FORECAST_COLUMN,
                    CARBON_EMISSIONS_FORECAST_UNCERTAINTY_COLUMN,
                ],
                id_column="market_id",
                id=entity_id,
            )
            self.carbon_emission_forecast_uncertainties = DBBackend(
                module_db,
                module_engine,
                CARBON_EMISSIONS_FORECAST_UNCERTAINTY_TABLE,
                columns=[CARBON_EMISSIONS_FORECAST_UNCERTAINTY_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.carbon_emissions_actual = DBBackend(
                module_db,
                module_engine,
                CARBON_EMISSIONS_ACTUAL_TABLE,
                columns=[CARBON_EMISSIONS_ACTUAL_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.carbon_prices_forecast = DBBackend(
                module_db,
                module_engine,
                CARBON_PRICES_FORECAST_TABLE,
                columns=[
                    CARBON_PRICES_FORECAST_COLUMN,
                    CARBON_PRICES_FORECAST_UNCERTAINTY_COLUMN,
                ],
                id_column="market_id",
                id=entity_id,
            )
            self.carbon_price_forecast_uncertainties = DBBackend(
                module_db,
                module_engine,
                CARBON_PRICES_FORECAST_UNCERTAINTY_TABLE,
                columns=[CARBON_PRICES_FORECAST_UNCERTAINTY_COLUMN],
                id_column="market_id",
                id=entity_id,
            )
            self.carbon_prices_actual = DBBackend(
                module_db,
                module_engine,
                CARBON_PRICES_ACTUAL_TABLE,
                columns=[CARBON_PRICES_ACTUAL_COLUMN],
                id_column="market_id",
                id=entity_id,
            )

            self._register_id_to_backend()

    def insert(self, data: t.Dict[str, t.Any]) -> None:
        """Insert data into the database"""
        if "timestamp" not in data.keys():
            raise ValueError("Data must contain a timestamp")

        module_db, _ = self._get_db_as_context()

        with module_db() as module_db:
            if "forecast_price" in data.keys():
                if "forecast_uncertainty_price" not in data.keys():
                    data["forecast_uncertainty_price"] = 0
                if "forecast_uncertainty_volume" not in data.keys():
                    data["forecast_uncertainty_volume"] = 0
                module_db.add(
                    self.PRICE_FORECAST_TABLE(
                        id=hash(data["timestamp"]),
                        market_id=self.market_id,
                        entity_forecasting_id=self.entity_forecasting_id,
                        clearing_price=data["forecast_price"],
                        uncertainty_clearing_price=data[
                            "forecast_uncertainty_price"
                        ],
                        clearing_volume=data["forecast_volume"],
                        uncertainty_clearing_volume=data[
                            "forecast_uncertainty_volume"
                        ],
                        start_timestamp=data["timestamp"],
                        timestep=self.timestep,
                    )
                )
                module_db.commit()
                module_db.flush()

            if "actual_price" in data.keys():
                module_db.add(
                    self.PRICE_ACTUAL_TABLE(
                        market_id=self.market_id,
                        entity_forecasting_id=self.entity_forecasting_id,
                        clearing_price=data["actual_price"],
                        clearing_volume=data["actual_volume"],
                        start_timestamp=data["timestamp"],
                        timestep=self.timestep,
                    )
                )
                module_db.commit()
                module_db.flush()

            if "carbon_emissions_forecast" in data.keys():
                module_db.add(
                    self.CARBON_EMISSIONS_FORECAST_TABLE(
                        market_id=self.market_id,
                        entity_forecasting_id=self.entity_forecasting_id,
                        marginal_carbon_intensity=data[
                            "carbon_emissions_forecast"
                        ],
                        uncertainty=data[
                            "carbon_emissions_forecast_uncertainty"
                        ],
                        start_timestamp=data["timestamp"],
                        timestep=self.timestep,
                    )
                )
                module_db.commit()
                module_db.flush()

            if "carbon_emissions_actual" in data.keys():
                module_db.add(
                    self.CARBON_EMISSIONS_ACTUAL_TABLE(
                        market_id=self.market_id,
                        entity_forecasting_id=self.entity_forecasting_id,
                        marginal_carbon_intensity=data[
                            "carbon_emissions_actual"
                        ],
                        start_timestamp=data["timestamp"],
                        timestep=self.timestep,
                    )
                )
                module_db.commit()
                module_db.flush()

    @staticmethod
    def parse_df(
        df: pd.DataFrame,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        market_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: np.timedelta64 = None,
        *args,
        **kwargs,
    ) -> MarketData:
        """Parse a DataFrame into MarketData"""
        market_data = MarketData(
            entity_id,
            in_memory,
            market_id,
            entity_forecasting_id,
            timestep,
            *args,
            **kwargs,
        )

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a timestamp column")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        for i in range(len(df)):
            data = {}
            row = df.iloc[i]
            data["timestamp"] = pd.Timestamp(row["timestamp"], unit="m")
            if "forecast_price" in df.columns:
                data["forecast_price"] = row["forecast_price"]
                if "forecast_uncertainty_price" in df.columns:
                    data["forecast_uncertainty_price"] = row[
                        "forecast_uncertainty_price"
                    ]
                else:
                    data["forecast_uncertainty_price"] = {}
                if "forecast_volume" in df.columns:
                    data["forecast_volume"] = row["forecast_volume"]
                else:
                    data["forecast_volume"] = 0

                if "forecast_uncertainty_volume" in df.columns:
                    data["forecast_uncertainty_volume"] = row[
                        "forecast_uncertainty_volume"
                    ]
                else:
                    data["forecast_uncertainty_volume"] = {}

            if "actual_price" in df.columns:
                data["actual_price"] = row["actual_price"]
                if "actual_volume" in df.columns:
                    data["actual_volume"] = row["actual_volume"]
                else:
                    data["actual_volume"] = 0

            if "carbon_emissions_forecast" in df.columns:
                data["carbon_emissions_forecast"] = row[
                    "carbon_emissions_forecast"
                ]
                if "carbon_emissions_forecast_uncertainty" in df.columns:
                    data["carbon_emissions_forecast_uncertainty"] = row[
                        "carbon_emissions_forecast_uncertainty"
                    ]
                else:
                    data["carbon_emissions_forecast_uncertainty"] = {}

            if "carbon_emissions_actual" in df.columns:
                data["carbon_emissions_actual"] = row["carbon_emissions_actual"]

            market_data.insert(data)

        return market_data

    @staticmethod
    def parse_csv(
        csv_path,
        entity_id,
        in_memory=True,
        market_id=None,
        entity_forecasting_id=None,
        timestep=None,
        *args,
        **kwargs,
    ) -> MarketData:
        """Parse a CSV file into MarketData"""
        df = pd.read_csv(csv_path)
        return MarketData.parse_df(
            df,
            entity_id,
            in_memory,
            market_id,
            entity_forecasting_id,
            timestep,
            *args,
            **kwargs,
        )

    def __getitem__(self, key: t.Union[np.datetime64, t.Tuple]) -> t.Any:
        """Query market data for a given timestamp(s)"""
        prices = self.prices[key]
        price_uncertainties = self.price_uncertainties[key]
        volumes = self.volumes[key]
        volume_uncertainties = self.volume_uncertainties[key]
        carbon_emissions = self.carbon_emissions[key]
        carbon_emission_uncertainties = self.carbon_emission_uncertainties[key]
        carbon_prices = self.carbon_prices[key]
        carbon_price_uncertainties = self.carbon_price_uncertainties[key]

        raise NotImplementedError

    @staticmethod
    def parse_backend(
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        market_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: np.timedelta64 = None,
        connect_to_db: bool = False,
        price_forecast: DBBackend = None,
        price_forecast_uncertainties: DBBackend = None,
        price_actual: DBBackend = None,
        volume_forecast: DBBackend = None,
        volume_forecast_uncertainties: DBBackend = None,
        volume_actual: DBBackend = None,
        carbon_emissions_forecast: DBBackend = None,
        carbon_emission_forecast_uncertainties: DBBackend = None,
        carbon_emissions_actual: DBBackend = None,
        carbon_prices_forecast: DBBackend = None,
        carbon_price_forecast_uncertainties: DBBackend = None,
        carbon_prices_actual: DBBackend = None,
    ) -> MarketData:
        data = MarketData(
            entity_id,
            in_memory,
            market_id,
            entity_forecasting_id,
            timestep,
            connect_to_db,
        )

        data.price_forecast = price_forecast
        data.price_forecast_uncertainties = price_forecast_uncertainties
        data.price_actual = price_actual
        data.volume_forecast = volume_forecast
        data.volume_forecast_uncertainties = volume_forecast_uncertainties
        data.volume_actual = volume_actual
        data.carbon_emissions_forecast = carbon_emissions_forecast
        data.carbon_emission_forecast_uncertainties = (
            carbon_emission_forecast_uncertainties
        )
        data.carbon_emissions_actual = carbon_emissions_actual
        data.carbon_prices_forecast = carbon_prices_forecast
        data.carbon_price_forecast_uncertainties = (
            carbon_price_forecast_uncertainties
        )
        data.carbon_prices_actual = carbon_prices_actual

        return data

    def get_state(
        self, start_time: np.datetime64, end_time: np.datetime64, type: str
    ):
        state = OrderedDict()
        if type == "forecast":
            # state['price_buy'] = self.price_buy_forecast[start_time:end_time]
            # state['price_sell'] = self.price_sell_forecast[start_time:end_time]
            state["price"] = self.price_forecast[start_time:end_time]
            state["emission"] = self.carbon_emissions_forecast[
                start_time:end_time
            ]
        elif type == "actual":
            # state['price_buy'] = self.price_buy_actual[start_time:end_time]
            # state['price_sell'] = self.price_sell_actual[start_time:end_time]
            state["price"] = self.price_actual[start_time:end_time]
            state["emission"] = self.carbon_emissions_actual[
                start_time:end_time
            ]
        return state

    def is_valid(self, query_time):
        forecast, actual = True, True
        try:
            self.price_forecast[query_time]
        except KeyError as e:
            forecast = False

        try:
            self.price_actual[query_time]
        except KeyError as e:
            actual = False

        if forecast or actual:
            return True
        else:
            return False


class ConsumerData(Data):

    demand_forecast: DBBackend
    demand_uncertainty_forecast: DBBackend
    demand_actual: DBBackend
    demand_uncertainty_actual: DBBackend
    carbon_price_forecast: DBBackend
    carbon_price_uncertainty_forecast: DBBackend
    carbon_price_actual: DBBackend
    carbon_price_uncertainty_actual: DBBackend
    carbon_emissions_forecast: DBBackend
    carbon_emission_uncertainty_forecast: DBBackend
    carbon_emissions_actual: DBBackend
    carbon_emission_uncertainty_actual: DBBackend

    def __init__(
        self,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        consumer_id: t.Any = None,
        carbon_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: t.Any = None,
        connect_to_db: bool = True,
        DEMAND_TABLE_FORECAST: str = consumer_forecast_demand.ConsumerForecastDemand,
        DEMAND_COLUMN_FORECAST: str = "demand",
        DEMAND_UNCERTAINTY_COLUMN_FORECAST: str = "demand_uncertainty",
        DEMAND_TABLE_ACTUAL: str = consumer_actual_demand.ConsumerActualDemand,
        DEMAND_COLUMN_ACTUAL: str = "demand",
        CARBON_EMISSION_TABLE_FORECAST: str = consumer_forecast_demand.ConsumerForecastDemand,
        CARBON_EMISSION_COLUMN_FORECAST: str = "marginal_carbon_intensity",
        CARBON_EMISSION_TABLE_ACTUAL: str = consumer_actual_demand.ConsumerActualDemand,
        CARBON_EMISSION_COLUMN_ACTUAL: str = "marginal_carbon_intensity",
    ) -> None:
        super().__init__(entity_id, in_memory)

        self.DEMAND_TABLE_FORECAST = DEMAND_TABLE_FORECAST
        self.DEMAND_COLUMN_FORECAST = DEMAND_COLUMN_FORECAST
        self.DEMAND_UNCERTAINTY_COLUMN_FORECAST = (
            DEMAND_UNCERTAINTY_COLUMN_FORECAST
        )
        self.DEMAND_TABLE_ACTUAL = DEMAND_TABLE_ACTUAL
        self.DEMAND_COLUMN_ACTUAL = DEMAND_COLUMN_ACTUAL
        self.CARBON_EMISSION_TABLE_FORECAST = CARBON_EMISSION_TABLE_FORECAST
        self.CARBON_EMISSION_COLUMN_FORECAST = CARBON_EMISSION_COLUMN_FORECAST
        self.CARBON_EMISSION_TABLE_ACTUAL = CARBON_EMISSION_TABLE_ACTUAL
        self.CARBON_EMISSION_COLUMN_ACTUAL = CARBON_EMISSION_COLUMN_ACTUAL

        assert consumer_id is not None, "Consumer ID must be provided"
        assert carbon_id is not None, "Carbon ID must be provided"
        assert (
            entity_forecasting_id is not None
        ), "Entity forecasting ID must be provided"

        self.consumer_id = consumer_id
        self.carbon_id = carbon_id
        self.entity_forecasting_id = entity_forecasting_id
        self.timestep = timestep

        if connect_to_db:

            self.demand_forecast = DBBackend(
                *self._get_db_as_context(),
                DEMAND_TABLE_FORECAST,
                DEMAND_COLUMN_FORECAST,
                "consumer_id",
                consumer_id,
            )

            self.demand_uncertainty_forecast = DBBackend(
                *self._get_db_as_context(),
                DEMAND_TABLE_FORECAST,
                DEMAND_UNCERTAINTY_COLUMN_FORECAST,
                "consumer_id",
                consumer_id,
            )

            self.demand_actual = DBBackend(
                *self._get_db_as_context(),
                DEMAND_TABLE_ACTUAL,
                DEMAND_COLUMN_ACTUAL,
                "consumer_id",
                consumer_id,
            )

            self.carbon_emissions_forecast = DBBackend(
                *self._get_db_as_context(),
                CARBON_EMISSION_TABLE_FORECAST,
                CARBON_EMISSION_COLUMN_FORECAST,
                "consumer_id",
                carbon_id,
            )

            self.carbon_emissions_actual = DBBackend(
                *self._get_db_as_context(),
                CARBON_EMISSION_TABLE_ACTUAL,
                CARBON_EMISSION_COLUMN_ACTUAL,
                "consumer_id",
                carbon_id,
            )

            self._register_id_to_backend()

    def insert(self, data: t.Dict[str, t.Any]) -> None:
        """Insert data into the database"""
        if "timestamp" not in data.keys():
            raise ValueError("Data must contain a timestamp")

        if "demand_forecast" not in data.keys():
            raise ValueError("Data must contain a demand forecast")

        if "demand_forecast_uncertainty" not in data.keys():
            data["demand_forecast_uncertainty"] = 0.0

        if "forecast_carbon_emissions" not in data.keys():
            data["forecast_carbon_emissions"] = 0.0

        if "actual_carbon_emissions" not in data.keys():
            data["actual_carbon_emissions"] = 0.0

        module_db, _ = self._get_db_as_context()
        with module_db() as module_db:
            module_db.add(
                self.DEMAND_TABLE_FORECAST(
                    consumer_id=self.consumer_id,
                    entity_forecasting_id=self.entity_forecasting_id,
                    demand=data["demand_forecast"],
                    start_timestamp=data["timestamp"],
                    uncertainty=data["demand_forecast_uncertainty"],
                    timestep=self.timestep,
                    marginal_carbon_intensity=data["forecast_carbon_emissions"],
                )
            )
            module_db.commit()

            if "demand_actual" in data.keys():
                module_db.add(
                    self.DEMAND_TABLE_ACTUAL(
                        consumer_id=self.consumer_id,
                        entity_forecasting_id=self.entity_forecasting_id,
                        demand=data["demand_actual"],
                        start_timestamp=data["timestamp"],
                        timestep=self.timestep,
                        marginal_carbon_intensity=data[
                            "actual_carbon_emissions"
                        ],
                    )
                )
                module_db.commit()

    def __getitem__(self, key: t.Union[np.datetime64, t.Tuple]) -> t.Any:
        """Query consumer data for a given timestamp(s)"""
        if isinstance(key, np.datetime64):
            pass
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError("Tuple must have two elements")

            if not isinstance(key[0], np.datetime64):
                raise ValueError(
                    "First element of tuple must be a np.datetime64"
                )

            if not isinstance(key[1], np.datetime64):
                raise ValueError(
                    "Second element of tuple must be a np.datetime64"
                )

        else:
            raise ValueError("Key must be a np.datetime64 or a tuple")

    @staticmethod
    def parse_df(
        df: pd.DataFrame,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        *args,
        **kwargs,
    ) -> Data:
        """Parse a DataFrame into a Data object"""
        data = ConsumerData(entity_id, in_memory, *args, **kwargs)

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a timestamp column")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        if "demand_forecast" not in df.columns:
            raise ValueError("DataFrame must contain a demand_forecast column")

        for i in range(len(df)):
            consumer_data = {}
            row = df.iloc[i]
            consumer_data["timestamp"] = row["timestamp"]
            consumer_data["demand_forecast"] = row["demand_forecast"]
            if "demand_forecast_uncertainty" in df.columns:
                consumer_data["demand_forecast_uncertainty"] = row[
                    "demand_forecast_uncertainty"
                ]

            if "demand_actual" in df.columns:
                consumer_data["demand_actual"] = row["demand_actual"]

            if "forecast_carbon_emissions" in df.columns:
                consumer_data["forecast_carbon_emissions"] = row[
                    "forecast_carbon_emissions"
                ]

            if "actual_carbon_emissions" in df.columns:
                consumer_data["actual_carbon_emissions"] = row[
                    "actual_carbon_emissions"
                ]

        return data

    @staticmethod
    def parse_backend(
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        consumer_id: t.Optional[int] = None,
        carbon_id: t.Optional[int] = None,
        timestep: t.Optional[int] = None,
        entity_forecasting_id: t.Optional[int] = None,
        connect_to_db: bool = False,
        demand_forecast: DBBackend = None,
        demand_uncertainty_forecast: DBBackend = None,
        demand_actual: DBBackend = None,
        demand_uncertainty_actual: DBBackend = None,
        carbon_price_forecast: DBBackend = None,
        carbon_price_uncertainty_forecast: DBBackend = None,
        carbon_price_actual: DBBackend = None,
        carbon_price_uncertainty_actual: DBBackend = None,
        carbon_emissions_forecast: DBBackend = None,
        carbon_emission_uncertainty_forecast: DBBackend = None,
        carbon_emissions_actual: DBBackend = None,
        carbon_emission_uncertainty_actual: DBBackend = None,
    ) -> Data:
        """Parse a backend into a Data object"""
        data = ConsumerData(
            entity_id,
            in_memory,
            consumer_id,
            carbon_id,
            timestep,
            entity_forecasting_id,
            connect_to_db,
        )

        data.demand_forecast = demand_forecast
        data.demand_uncertainty_forecast = demand_uncertainty_forecast
        data.demand_actual = demand_actual
        data.demand_uncertainty_actual = demand_uncertainty_actual
        data.carbon_price_forecast = carbon_price_forecast
        data.carbon_price_uncertainty_forecast = (
            carbon_price_uncertainty_forecast
        )
        data.carbon_price_actual = carbon_price_actual
        data.carbon_price_uncertainty_actual = carbon_price_uncertainty_actual
        data.carbon_emissions_forecast = carbon_emissions_forecast
        data.carbon_emission_uncertainty_forecast = (
            carbon_emission_uncertainty_forecast
        )
        data.carbon_emissions_actual = carbon_emissions_actual
        data.carbon_emission_uncertainty_actual = (
            carbon_emission_uncertainty_actual
        )

        return data

    def get_state(
        self, start_time: np.datetime64, end_time: np.datetime64, type: str
    ):
        state = OrderedDict()
        if type == "forecast":
            state["demand"] = self.demand_forecast[start_time:end_time]
        elif type == "actual":
            state["demand"] = self.demand_actual[start_time:end_time]
        return state

    def is_valid(self, query_time):
        forecast, actual = True, True
        try:
            self.demand_forecast[query_time]
        except ValueError as e:
            forecast = False

        try:
            self.demand_actual[query_time]
        except ValueError as e:
            actual = False

        if not forecast and not actual:
            raise NotImplementedError("Haven't implemented this logic")

        if forecast or actual:
            return True


class SourceData(Data):

    generation_forecast: DBBackend
    generation_forecast_uncertainty: DBBackend
    generation_actual: DBBackend
    carbon_price_forecast: DBBackend
    carbon_price_uncertainty_forecast: DBBackend
    carbon_price_actual: DBBackend
    carbon_emissions_forecast: DBBackend
    carbon_emission_uncertainty_forecast: DBBackend
    carbon_emissions_actual: DBBackend

    def __init__(
        self,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        source_id: t.Any = None,
        carbon_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: np.timedelta64 = None,
        connect_to_db: bool = True,
        GENERATION_FORECAST_TABLE: str = source_forecast_generation.SourceForecastGeneration,
        GENERATION_FORECAST_COLUMN: str = "generation",
        GENERATION_FORECAST_UNCERTAINTIES_TABLE: str = source_forecast_generation.SourceForecastGeneration,
        GENERATION_FORECAST_UNCERTAINTIES_COLUMN: str = "uncertainty",
        GENERATION_ACTUAL_TABLE: str = source_actual_generation.SourceActualGeneration,
        GENERATION_ACTUAL_COLUMN: str = "generation",
        CARBON_PRICE_FORECAST_TABLE: str = None,
        CARBON_PRICE_FORECAST_COLUMN: str = "carbon_price",
        CARBON_PRICE_FORECAST_UNCERTAINTIES_TABLE: str = None,
        CARBON_PRICE_FORECAST_UNCERTAINTIES_COLUMN: str = "uncertainty",
        CARBON_PRICE_ACTUAL_TABLE: str = None,
        CARBON_PRICE_ACTUAL_COLUMN: str = "carbon_price",
        CARBON_EMISSIONS_FORECAST_TABLE: str = None,
        CARBON_EMISSIONS_FORECAST_COLUMN: str = "carbon_emissions",
        CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_TABLE: str = None,
        CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_COLUMN: str = "uncertainty",
        CARBON_EMISSIONS_ACTUAL_TABLE: str = None,
        CARBON_EMISSIONS_ACTUAL_COLUMN: str = "carbon_emissions",
    ) -> None:

        super().__init__(entity_id, in_memory)

        assert source_id is not None, "source_id must be provided"
        assert (
            entity_forecasting_id is not None
        ), "entity_forecasting_id must be provided"
        assert timestep is not None, "timestep must be provided"

        self.source_id = source_id
        self.carbon_id = carbon_id
        self.entity_forecasting_id = entity_forecasting_id
        self.timestep = timestep.astype("timedelta64[m]").astype(np.int32)

        if connect_to_db:
            self.GENERATION_FORECAST_TABLE = GENERATION_FORECAST_TABLE
            self.GENERATION_FORECAST_COLUMN = GENERATION_FORECAST_COLUMN
            self.GENERATION_FORECAST_UNCERTAINTIES_TABLE = (
                GENERATION_FORECAST_UNCERTAINTIES_TABLE
            )
            self.GENERATION_FORECAST_UNCERTAINTIES_COLUMN = (
                GENERATION_FORECAST_UNCERTAINTIES_COLUMN
            )
            self.GENERATION_ACTUAL_TABLE = GENERATION_ACTUAL_TABLE
            self.GENERATION_ACTUAL_COLUMN = GENERATION_ACTUAL_COLUMN
            self.CARBON_PRICE_FORECAST_TABLE = CARBON_PRICE_FORECAST_TABLE
            self.CARBON_PRICE_FORECAST_COLUMN = CARBON_PRICE_FORECAST_COLUMN
            self.CARBON_PRICE_FORECAST_UNCERTAINTIES_TABLE = (
                CARBON_PRICE_FORECAST_UNCERTAINTIES_TABLE
            )
            self.CARBON_PRICE_FORECAST_UNCERTAINTIES_COLUMN = (
                CARBON_PRICE_FORECAST_UNCERTAINTIES_COLUMN
            )
            self.CARBON_PRICE_ACTUAL_TABLE = CARBON_PRICE_ACTUAL_TABLE
            self.CARBON_PRICE_ACTUAL_COLUMN = CARBON_PRICE_ACTUAL_COLUMN
            self.CARBON_EMISSIONS_FORECAST_TABLE = (
                CARBON_EMISSIONS_FORECAST_TABLE
            )
            self.CARBON_EMISSIONS_FORECAST_COLUMN = (
                CARBON_EMISSIONS_FORECAST_COLUMN
            )
            self.CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_TABLE = (
                CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_TABLE
            )
            self.CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_COLUMN = (
                CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_COLUMN
            )
            self.CARBON_EMISSIONS_ACTUAL_TABLE = CARBON_EMISSIONS_ACTUAL_TABLE
            self.CARBON_EMISSIONS_ACTUAL_COLUMN = CARBON_EMISSIONS_ACTUAL_COLUMN

            self.generation_forecast = DBBackend(
                *self._get_db_as_context(),
                GENERATION_FORECAST_TABLE,
                [GENERATION_FORECAST_COLUMN],
                "source_id",
                source_id,
            )

            self.generation_forecast_uncertainty = DBBackend(
                *self._get_db_as_context(),
                GENERATION_FORECAST_UNCERTAINTIES_TABLE,
                [GENERATION_FORECAST_UNCERTAINTIES_COLUMN],
                "source_id",
                source_id,
            )

            self.generation_actual = DBBackend(
                *self._get_db_as_context(),
                GENERATION_ACTUAL_TABLE,
                [GENERATION_ACTUAL_COLUMN],
                "source_id",
                source_id,
            )

            self.carbon_price_forecast = DBBackend(
                *self._get_db_as_context(),
                CARBON_PRICE_FORECAST_TABLE,
                [CARBON_PRICE_FORECAST_COLUMN],
                "carbon",
                carbon_id,
            )

            self.carbon_price_uncertainty_forecast = DBBackend(
                *self._get_db_as_context(),
                CARBON_PRICE_FORECAST_UNCERTAINTIES_TABLE,
                [CARBON_PRICE_FORECAST_UNCERTAINTIES_COLUMN],
                "carbon",
                carbon_id,
            )

            self.carbon_price_actual = DBBackend(
                *self._get_db_as_context(),
                CARBON_PRICE_ACTUAL_TABLE,
                [CARBON_PRICE_ACTUAL_COLUMN],
                "carbon",
                carbon_id,
            )

            self.carbon_emissions_forecast = DBBackend(
                *self._get_db_as_context(),
                CARBON_EMISSIONS_FORECAST_TABLE,
                [CARBON_EMISSIONS_FORECAST_COLUMN],
                "carbon",
                carbon_id,
            )

            self.carbon_emission_uncertainty_forecast = DBBackend(
                *self._get_db_as_context(),
                CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_TABLE,
                [CARBON_EMISSIONS_FORECAST_UNCERTAINTIES_COLUMN],
                "carbon",
                carbon_id,
            )

            self.carbon_emissions_actual = DBBackend(
                *self._get_db_as_context(),
                CARBON_EMISSIONS_ACTUAL_TABLE,
                [CARBON_EMISSIONS_ACTUAL_COLUMN],
                "carbon",
                carbon_id,
            )

            self._register_id_to_backend()

    def insert(self, data: t.Dict[str, t.Any]) -> None:
        """Insert data into the database"""
        if "generation_forecast" in data.keys():
            module_db, _ = self._get_db_as_context()
            with module_db() as module_db:
                module_db.add(
                    self.GENERATION_FORECAST_TABLE(
                        generation=data["generation_forecast"],
                        entity_forecasting_id=self.entity_forecasting_id,
                        uncertainty=data["generation_forecast_uncertainty"],
                        start_timestamp=data["timestamp"],
                        source_id=self.source_id,
                        timestep=self.timestep,
                    )
                )
                module_db.commit()
        elif "generation_actual" in data.keys():
            with module_db() as module_db:
                module_db.add(
                    self.GENERATION_ACTUAL_TABLE(
                        generation=data["generation_actual"],
                        source_id=self.source_id,
                        entity_id=self.source_id,
                        start_timestamp=data["timestamp"],
                    )
                )
                module_db.commit()
        else:
            logger.warn("No generation data found in data")

    def __getitem__(self, key: t.Union[np.datetime64, t.Tuple]) -> t.Any:
        """Query source data for a given timestamp(s)"""
        raise NotImplementedError("Not implemented yet")

    @staticmethod
    def parse_df(
        df: pd.DataFrame,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        source_id: t.Any = None,
        carbon_id: t.Any = None,
        *args,
        **kwargs,
    ) -> Data:
        """Parse a DataFrame into a Data object"""
        data = SourceData(
            entity_id,
            in_memory,
            source_id=source_id,
            carbon_id=carbon_id,
            *args,
            **kwargs,
        )
        assert df is not None, "DataFrame must not be None"

        if "timestamps" not in list(df.columns):
            raise ValueError(
                f"DataFrame must contain a timestamps column, received: {df.columns}"
            )

        if "generation_forecast" not in list(df.columns):
            raise ValueError(
                f"DataFrame must contain a generation_forecast column, received: {df.columns}"
            )

        for i in range(len(df)):
            source_data = {}
            source_data["timestamp"] = df["timestamps"].iloc[i]
            source_data["generation_forecast"] = df["generation_forecast"].iloc[
                i
            ]

            if "generation_forecast_uncertainty" not in list(df.columns):
                data.generation_forecast_uncertainty.set_constant(0)

            if "generation_actual" not in list(df.columns):
                logger.warn(
                    f"DataFrame must contain a generation_actual column, received: {df.columns}"
                )
            else:
                source_data["generation_actual"] = df["generation_actual"].iloc[
                    i
                ]

            if "carbon_price_forecast" not in list(df.columns):
                data.carbon_price_forecast.set_constant(0)
            else:
                source_data["carbon_price_forecast"] = df[
                    "carbon_price_forecast"
                ].iloc[i]

            if "carbon_price_uncertainty_forecast" not in list(df.columns):
                data.carbon_price_uncertainty_forecast.set_constant(0)
            else:
                source_data["carbon_price_uncertainty_forecast"] = df[
                    "carbon_price_uncertainty_forecast"
                ].iloc[i]

            if "carbon_price_actual" not in list(df.columns):
                data.carbon_price_actual.set_constant(0)
            else:
                source_data["carbon_price_actual"] = df[
                    "carbon_price_actual"
                ].iloc[i]

            if "carbon_emissions_forecast" not in list(df.columns):
                data.carbon_emissions_forecast.set_constant(0)
            else:
                source_data["carbon_emissions_forecast"] = df[
                    "carbon_emissions_forecast"
                ].iloc[i]

            if "carbon_emission_uncertainty_forecast" not in list(df.columns):
                data.carbon_emission_uncertainty_forecast.set_constant(0)
            else:
                source_data["carbon_emission_uncertainty_forecast"] = df[
                    "carbon_emission_uncertainty_forecast"
                ].iloc[i]

            if "carbon_emissions_actual" not in list(df.columns):
                data.carbon_emissions_actual.set_constant(0)
            else:
                source_data["carbon_emissions_actual"] = df[
                    "carbon_emissions_actual"
                ].iloc[i]

            data.insert(source_data)

        return data

    @staticmethod
    def parse_csv(
        file_path: str,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        source_id: t.Any = None,
        carbon_id: t.Any = None,
        *args,
        **kwargs,
    ) -> Data:
        """Parse a CSV file into a Data object"""
        df = load_data(file_path, "source")
        return SourceData.parse_df(
            df, entity_id, in_memory, source_id, carbon_id, *args, **kwargs
        )

    @staticmethod
    def parse_backend(
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        source_id: t.Any = None,
        carbon_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: t.Any = None,
        connect_to_db: bool = True,
        generation_forecast: DBBackend = None,
        generation_forecast_uncertainty: DBBackend = None,
        generation_actual: DBBackend = None,
        carbon_price_forecast: DBBackend = None,
        carbon_price_uncertainty_forecast: DBBackend = None,
        carbon_price_actual: DBBackend = None,
        carbon_emissions_forecast: DBBackend = None,
        carbon_emission_uncertainty_forecast: DBBackend = None,
        carbon_emissions_actual: DBBackend = None,
    ):
        """Parse a DataFrame into a Data object"""
        data = SourceData(
            entity_id,
            in_memory,
            source_id,
            carbon_id,
            entity_forecasting_id,
            timestep,
            connect_to_db,
        )

        data.generation_forecast = generation_forecast
        data.generation_forecast_uncertainty = generation_forecast_uncertainty
        data.generation_actual = generation_actual
        data.carbon_price_forecast = carbon_price_forecast
        data.carbon_price_uncertainty_forecast = (
            carbon_price_uncertainty_forecast
        )
        data.carbon_price_actual = carbon_price_actual
        data.carbon_emissions_forecast = carbon_emissions_forecast
        data.carbon_emission_uncertainty_forecast = (
            carbon_emission_uncertainty_forecast
        )
        data.carbon_emissions_actual = carbon_emissions_actual

        return data

    def get_state(
        self, start_time: np.datetime64, end_time: np.datetime64, type: str
    ):
        state = OrderedDict()
        if type == "forecast":
            state["generation"] = self.generation_forecast[start_time:end_time]
        elif type == "actual":
            state["generation"] = self.generation_actual[start_time:end_time]

    def is_valid(self, query_time):
        forecast, actual = True, True
        try:
            self.generation_forecast[query_time]
        except ValueError as e:
            forecast = False

        try:
            self.generation_actual[query_time]
        except ValueError as e:
            actual = False

        if not forecast and not actual:
            raise NotImplementedError("Haven't implemented this logic")

        if forecast or actual:
            return True


class TransmissionData(Data):

    forecast_marginal_carbon_intensity: DataBackend
    actual_marginal_carbon_intensity: DataBackend

    def __init__(
        self,
        forecast_marginal_carbon_intensity: DataBackend,
        actual_marginal_carbon_intensity: DataBackend,
    ):
        self.forecast_marginal_carbon_intensity = (
            forecast_marginal_carbon_intensity
        )
        self.actual_marginal_carbon_intensity = actual_marginal_carbon_intensity


class UtilityGridData(Data):

    price_buy_forecast: DataBackend
    price_sell_forecast: DataBackend
    price_buy_actual: DataBackend
    price_sell_actual: DataBackend

    def __init__(
        self,
        entity_id: t.Union[int, UUID, str],
        in_memory: bool = True,
        grid_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: np.timedelta64 = None,
        connect_to_db: bool = True,
    ):
        super().__init__(entity_id, in_memory)

        assert grid_id is not None, "grid_id is required"
        assert (
            entity_forecasting_id is not None
        ), "entity_forecasting_id is required"
        assert timestep is not None, "timestep is required"

        self.grid_id = grid_id
        self.entity_forecasting_id = entity_forecasting_id
        self.timestep = timestep.astype("timedelta64[m]").astype(np.int32)

    @staticmethod
    def parse_backend(
        entity_id: t.Union[int, UUID, str],
        in_memory: bool,
        grid_id: t.Any = None,
        entity_forecasting_id: t.Any = None,
        timestep: np.timedelta64 = None,
        connect_to_db: bool = False,
        price_buy_forecast: DBBackend = None,
        price_buy_forecast_uncertainties: DBBackend = None,
        price_buy_actual: DBBackend = None,
        price_sell_forecast: DBBackend = None,
        price_sell_forecast_uncertainties: DBBackend = None,
        price_sell_actual: DBBackend = None,
    ) -> UtilityGridData:

        data = UtilityGridData(
            entity_id,
            in_memory,
            grid_id,
            entity_forecasting_id,
            timestep,
            connect_to_db,
        )

        data.price_buy_forecast = price_buy_forecast
        data.price_buy_forecast_uncertainties = price_buy_forecast_uncertainties
        data.price_buy_actual = price_buy_actual
        data.price_sell_forecast = price_sell_forecast
        data.price_sell_forecast_uncertainties = (
            price_sell_forecast_uncertainties
        )
        data.price_sell_actual = price_sell_actual

        return data

    def get_state(
        self, start_time: np.datetime64, end_time: np.datetime64, type: str
    ):
        state = OrderedDict()
        if type == "forecast":
            state["price_sell"] = self.price_sell_forecast[start_time:end_time]
            state["price_buy"] = self.price_buy_forecast[start_time:end_time]
        elif type == "actual":
            state["price_sell"] = self.price_sell_actual[start_time:end_time]
            state["price_buy"] = self.price_buy_actual[start_time:end_time]
        return state
