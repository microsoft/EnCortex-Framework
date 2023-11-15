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
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, load_only

from encortex.callbacks.backend_callback import BackendCallback


class DataBackend:
    """Abstraction of a backend for data write and read"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __getitem__(self, i: t.Union[int, datetime, np.datetime64, t.Tuple]):
        raise NotImplementedError

    def insert(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, i: t.Union[int, datetime, np.datetime64, t.Tuple]):
        raise NotImplementedError


class DFBackend(DataBackend):
    """DataFrame Backend for Read/Write/Rewrite using a DataFrame."""

    callbacks: t.List[BackendCallback]

    def __init__(
        self,
        data_column: np.ndarray = None,
        index_column: pd.Series = None,
        is_static=True,
        callbacks=[BackendCallback()],
        timestep=None,
    ) -> None:
        self.is_static = is_static
        self.callbacks = callbacks
        if not is_static:
            assert timestep is not None, "Timestep should be provided"
            self.timestep = timestep

        if data_column is None or index_column is None:
            self.set_constant(0)
        else:
            self.is_constant = False
            self.data_column = data_column
            self.index_column = pd.to_datetime(index_column)
            self.index = pd.Index(self.index_column)

    def set_constant(self, constant: int):
        self.is_constant = True
        self.constant = constant

    def __getitem__(
        self, i: t.Union[int, datetime, np.datetime64, t.Tuple, slice]
    ):
        for callback in self.callbacks:
            i = callback.on_before_data_load(i)
        load_one = True

        if self.is_constant:
            return self.constant

        if isinstance(i, int):
            fetched_data = self.data_column[i]
        elif isinstance(i, datetime):
            fetched_data = self.data_column[
                self.index_column == pd.Timestamp(i)
            ][0]
        elif isinstance(i, np.datetime64):
            fetched_data = self.data_column[
                self.index_column == pd.Timestamp(i)
            ][0]
        elif isinstance(i, pd.Timestamp):
            fetched_data = self.data_column[self.index_column == i][0]
        elif isinstance(i, (slice)):
            load_one = False
            if isinstance(i.start, (datetime, np.datetime64, pd.Timestamp)):
                start = self.index.get_loc(pd.Timestamp(i.start))
                end = self.index.get_loc(pd.Timestamp(i.stop))
                if not self.is_static:
                    difference = int((end - start) / self.timestep)
            elif isinstance(i.stop, int):
                start = i.start
                end = i.stop
                if not self.is_static:
                    difference = end - start
            else:
                raise ValueError("Invalid slice")
            if self.is_static:
                fetched_data = self.data_column[start:end].to_numpy()
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler = scaler.fit(fetched_data.reshape(-1, 1))
                grid_price_scaled = scaler.transform(
                    fetched_data.reshape(-1, 1)
                ).reshape(-1)

                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                fetched_data = np.asarray(self.data_column[start])[:difference]
        elif isinstance(i, t.Tuple):
            i = slice(*i)
            return self.__getitem__(i)
        else:
            raise ValueError("Invalid index type")

        for callback in self.callbacks:
            indices, fetched_data = callback.on_after_data_load(i, fetched_data)

        if load_one and not self.is_static:
            return fetched_data[0]

        return fetched_data

    def __len__(self):
        return len(self.index_column)

    def __iter__(self):
        return iter(self.data_column)

    def insert(self, timestamp, data):
        if isinstance(timestamp, t.Iterable):
            assert len(timestamp) == len(data)
            for t, d in zip(timestamp, data):
                self.insert(t, d)

        if timestamp in self.df.index:
            self.df.loc[timestamp] = data
        else:
            self.df.append(data, index=[timestamp])

    def get_end_of_data_time(self):
        return self.index_column[len(self.index_column) - 1]


class DBBackend(DataBackend):
    """Database backend for Read/Write/Rewrite using a DataFrame."""

    def __init__(
        self,
        module_db: Session,
        module_engine: Engine,
        db_class: t.Any = None,
        columns: t.List[str] = None,
        id_column: str = "id",
        id: t.Optional[int] = None,
        vectorize: bool = True,
    ) -> None:
        super().__init__(module_db, db_class)

        self.db = module_db
        self.engine = module_engine
        self.id_column = id_column

        self.is_constant = False
        self.value = None
        self.vectorize = vectorize

        with self.db() as db:
            if isinstance(db_class, str):
                metadata = MetaData(bind=self.engine, reflect=True)
                if (
                    db_class not in metadata.tables.keys()
                    or db_class is None
                    or columns is None
                ):
                    self.set_constant(0)
                self.db_class = metadata.tables[db_class]

        if not self.is_constant:
            self.db_class = db_class
            self.columns = columns

        if self.db_class is not None:
            assert id is not None, f"No id set for {self.db_class} in DBBackend"
            self.id = id

    def __getitem__(self, i: t.Union[datetime, np.datetime64, t.Tuple]):
        if self.is_constant:
            return self.get_value()

        assert (
            self.id is not None
        ), f"No id set for {self.db_class} in DBBackend"

        with self.db() as db:
            if isinstance(i, datetime):
                value = (
                    db.query(self.db_class)
                    .options(load_only(*self.columns))
                    .filter(getattr(self.db_class, self.id_column) == self.id)
                    .filter(self.db_class.start_timestamp == i)
                    .first()
                )

                if value is None:
                    raise KeyError(f"No data found for {i}")

                if self.vectorize:
                    return np.array([getattr(value, self.columns[0])])

                value
            elif isinstance(i, np.datetime64):
                value = (
                    db.query(self.db_class)
                    .options(load_only(*self.columns))
                    .filter(getattr(self.db_class, self.id_column) == self.id)
                    .filter(self.db_class.start_timestamp == pd.Timestamp(i))
                    .first()
                )

                if value is None:
                    raise KeyError(f"No data found for {i}")

                if self.vectorize:
                    return np.array([getattr(value, self.columns[0])])

                return value
            elif isinstance(i, int):
                value = (
                    db.query(self.db_class)
                    .options(load_only(*self.columns))
                    .filter(getattr(self.db_class, self.id_column) == self.id)
                    .all()
                )

                if value is None:
                    raise KeyError(f"No data found for {i}")

                if self.vectorize:
                    return np.array([getattr(value[i], self.columns[0])])

                return value[i]
            elif isinstance(i, t.Tuple):
                if isinstance(i[0], (datetime, pd.Timestamp)):
                    values = (
                        db.query(self.db_class)
                        .options(load_only(*self.columns))
                        .filter(
                            getattr(self.db_class, self.id_column) == self.id
                        )
                        .filter(
                            self.db_class.start_timestamp >= i[0],
                            self.db_class.start_timestamp < i[1],
                        )
                        .all()
                    )
                    if value is None:
                        raise KeyError(f"No data found for {i}")
                    if self.vectorize:
                        values = np.asarray(
                            [getattr(i, self.columns[0]) for i in values]
                        )

                    return values

                elif isinstance(i[0], np.datetime64):
                    values = (
                        db.query(self.db_class)
                        .filter(
                            getattr(self.db_class, self.id_column) == self.id
                        )
                        .filter(
                            self.db_class.start_timestamp >= pd.Timestamp(i[0]),
                            self.db_class.start_timestamp < pd.Timestamp(i[1]),
                        )
                        .all()
                    )
                    if values is None:
                        raise KeyError(f"No data found for {i}")
                    if self.vectorize:
                        values = np.asarray(
                            [getattr(i, self.columns[0]) for i in values]
                        )

                    return values
                elif isinstance(i[0], int):
                    values = (
                        db.query(self.db_class)
                        .options(load_only(*self.columns))
                        .filter(
                            getattr(self.db_class, self.id_column) == self.id
                        )
                        .all()
                    )
                    if value is None:
                        raise KeyError(f"No data found for {i}")
                    if self.vectorize:
                        values = np.asarray(
                            [
                                getattr(i, self.columns[0])
                                for i in values[i[0] : i[1]]
                            ]
                        )

                    return values[i[0] : i[1]]
                else:
                    values = (
                        db.query(self.db_class)
                        .options(load_only(*self.columns))
                        .filter(
                            getattr(self.db_class, self.id_column) == self.id
                        )
                        .all()
                    )
                    if value is None:
                        raise KeyError(f"No data found for {i}")
                    if self.vectorize:
                        values = np.asarray(
                            [getattr(i, self.columns[0]) for i in values[i]]
                        )

                    return values[i]
            elif isinstance(i, slice):
                return self.__getitem__((i.start, i.stop))
            else:
                raise ValueError(f"Invalid index type: {type(i)} {i}")

    def __len__(self):
        if self.is_constant:
            return 1
        with self.db() as db:
            return (
                db.query(self.db_class)
                .filter(getattr(self.db_class, self.id_column) == self.id)
                .count()
            )

    def __iter__(self):
        if self.is_constant:
            return iter(self.get_value())
        with self.db() as db:
            data = (
                db.query(self.db_class)
                .filter(getattr(self.db_class, self.id_column) == self.id)
                .options(load_only(self.columns))
                .all()
            )
            for d in data:
                if self.vectorize:
                    yield np.asarray(getattr(d, self.columns[0]))
                else:
                    yield d

    def insert(self, data):
        """Insert a new data point into the database.

        Args:
            time (_type_): _description_
            data (_type_): _description_
        """
        if self.is_constant:
            raise ValueError("Cannot insert data into a constant database")

        with self.db() as db:
            current_data = (
                db.query(self.db_class)
                .filter(
                    self.db_class.start_timestamp
                    == pd.Timestamp(data.start_timestamp)
                )
                .first()
            )
            if current_data:
                current_data.update(data)  # TODO: update this
            else:
                db.add(data)
                db.commit()
                db.flush()

    def set_constant(self, constant: np.float32):
        self.is_constant = True
        self.value = constant

    def get_value(self):
        if self.is_constant:
            assert self.value is not None, "No constant value set"
            return self.value
        raise ValueError("Database is not constant")
