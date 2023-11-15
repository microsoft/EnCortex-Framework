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
import uuid
from collections import OrderedDict
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym.spaces import Dict, Tuple
from rsome import ro

from encortex.callbacks.decision_unit_callback import DecisionUnitCallback
from encortex.consumer import Consumer
from encortex.contract import Contract
from encortex.entity import Entity
from encortex.grid import Grid
from encortex.market import Market
from encortex.microgrid import Microgrid
from encortex.source import Source
from encortex.sources.battery import Battery
from encortex.sources.storage import Storage
from encortex.transmission import Transmission
from encortex.utils.transform import vectorize_dict


logger = logging.getLogger(__name__)


class DecisionUnit:
    """Decision Unit of EnCortex"""

    def __init__(
        self,
        contracts: t.List[Contract],
        callbacks: t.List[DecisionUnitCallback] = [DecisionUnitCallback()],
    ) -> None:
        self.contracts = (
            contracts  # TODO: parse and extract sources, markets, etc.
        )
        self.callbacks = callbacks

        self.sources = []
        self.consumers = []
        self.markets = []
        self.transmissions = []
        self.storage_entities = []
        self.microgrids = []
        self.graph = nx.DiGraph()

        self.entities: t.Dict[t.Any, Entity] = OrderedDict()

        self._parse_contracts()
        self._set_timestep()
        self._set_horizon()

    def _parse_contracts(self):
        for contract in self.contracts:
            if isinstance(contract.contractor, Source):
                self.sources.append(contract.contractor)
            if isinstance(contract.contractor, Market):
                self.markets.append(contract.contractor)
            if isinstance(contract.contractor, Consumer):
                self.consumers.append(contract.contractor)
            if isinstance(contract.contractor, Storage):
                self.storage_entities.append(contract.contractor)
            if isinstance(contract.contractor, Microgrid):
                self.microgrids.append(contract.contractor)

            if isinstance(contract.contractee, Source):
                self.sources.append(contract.contractee)
            if isinstance(contract.contractee, Market):
                self.markets.append(contract.contractee)
            if isinstance(contract.contractee, Consumer):
                self.consumers.append(contract.contractee)
            if isinstance(contract.contractee, Storage):
                self.storage_entities.append(contract.contractee)
            if isinstance(contract.contractee, Microgrid):
                self.microgrids.append(contract.contractee)

            if isinstance(contract.transmission, Transmission):
                self.transmissions.append(contract.transmission)

            for edge in contract.edge:
                u_node = edge[0]
                v_node = edge[1]
                self.graph.add_node(u_node[0], **u_node[1])
                self.graph.add_node(v_node[0], **v_node[1])

                self.graph.add_edge(u_node[0], v_node[0])
                if contract.bidirectional:
                    self.graph.add_edge(v_node[0], u_node[0])

                self.entities[u_node[0]] = u_node[1]["entity"]

        self._check_graph()

        self.markets = list(
            sorted(self.markets, key=lambda x: x.commit_end_schedule)
        )

        for source in self.sources:
            self.entities[source.id] = source

        for market in self.markets:
            self.entities[market.id] = market

        for consumer in self.consumers:
            self.entities[consumer.id] = consumer

        for transmission in self.transmissions:
            self.entities[transmission.id] = transmission

        for storage_entity in self.storage_entities:
            self.entities[storage_entity.id] = storage_entity

        for microgrid in self.microgrids:
            self.entities[microgrid.id] = microgrid

        self.id_to_contracts = {}
        for c in self.contracts:
            self.id_to_contracts[c.id] = c

    def _check_graph(self):
        assert self.graph is not None, "Graph is None"
        assert isinstance(
            self.graph, nx.DiGraph
        ), "Graph not of type nx.DiGraph"

        degrees = [degree for _, degree in self.graph.degree()]

        # assert np.count_nonzero(np.asarray(degrees) > 1) <= 1, f"EnCortex currently doesn't support multiple nodes with degree greater than 1. Please check your graph: {np.asarray(degrees), self.graph.edges}"

    def _set_timestep(self):
        timesteps = []
        for entity in self.entities.values():
            timesteps.append(
                entity.timestep.astype("timedelta64[m]").astype(np.int32)
            )

        gcf = np.gcd.reduce(timesteps)
        if gcf == 1:
            warn("No common timestep found")
            raise ValueError("No common timestep found")

        self.timestep = gcf.astype("timedelta64[m]")

    def _set_horizon(self):
        self.horizon = np.timedelta64(1, "D")
        for market in self.markets:
            market: Market
            if (
                market.commit_end_schedule.astype("timedelta64[D]")
                > self.horizon
            ):
                self.horizon = market.commit_end_schedule.astype(
                    "timedelta64[D]"
                )

    def get_entity(
        self, id: t.Union[int, uuid.UUID]
    ) -> t.Union[Source, Market, Consumer, Grid, Battery]:
        return self.entities[id]

    def step(self, timestamp, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_decision_unit_before_step(self)

        if not timestamp in self.schedule.keys():
            logger.info("Skipping")
            return True

        for source in self.sources:
            source.step(timestamp, *args, **kwargs)

        for market in self.markets:
            market.step(timestamp, *args, **kwargs)

        for consumer in self.consumers:
            consumer.step(timestamp, *args, **kwargs)

        for storage_entity in self.storage_entities:
            storage_entity.step(timestamp, *args, **kwargs)

        for microgrid in self.microgrids:
            microgrid.step(timestamp, *args, **kwargs)

        for transmission in self.transmissions:
            transmission.step(timestamp, *args, **kwargs)

        # if len(self.schedule[timestamp]) == self.timestep_idx:
        #     return True

        self.optimize_entity = self.get_entity(self.schedule[timestamp][0])
        if isinstance(self.optimize_entity, Market):
            self.optimize_market = self.optimize_entity
            self.optimize_market_position = self.markets.index(
                self.optimize_market
            )
        elif isinstance(self.optimize_entity, Transmission):
            pass

        # self.timestep_idx += 1

        for callback in self.callbacks:
            callback.on_decision_unit_after_step(self)

        return True

    def get_market(self, id: t.Union[int, uuid.UUID]) -> Market:
        for i, market in enumerate(self.markets):
            if market.id == id:
                return market, i
        assert False, f"Market with id {id} not found"

    def get_action_space(
        self, return_active_instances: bool = True, return_as_dict=True
    ):
        actions = {}
        active = []
        for source in self.sources:
            if source.has_action():
                actions[source.id] = source.get_actions()
            active.append(1)

        for idx, market in enumerate(self.markets):
            if market.has_action():
                actions[market.id] = market.get_actions()
            if not hasattr(self, "optimize_market"):
                active.append(1)
            else:
                active.append(
                    0
                ) if idx < self.optimize_market else active.append(1)

        for consumer in self.consumers:
            if consumer.has_action():
                actions[consumer.id] = consumer.get_actions()
            active.append(1)

        for transmission in self.transmissions:
            if transmission.has_action:
                actions[transmission.id] = transmission.get_actions()
            active.append(1)

        if return_active_instances:
            return Tuple(actions), np.array(active).astype(np.int8)

        if return_as_dict:
            return Dict(actions)
        else:
            return Tuple(tuple([action for action in actions.values()]))

    def get_state(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        vectorize=False,
        type="forecast",
    ):
        state = {}
        for entity in self.entities.values():
            state[entity] = entity.get_state(
                start_time, end_time, vectorize=vectorize, type=type
            )

        if vectorize:
            state = vectorize_dict(state)
        return state

    def reset_step(self):
        self.timestep_idx = 0
        for entity in self.entities.values():
            entity.reset()

    def reset(self):
        self.reset_step()

    def get_contracts(self):
        return self.contracts

    def __call__(self, timestamp, *args: t.Any, **kwargs: t.Any) -> t.Any:
        self.step(timestamp, *args, **kwargs)

    def _get_current_action_space(self):
        """
        Returns the current action space for all the entities.
        """
        action_space = {}
        for entity in self.entities.values():
            action_space[entity.id] = entity.get_action_space()
        return action_space

    def act(self, time, actions: t.Dict, train_flag: bool):
        """Executes the actions.

        The method parses the verified actions to the respective entities.

        Args:
            actions (Dict): Dictionary mapping ids to actions.
        """
        time = time.astype("datetime64[m]")
        assert actions is not None, "Actions cannot be None"
        for callback in self.callbacks:
            callback.on_decision_unit_before_actions(self, actions)

        # assert set(list(actions.keys())).issubset(
        #     set(list(self.get_action_space().keys()))
        # ), f"{actions.keys()} != {self.get_action_space().keys()}"  # Check if all actions are as expected
        du_action = {}
        # assert list(sorted(actions.keys())) == list(
        #     sorted(self.get_current_schedule().keys())
        # )
        for contract_id, time_action_pair in actions.items():
            contract = self.get_contract(contract_id)
            contract_penalties = []
            for time, action in time_action_pair.items():
                contract_action, contract_penalty, contract_info = contract.act(
                    time, action, train_flag
                )
                contract_penalties.append(contract_penalty)
                # time_action_pair[time] = contract_action
            du_action[contract.id] = contract_penalties
        for callback in self.callbacks:
            callback.on_decision_unit_after_actions(self, actions)
        return du_action

    def get_schedule(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ) -> t.OrderedDict:
        """
        Generates the schedule for all the contracts related to all the related sources with reference to current reference time
        """
        schedule = {}
        contracts = self.get_contracts()
        for contract in contracts:
            if state is not None:
                schedule[contract.id] = contract.get_schedule(
                    start_time, end_time, model, apply_constraints, state
                )
            else:
                schedule[contract.id] = contract.get_schedule(
                    start_time, end_time, model, apply_constraints, None
                )

        return schedule

    def get_valid_action_space(self) -> t.Tuple[bool, bool, bool]:
        """Return's if price, volume and battery actions exist

        Returns:
            t.Tuple[bool, bool, bool]: price, volume and a
        """
        if isinstance(self.optimize_entity, Market):
            return True, True, self.has_battery_actions
        elif isinstance(self.optimize_entity, (Transmission, Grid)):
            return False, True, self.has_battery_actions
        else:
            raise NotImplementedError

    def get_current_action_space(self) -> t.Tuple[t.Dict, t.Dict, t.Dict]:
        valid_action_space = self.get_valid_action_space()
        action_space = []
        if valid_action_space[0]:
            action_space.append(self.price_action)

        if valid_action_space[1]:
            action_space.append(self.volume_action)

        if valid_action_space[2]:
            action_space.append(self.battery_action)

        return tuple(action_space)

    def set_time(self, time: np.datetime64):
        for entity in self.entities.values():
            entity.set_time(time)
        self.time = time

    def set_start_time(self, time: np.datetime64):
        for entity in self.entities.values():
            entity.set_start_time(time)
        self.time = time

    def visualize_graph(self):
        G = self.graph
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots()

        # Note: the min_source/target_margin kwargs only work with FancyArrowPatch objects.
        # Force the use of FancyArrowPatch for edge drawing by setting `arrows=True`,
        # but suppress arrowheads with `arrowstyle="-"`
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            arrows=True,
            arrowstyle="-",
            min_source_margin=15,
            min_target_margin=15,
        )

        # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
        tr_figure = ax.transData.transform
        # Transform from display to figure coordinates
        tr_axes = fig.transFigure.inverted().transform

        # Select the size of the image (relative to the X axis)
        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
        icon_center = icon_size / 2.0

        # Add the respective image to each node
        for n in G.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes(
                [xa - icon_center, ya - icon_center, icon_size, icon_size]
            )
            a.imshow(G.nodes[n]["image"])
            a.axis("off")
        plt.savefig("plt.jpg")

    def get_decision_unit_variables_at_times(
        self,
        contracts: t.List[Contract],
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
    ):
        # Cluster Time slots
        # Get actions for each time step
        # 1. Change schedule to accommodate actions
        # 2. variables unified with actions
        # 3. Give access to all necessary variables through this
        # 4. Unify data access
        # 5. Environment will take care of stitching the models together - optimizer will be a solver.
        variables = {}
        for contract in contracts:
            variables[contract.id] = contract.get_target_variables(
                start_time, end_time, model
            )

        return variables

    def is_schedule_uniform(self):
        return all([entity.is_schedule_uniform() for entity in self.entities])

    def get_current_schedule(self, model: ro.Model):
        return self.get_schedule(self.time, self.time, model)

    def get_contract(self, id):
        return self.id_to_contracts[id]

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        for entity in self.entities.values():
            logger.info(f"Done: Entity {entity} {entity.is_done()}")
        return any(entity.is_done() for entity in self.entities.values())

    def get_config(self):
        config = {}
        for contract in self.contracts:
            config[str(contract)] = contract.get_config()

        return config
