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

import networkx as nx
from networkx.exception import NetworkXNoCycle

from encortex.decision_unit import DecisionUnit

from .consumer import Consumer
from .contract import Contract
from .market import Market
from .source import Source


logger = logging.getLogger(__name__)


class Producer:
    """Class to encapsulate the decision units of an energy producing company"""

    def __init__(
        self, id: int = uuid.uuid4(), contracts: t.List[Contract] = list()
    ) -> None:
        self.id = id

        self.sources = set()
        self.consumers = set()
        self.markets = set()
        self.contracts = []
        self.decision_units = set()

        for contract in contracts:
            self.add(contract)

        self.graph = None

    def add(self, contract: Contract) -> None:
        """Register a contract between entities to the producer

        Args:
            contract (Contract): Contract between entities that needs to be registered
        """
        self.contracts.append(contract)
        self._add_entity(contract.contractee)
        self._add_entity(contract.contractor)

    def _add_entity(self, entity: t.Union[Source, Market, Consumer]) -> None:
        if isinstance(entity, Source):
            if entity in self.sources:
                logger.warn("Source previously added. Ignoring.")
            self.sources.add(entity)
        elif isinstance(entity, Market):
            if entity in self.markets:
                logger.warn("Market previously added. Ignoring.")
            self.markets.add(entity)
        elif isinstance(entity, Consumer):
            if entity in self.consumers:
                logger.warn("Consumer previously added. Ignoring.")
            self.consumers.add(entity)

    def create_decision_units(self) -> t.List[DecisionUnit]:
        """Create a list of independent sub-graph of the decision units of the producer"""

        if self.graph is not None:
            logger.warn("Graph already exists. Re-writing the graph")

        self.graph = nx.Graph()
        for contract in self.contracts:
            self.graph.add_node(contract.contractor)
            self.graph.add_node(contract.contractee)
            self.graph.add_edge(
                contract.contractor, contract.contractee, contract=contract
            )
            # self.graph[contract.contractor][contract.contractee]["contract"] = contract

        try:
            nx.find_cycle(self.graph)
            raise Exception(
                "Cycle found. Check the contracts and remove cycles."
            )
        except NetworkXNoCycle as e:
            pass

        connected_components = list(nx.connected_components(self.graph))

        decision_units = []
        for components in connected_components:
            contracts = [
                i[2]["contract"]
                for i in self.graph.edges(list(components), data=True)
            ]
            decision_unit = DecisionUnit(contracts)
            decision_units.append(decision_unit)

        self.decision_units = decision_units
        return decision_units

    def get_decision_units(self) -> t.List[DecisionUnit]:
        """Return all the decision units of the producer"""
        if len(self.decision_units) == 0:
            logger.info("No decision units found. Creating them..")
            self.create_decision_units()
        return self.decision_units

    def get_decision_units_graph(self) -> nx.Graph:
        """Returns the networkx graph of the decision units of the producer

        Returns:
            nx.Graph: Networkx graph of a all the entities in the decision unit.
        """
        if self.graph is None:
            logger.info("Graph not found. Creating it..")
            self.create_decision_units()
        return self.graph

    def visualize(self):
        pass
