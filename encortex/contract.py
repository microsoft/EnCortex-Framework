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
import uuid
from collections import OrderedDict

import numpy as np
from rsome import ro

# from encortex.consumer import Consumer
import warnings
from encortex.entity import Entity

# from encortex.grid import Grid
# from encortex.market import Market
# from encortex.source import Source
from encortex.transmission import Transmission


class Contract:
    """Contract between two entities in an EnCortex graph"""

    contractor: Entity
    contractee: Entity
    transmission: Transmission
    penalty_structure: t.Dict

    def __init__(
        self,
        contractor: Entity,
        contractee: Entity,
        bidirectional: bool,
        delivery_commit_schedule: str = None,
        delivery_commit_difference: np.timedelta64 = np.timedelta64("15", "m"),
        delivery_revision_schedule: str = None,
        delivery_revision_difference: np.timedelta64 = np.timedelta64(
            "30", "m"
        ),
        penalty_structure: t.Dict = None,
        id: int = uuid.uuid4(),
    ) -> None:
        """Initialize a contract between two entities in an encortex network

        Args:
            contractor (t.Union[Source, Market]): First entity of a contract(from)
            contractee (t.Union[Consumer, Market]): Second entity of a contract(to)
            delivery_commit_schedule (str, optional): Cron string of delivery commit schedule. Defaults to None.
            delivery_commit_difference (np.timedelta64, optional): Time ahead commit can be conducted till end of day. Defaults to np.timedelta64("15", "m").
            delivery_revision_schedule (str, optional): Cron string of a delivery revision schedule. Defaults to None.
            delivery_revision_difference (np.timedelta64, optional): Time ahead revisions can be conducted till end of day. Defaults to np.timedelta64( "30", "m" ).
            penalty_structure (t.Dict, optional): Callable penalty function. Defaults to None.
            id (int, optional): ID. Defaults to uuid.uuid4().
        """
        self.contractor = contractor
        self.contractee = contractee
        self.bidirectional = bidirectional

        if delivery_commit_schedule is not None:
            self.transmission = Transmission(
                delivery_commit_schedule,
                delivery_revision_schedule,
                delivery_commit_difference,
                delivery_revision_difference,
                self,
                None,
            )
        else:
            self.transmission = None
        self.penalty_structure = penalty_structure
        self.id = id

        self.contractor.contracts.append(self)
        self.contractee.contracts.append(self)

        self.action_log = OrderedDict()

    def calculate_penalty(
        self, step: int, expected_values: np.ndarray, actual_values: np.ndarray
    ):
        if step in self.penalty_structure.keys():
            self.penalty_structure[step](expected_values, actual_values)

    def get_actions(self):
        return (
            self.contractor.get_actions()
            + self.contractee.get_actions()
            + self.transmission.get_action()
        )

    def __eq__(self, __o: object) -> bool:
        if type(__o) is Contract:
            return self.id == __o.id
        else:
            return False

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __repr__(self) -> str:
        return super().__repr__() + f"({self.contractor}, {self.contractee})"

    @property
    def edge(
        self,
    ) -> t.Union[t.Tuple[t.Tuple[Entity, Entity]], t.Tuple[Entity, Entity]]:
        if self.bidirectional:
            return (
                (self.contractor.node, self.contractee.node),
                (self.contractor.node, self.contractee.node),
            )
        else:
            return ((self.contractor.node, self.contractee.node),)

    def get_volume_variable(self, model: ro.Model, time: np.datetime64):
        contractee_dvar = self.contractee.get_volume_variable(model, time)
        contractor_dvar = self.contractor.get_volume_variable(model, time)

        if not self.bidirectional:
            model.st(contractee_dvar <= 0)
            model.st(contractor_dvar >= 0)

        return contractee_dvar, contractor_dvar

    def get_volume_variables(
        self,
        model: ro.Model,
        start_time: np.datetime64,
        end_time: np.datetime64,
        timestep: np.timedelta64,
    ):
        assert isinstance(
            (end_time - start_time) / timestep, (int, np.int32)
        ), "End Time and Start time don't align with timestep"

        current_time = start_time

        contractor_dvar = []
        contractee_dvar = []

        while current_time != end_time:
            contractee_dvar_t, contractor_dvar_t = self.get_volume_variable(
                model, current_time
            )
            contractee_dvar.append(contractee_dvar_t)
            contractor_dvar.append(contractor_dvar_t)

        return contractee_dvar, contractor_dvar

    def get_schedule(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        contractor_schedule = self.contractor.get_schedule(
            start_time, end_time, model, apply_constraints, state=state
        )
        contractee_schedule = self.contractee.get_schedule(
            start_time, end_time, model, apply_constraints, state=state
        )

        if contractor_schedule == {}:
            return contractee_schedule
        elif contractee_schedule == {}:
            return contractor_schedule
        else:
            if len(contractor_schedule.keys()) == len(
                contractee_schedule.keys()
            ):
                scheduled = OrderedDict()
                try:
                    for (
                        scheduled_time,
                        contractor_actions,
                    ) in contractor_schedule.items():
                        contractee_actions = contractee_schedule[scheduled_time]
                        scheduled[scheduled_time] = {
                            self.contractor.id: contractor_actions,
                            self.contractee.id: contractee_actions,
                        }

                        try:
                            contractor_volume = (
                                contractor_actions[self.id]["volume"][0]
                                if "volume"
                                in contractor_actions[self.id].keys()
                                else contractor_actions[self.id][
                                    scheduled_time
                                ]["volume"][0]
                            )
                            contractee_volume = (
                                contractee_actions[self.id]["volume"][0]
                                if "volume"
                                in contractee_actions[self.id].keys()
                                else contractee_actions[self.id][
                                    scheduled_time
                                ]["volume"][0]
                            )

                            model.st(contractee_volume == -contractor_volume)
                            if not self.bidirectional:
                                model.st(
                                    contractee_actions[self.id]["volume"][0]
                                    <= 0.0
                                )
                        except Exception as e:
                            warnings.warn(
                                "Contract volume not being matched. Please be careful before proceeding. Exception: ",
                                e,
                            )

                    return scheduled
                except Exception as e:
                    raise Exception(
                        f"Schedule error for Contract: {self.id} \n Contractor: {self.contractor}\n {contractor_schedule.keys()} \n Contractee: {self.contractee}\n {contractee_schedule.keys()} \n Exception {e}"
                    )
            else:
                raise NotImplementedError(
                    f"Multiple schedules in a contract with uneven schedules not implemented yet :{(len(contractor_schedule), len(contractee_schedule))} Contract: {self.id} \n Contractor: {self.contractor}\n {contractor_schedule.keys()} \n Contractee: {self.contractee}\n {contractee_schedule.keys()}"
                )

    def get_target_variables(
        self,
        start_time: np.datetime64,
        end_time: np.datetime64,
        model: ro.Model,
    ) -> t.Dict:
        if self.transmission is not None:
            return (
                self.contractor.get_target_variables(
                    start_time, end_time, model, self.id
                ),
                self.contractee.get_target_variables(
                    start_time, end_time, model, self.id
                ),
                self.transmission.get_target_variables(
                    start_time, end_time, model
                ),
            )
        return (
            self.contractor.get_target_variables(
                start_time, end_time, model, self.id
            ),
            self.contractee.get_target_variables(
                start_time, end_time, model, self.id
            ),
        )

    def act(self, time, actions: t.Dict, train_flag: bool):
        (
            contractor_action,
            penalty_contractor,
            extra_contractor_info,
        ) = self.contractor.act(time, actions[self.contractor.id], train_flag)
        (
            contractee_action,
            penalty_contractee,
            extra_contractee_info,
        ) = self.contractee.act(time, actions[self.contractee.id], train_flag)

        self.log_action(
            time,
            {
                "contractor_action": contractor_action,
                "contractee_action": contractee_action,
            },
        )

        return (
            (contractor_action, contractee_action),
            (penalty_contractor, penalty_contractee),
            (extra_contractor_info, extra_contractee_info),
        )

    def get_entity(self, id) -> Entity:
        if id == self.contractor.id:
            return self.contractor
        elif id == self.contractee.id:
            return self.contractee
        else:
            raise KeyError(
                f"ID not found in contract: {self.id} - please check again."
            )

    def get_config(self):
        config = {}
        config[str(self.contractee)] = self.contractee.get_config()
        config[str(self.contractor)] = self.contractor.get_config()
        return config

    def log_action(self, time, action):
        if isinstance(time, t.List):
            for idx, ti in enumerate(time):
                self.action_log[ti] = action[idx]
        else:
            self.action_log[time] = action
