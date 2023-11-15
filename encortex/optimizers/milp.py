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
from types import ModuleType


import numpy as np
from rsome import ro
from rsome.lp import Vars


import encortex
from encortex.env import EnCortexEnv
from encortex.optimizer import Optimizer


logger = logging.getLogger(__name__)


def get_solver(solver: str) -> ModuleType:
    """Integer Linear Programming solver (via RSOME)

    Currently, we support Gurobi: 'grb', OR-Tools: 'ort'

    Args:
        solver (str): Name of the solver

    Returns:
        _type_: _description_
    """
    if solver == "grb":
        from rsome import grb_solver

        return grb_solver
    elif solver == "ort":
        from rsome import ort_solver

        return ort_solver


class MILPOptimizer(Optimizer):
    """MILP Optimizer"""

    def __init__(
        self, env: EnCortexEnv, solver: str = "grb", obj: str = "min"
    ) -> None:
        """MILP Optimizer for an MILP EnCortexEnvironment

        Args:
            env (EnCortexEnv): EnCortex environment
            solver (str, optional): Name of the solver to use. See :func:`encortex.optimizers.milp.get_solver` for supported solvers. Defaults to 'ort'.
        """
        super().__init__(env)

        self.solver = get_solver(solver)
        encortex.set_optimizer(solver)

        assert obj in [
            "min",
            "max",
        ], "Wrong objective passed. Expected 'min' or 'max'"
        self.obj = obj

    def predict(
        self, state: t.Dict, variables: t.Dict, model: ro.Model = None
    ) -> t.Tuple[float, t.Dict, t.Dict]:
        """_summary_

        Args:
            state (t.Dict): _description_
            variables (t.Dict): _description_

        Returns:
            t.Tuple[float, t.Dict, t.Dict]: Returns the value of the objective function, the state and the variables
        """
        if model is None:
            model = ro.Model()
            logging.warn("Warning: creating a new model")

        objective = self.env.get_objective_function(state, variables, model)
        if self.obj == "min":
            model.min(objective.sum())
        elif self.obj == "max":
            model.max(objective.sum())
        model.solve(
            self.solver,
            display=str(logger.level).upper() in ["DEBUG", "ERROR", "CRITICAL"],
        )
        logger.info(f"Model get: {model.get()}")

        variables_values = self.modify(variables)
        objective = self.env.get_objective_function(state, variables_values)
        return objective, state, variables

    def modify(self, variables: t.Dict):
        for cid in variables.keys():
            for time, actions in variables[cid].items():
                storage_id = None
                for entity_id in actions.keys():
                    for contract_id, contract_actions in actions[
                        entity_id
                    ].items():
                        for (
                            contract_action_variable_name,
                            contract_action_variables,
                        ) in contract_actions.items():
                            if isinstance(contract_action_variables, t.List):
                                contract_actions[
                                    contract_action_variable_name
                                ] = self._parse_mod_list(
                                    contract_action_variables
                                )
                            elif isinstance(contract_action_variables, t.Dict):
                                contract_actions[
                                    contract_action_variable_name
                                ] = self._parse_mod_dict(
                                    contract_action_variables
                                )
                            elif isinstance(contract_action_variables, Vars):
                                contract_actions[
                                    contract_action_variable_name
                                ] = self._parse_mod_vars(
                                    contract_action_variables
                                )
                            else:
                                raise NotImplementedError(
                                    "Modify: not supported: ",
                                    contract_action_variables,
                                )

        return variables

    def _parse_mod_list(self, variables: t.List):
        for idx, i in enumerate(variables):
            value = i
            if isinstance(i, t.List):
                value = self._parse_mod_list(i)
            elif isinstance(i, t.Dict):
                value = self._parse_mod_dict(i)
            elif isinstance(i, Vars):
                value = self._parse_mod_vars(i)
            variables[idx] = value
        return variables

    def _parse_mod_dict(self, variables: t.Dict):
        for key, value in variables.items():
            i = value
            if isinstance(i, t.List):
                value = self._parse_mod_list(i)
            elif isinstance(i, t.Dict):
                value = self._parse_mod_dict(i)
            elif isinstance(i, Vars):
                value = self._parse_mod_vars(i)
            variables[key] = value
        return variables

    def _parse_mod_vars(self, variable: Vars):
        return variable.get()

    def train(self, *args, **kwargs):
        pass

    def run(self, time: np.datetime64) -> t.Tuple[float, float, np.datetime64]:
        """The running sequence for an MILP Environment and Optimizer

        Args:
            time (np.datetime64): Time to run the environment at.

        Returns:
            t.Tuple[float, t.Dict, t.Dict]: Objective function value, Reward of the action and the next timestamp to take the action at
        """
        state = self.env.get_state()
        model = ro.Model()
        actions = self.env.get_action_space(time, model=model, state=state)
        objective, state, _ = self.predict(state, actions, model)

        reward, time, done = self.env.step(actions)
        self.time = self.env.time

        return objective, reward, time, done

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass
