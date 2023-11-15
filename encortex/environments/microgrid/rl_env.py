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
from time import time as stime

import numpy as np
from gym import spaces
from rsome import ro  # isort:skip
from rsome.lp import Vars  # isort:skip
from pytorch_lightning.loggers import LightningLoggerBase  # isort:skip

from encortex.callbacks import EnvCallback
from encortex.decision_unit import DecisionUnit
from encortex.microgrid import Microgrid
from encortex.env import EnCortexEnv
from encortex.utils.time_utils import tuple_to_np_timedelta
from rsome import grb_solver as grb
from rsome import ort_solver as grb

from .milp_env import MicrogridMILPEnv

logger = logging.getLogger(__name__)


@EnCortexEnv.register
class MicrogridRLEnv(MicrogridMILPEnv):
    """Microgrid RL Environment inherits the Microgrid MILP environment.

    Action space: Discrete(3) to match the discrete storage devices
    State/Observation space: Vectorized form of all the forecasts and soc
    """

    def __init__(
        self,
        decision_unit: DecisionUnit,
        start_time: np.datetime64,
        seed: int = None,
        exp_logger: LightningLoggerBase = None,
        callbacks: t.List[EnvCallback] = ...,
        mode: str = "train",
        action_window: np.timedelta64 = ...,
        future_window: np.timedelta64 = np.timedelta64(24, "h"),
        use_safety: bool = False,
    ) -> None:
        self.future_window = tuple_to_np_timedelta(future_window)
        super().__init__(
            decision_unit,
            start_time,
            seed,
            exp_logger,
            callbacks,
            mode,
            action_window,
        )
        self.start_time = start_time
        self.objective_cost_total = []
        self.penalties = []
        self.rewards = []

        self.use_safety = use_safety

    @property
    def action_space(self):
        return spaces.Discrete(3)

    def get_observation_space(self):
        return spaces.Box(
            low=-1e4,
            high=1e4,
            shape=self.decision_unit.get_state(
                self.time,
                self.time + self.future_window,
                vectorize=True,
            ).shape,
        )

    def get_state(
        self,
        start_time: np.datetime64 = None,
        vectorize: bool = True,
        type="forecast",
    ):
        if start_time is None:
            start_time = self.time
        return self.decision_unit.get_state(
            start_time,
            start_time + self.future_window,
            vectorize=vectorize,
            type=type,
        )

    def predict(
        self,
        state: t.Dict,
        variables: t.Dict,
        model: ro.Model,
        prune: bool = False,
    ):
        objective = self.get_objective_function(state, variables)
        model.max(objective)
        model.solve(
            solver=grb,
            display=str(logger.level).upper() in ["DEBUG", "ERROR", "CRITICAL"],
        )

        variables = self.modify(variables)
        objective = self.get_objective_function(state, variables)

        if prune:
            time_idx_max = self.action_window / self.timestep
            for cid in variables.keys():
                time_idx = 0
                for time, actions in variables[cid].items():
                    if time_idx >= time_idx_max:
                        del variables[cid][time]

        return objective, state, variables

    def modify(self, variables: t.Dict):
        # variables = copy.deepcopy(variables)
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

    def get_action_space(
        self,
        time: np.datetime64,
        model: ro.Model = None,
        apply_constraints: bool = True,
        state: t.Dict = None,
    ):
        if model is None:
            model = ro.Model()
            logger.warn("Creating new model, beware")
        return self.decision_unit.get_schedule(
            time,
            time + self.action_window,
            model,
            apply_constraints,
            state,
        )

    def _step(
        self, action: int
    ) -> t.Tuple[t.Any, float, bool, t.Dict[str, t.Any]]:
        ti = stime()
        state = self.get_state(vectorize=False)
        # print("State fetch: ", stime() - ti)
        ti = stime()
        logger.info(f"Action: {action} Time: {self.time}")
        logger.info(f"State: {state}")
        model = ro.Model()
        actions = self.get_action_space(
            self.time, model, apply_constraints=False, state=state
        )

        for cid in actions.keys():
            for time, e_actions in actions[cid].items():
                contract = self.decision_unit.get_contract(cid)
                microgrid = (
                    contract.contractor
                    if isinstance(contract.contractor, Microgrid)
                    else contract.contractee
                )
                storage_devices = list(
                    e_actions[microgrid.id]["all"]["volume"][1][
                        "storage_devices"
                    ].keys()
                )
                for storage_device in storage_devices:
                    e_actions[microgrid.id]["all"]["volume"][1][
                        "storage_devices"
                    ][storage_device]["volume"][
                        1
                    ] = storage_device.action.transform_variables(
                        "rl",
                        action,
                        model,
                        e_actions[microgrid.id]["all"]["volume"][1][
                            "storage_devices"
                        ][storage_device]["volume"][1],
                    )
        # print("Variable set: ", stime() - ti)
        ti = stime()
        objective, state, actions = self.predict(state, actions, model, True)
        # print("Predict: ", stime() - ti)
        ti = stime()
        results = self.decision_unit.act(self.time, actions, True)
        penalty = self._get_penalties(results)
        actual_state = self.get_state(type="actual", vectorize=False)

        logger.info(f"Actual State: {actual_state} \n Forecast State: {state}")
        logger.info(f"Penalty: {penalty}")
        self.penalties.append(penalty)
        self.log_experiment_metrics(f"{self.mode}/penalty", penalty)
        # print("Post Predict: ", stime() - ti)
        ti = stime()
        _, (
            price_buys,
            price_sells,
            generations,
            demands,
            objectives,
            grid_vs,
        ) = self.get_objective_function(actual_state, actions, return_info=True)
        logger.info(f"COST SAVINGS: {objectives}")
        self.objective_cost_total.append(sum(objectives))
        reward = (
            self.get_objective_function(actual_state, actions) / 1e3 - penalty
        )
        self.rewards.append(reward)
        # print("Objective time: ", stime() - ti)
        ti = stime()

        next_state = self.get_state(vectorize=True)
        done = self.is_done(self.time, self.time + self.future_window)
        info = {}
        info["state"] = state
        logger.info(f"DONE: {done}")
        if done:
            logger.info(
                f"TOTAL COST SAVINGS:  {np.asarray(self.objective_cost_total).sum()}"
            )
            logger.info(f"TOTAL REWARDS: {sum(self.rewards)}")
            logger.info(f"TOTAL PENALTIES: {sum(self.penalties)}")
            self.log_experiment_metrics(
                f"{self.mode}/total_cost_savings",
                sum(self.objective_cost_total),
            )
            self.log_experiment_metrics(
                f"{self.mode}/total_rewards", sum(self.rewards)
            )
            self.log_experiment_metrics(
                f"{self.mode}/total_penalties", sum(self.penalties)
            )
        logger.info(f"Objective: {objective} | Reward: {reward}")
        self.log_experiment_metrics(f"{self.mode}/objective", objective)
        self.log_experiment_metrics(f"{self.mode}/reward", reward)

        self.time += self.get_schedule_timestep()
        self.decision_unit.set_time(self.time)
        return next_state, reward, done, info

    def _reset(self):
        logger.info("RESET")
        self.objective_cost_total = []
        self.rewards = []
        self.penalties = []
        self.time = self.start_time
        return self.get_state()

    def export_config(self):
        config = super().export_config()
        config[f"{self.mode}/action_window"] = self.action_window
        config[f"{self.mode}/future_window"] = self.future_window
        return config

    def get_action_mask(self):
        if self.use_safety:
            return (
                self.decision_unit.microgrids[0]
                .storage_devices[0]
                .get_action_mask()
            )
        else:
            return np.zeros(self.action_space.n)
