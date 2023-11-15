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
from collections import OrderedDict

import numpy as np
from gym import Env
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.base import LightningLoggerBase

from .callbacks import EnvCallback
from .decision_unit import DecisionUnit
from .logger import get_experiment_logger


class EnCortexEnv(Env):
    """Abstract class to an encortex environment"""

    registry = []

    def __init__(
        self,
        decision_unit: DecisionUnit,
        start_time: np.datetime64,
        seed: int = None,
        exp_logger: LightningLoggerBase = None,
        callbacks: t.List[EnvCallback] = [EnvCallback()],
        mode: str = "train",
    ) -> None:
        """EnCortex Environment initialization

        Args:
            decision_unit (DecisionUnit): Decision Unit from encortex network
            start_time (np.datetime64): Start time of the environment
            seed (int, optional): Set Seed of the environment. Defaults to None.
            exp_logger (LightningLoggerBase, optional): PyTorch Lightning experimental logger. Defaults to wandb.
            callbacks (t.List[EnvCallback], optional): EnCortex environment callback. Defaults to [EnvCallback()].
        """
        super().__init__()

        self.decision_unit = decision_unit
        self.decision_unit.set_time(start_time)

        self.start_time = start_time

        self.time = start_time
        if exp_logger is None:
            self.exp_logger = get_experiment_logger("wandb")
        else:
            self.exp_logger = exp_logger

        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.set_env(self)

        self.logger = OrderedDict()
        self.timestep = self.decision_unit.timestep
        self.mode = mode

        self.seed(seed)

    def _check_decision_unit_constraints(self, decision_unit: DecisionUnit):
        raise NotImplementedError

    def reset(self):
        for callback in self.callbacks:
            callback.on_before_reset()

        self.logger = OrderedDict()
        results = self._reset()

        for callback in self.callbacks:
            callback.on_after_reset()

        return results

    def _reset(self):
        raise NotImplementedError

    def step(
        self, action: t.Dict
    ) -> t.Tuple[t.Any, float, bool, t.Dict[str, t.Any]]:
        for callback in self.callbacks:
            callback.on_before_step(action)

        results = self._step(action)

        for callback in self.callbacks:
            callback.on_after_step(action, results)
        return results

    def _step(
        self, action: t.Dict
    ) -> t.Tuple[t.Any, float, bool, t.Dict[str, t.Any]]:
        raise NotImplementedError

    def get_state(self, *args, **kwargs):
        raise NotImplementedError

    def get_reward(self, *args, **kwargs):
        raise NotImplementedError

    def get_action_space(self, time: np.datetime64 = None, *args, **kwargs):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

    def get_reward_range(self):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def close(self):
        for callback in self.callbacks:
            callback.on_before_close()

        self._close()

        for callback in self.callbacks:
            callback.on_after_close()

    def _close(self):
        raise NotImplementedError

    def seed(self, seed: int = 40):
        seed_everything(seed)

    @property
    def action_space(self):
        """Action space of the environment

        Returns:
            <gym.spaces.Box>: the action space
        """
        d = {}
        return self.get_action_space()

    def is_done(
        self, start_time: np.datetime64 = None, end_time: np.datetime64 = None
    ):
        """Returns if current episode is done"""
        raise NotImplementedError

    @classmethod
    def register(cls=None, subcls=None):
        try:
            cls.registry.append(subcls)
        except:
            pass
        return subcls

    def get_objective_function(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def observation_space(self):
        return self.get_observation_space()

    def log(self, key: t.Any, value: t.Any):
        if key not in self.logger.keys():
            self.logger[key] = [value]
        else:
            self.logger[key].append(value)

    def get_log(self):
        return self.logger

    def log_experiment_metrics(self, key: str, value: t.Any):
        if self.exp_logger is not None:
            self.exp_logger.log_metrics({key: value})

    def log_experiment_hyperparameters(self, hyperparmaeters: t.Dict):
        if self.exp_logger is not None:
            self.exp_logger.log_hyperparams(hyperparmaeters)

    def export_config(self):
        decision_unit_config = self.decision_unit.get_config()
        config = {}
        config[f"{self.mode}/decision_unit"] = decision_unit_config
        config[f"{self.mode}/seed"] = str(self.seed)
        config[f"{self.mode}/start_time"] = str(self.start_time)
        config[f"{self.mode}/timestep"] = str(self.timestep)
        return config
