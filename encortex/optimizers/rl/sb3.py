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
import os

import numpy as np
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from pytorch_lightning.loggers.base import LightningLoggerBase

from encortex.env import EnCortexEnv
from encortex.optimizer import Optimizer


try:

    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    from tqdm import tqdm

from stable_baselines3.common import base_class  # pytype: disable=pyi-error

logger = logging.getLogger(__name__)


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )
        self.pbar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(
            total=self.locals["total_timesteps"] - self.model.num_timesteps
        )

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()


def get_agent(name: str) -> BaseAlgorithm:
    """Supported and Tested agents from various RL frameworks.

    Currently, we only support ['A2C', 'DQN' and 'PPO']

    Args:
        name (str): Algorithm name

    Returns:
        BaseAlgorithm: class of the algorithm relevant to the name
    """
    agents = {
        "A2C": A2C,
        "DQN": DQN,
        "PPO": PPO,
    }

    assert name in list(
        agents.keys()
    ), f"Unsuppported Agent: {name}. Currently, only {list(agents.keys())} are supported"
    return agents[name]


@Optimizer.register
class EnCortexRLOptimizer(Optimizer):
    """EnCortex RL Optimizer

    Args:
        env (EnCortexEnv): An RL based EnCortex environment
        name (str): Name of the RL algorithm. Currently, DQN, A2C and PPO are supported
        policy (str): Type of Policy. Currently, only MlpPolicy is supported
        seed (int): seed for the experiment
        enable_progress_bar (bool, optional): Enable tqdm progress bar for training. Defaults to True.
        logger (LightningLoggerBase, optional): Logger from encortex. Defaults to None.
        logger_kwargs (t.Dict, optional): Logger key word args that interface with SB3. Defaults to {}.
        enable_checkpoint (bool, optional): Enable checkpoint functionality during training. Defaults to False.
        checkpoint_kwargs (t.Dict, optional): Checkpoint key word args for SB3. Defaults to {}.
    """

    def __init__(
        self,
        env: EnCortexEnv,
        name: str,
        policy: str,
        seed: int,
        enable_progress_bar: bool = True,
        exp_logger: LightningLoggerBase = None,
        logger_kwargs: t.Dict = {},
        enable_checkpoint: bool = False,
        checkpoint_kwargs: t.Dict = {},
        **kwargs,
    ) -> None:
        super().__init__(env)

        self.model: BaseAlgorithm = get_agent(str(name).upper())(
            policy=policy, env=env, seed=seed, **kwargs
        )
        self.callbacks = []

        self.enable_progress_bar = enable_progress_bar
        try:
            from pytorch_lightning.loggers import WandbLogger

            if isinstance(exp_logger, WandbLogger):
                from wandb.integration.sb3 import WandbCallback

                self.callbacks.append(WandbCallback(**logger_kwargs))
        except Exception as e:
            logger.warn("Couldn't attach Wandb Logger")
            from pytorch_lightning.loggers import MLFlowLogger
            from encortex.utils.mlflow_utils import MLflowOutputFormat
            from stable_baselines3.common.logger import Logger

            if isinstance(logger, MLFlowLogger):
                self.callbacks.append(Logger(MLflowOutputFormat()))

        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_kwargs = checkpoint_kwargs

    def train(self, *args, **kwargs):
        """Train the agent

        Args:
            *args : Arguments for the learn function for an EnCortex compatible algorithm (see :func:`encortex.optimizers.rl.get_agent`)
            **kwargs : Keyword-Arguments for the learn function for an EnCortex compatible algorithm (see :func:`encortex.optimizers.rl.get_agent`)
        """
        callbacks = []
        if self.enable_progress_bar:
            callbacks.append(ProgressBarCallback())
        if self.enable_checkpoint:
            callbacks.append(CheckpointCallback(**self.checkpoint_kwargs))

        if "callback" in kwargs.keys():
            kwargs["callback"] = CallbackList(
                [kwargs["callback"]] + self.callbacks + callbacks
            )
        else:
            kwargs["callback"] = CallbackList(callbacks)

        self.model.learn(*args, **kwargs)

    def save(self, directory: str, model_name: str, *args, **kwargs) -> None:
        """Saves the agent weights.

        Args:
            directory (str): directory where the agent weights should be saved at
            model name (str): model name
        """
        self.model.save(os.path.join(directory, model_name), *args, **kwargs)

    def load(self, dirpath: str, model_name: str) -> BaseAlgorithm:
        """Load the agent weights

        Args:
            dirpath (str): Path to directory where the weights were saved
            model_name (str): File name of the saved model(exclude .zip).

        Returns:
            BaseAlgorithm: RL Algorithm class
        """
        self.model = self.model.load(os.path.join(dirpath, model_name))
        return self.model

    def run(self, time: np.datetime64) -> t.Tuple[float, float, np.datetime64]:
        state = self.env.get_state()

        action = self.model.predict(state, deterministic=True)
        if isinstance(action, t.Tuple):
            action = action[0]

        next_state, reward, done, info = self.env.step(action)
        logger.info(f"Action: {action} | Reward: {reward} | Info: {info}")
        self.time = self.env.time

        return None, reward, self.time, done
