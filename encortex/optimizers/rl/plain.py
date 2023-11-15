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

import os
import typing as t
import random

try:

    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    from tqdm import tqdm

from encortex.optimizer import Optimizer
from encortex.env import EnCortexEnv

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from torch import optim
from torch.distributions.categorical import Categorical

__all__ = ["DQN", "NoisyDQN", "PPO"]


class DQNQNetwork(nn.Module):
    def __init__(self, env, layer_config: t.List):
        super().__init__()

        network = [
            nn.Linear(
                np.array(env.observation_space.shape).prod(), layer_config[0]
            )
        ]
        for idx, config in enumerate(layer_config[1:]):
            network.append(nn.ReLU())
            network.append(nn.Linear(layer_config[idx], config))

        network.append(nn.Linear(layer_config[-1], env.action_space.n))
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class DQN(Optimizer):
    def __init__(
        self,
        env: EnCortexEnv,
        seed: int,
        exp_logger,
        torch_deterministic: bool = True,
        layer_config: t.List = [128, 64, 32],
        learning_rate: float = 1e-3,
        use_safety: bool = False,
        sample_strategy: str = "random",
        device: str = "cuda:0",
    ) -> None:
        super().__init__(env)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        self.q_network = DQNQNetwork(env, layer_config).to(device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate
        )
        self.target_network = DQNQNetwork(env, layer_config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            int(1e5),
            env.observation_space,
            env.action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        self.device = device
        self.exp_logger = exp_logger
        self.use_safety = use_safety

        assert sample_strategy in ["random", "linear"]
        self.sample_strategy = sample_strategy

    def train(
        self,
        total_timesteps: int,
        start_e: float,
        end_e: float,
        exploration_fraction: float,
        learning_starts: int,
        train_frequency: int,
        batch_size: int,
        gamma: float,
        target_network_frequency: int,
        tau: float,
    ):
        self.env.reset()
        obs = self.env.get_state()
        for global_step in tqdm(range(total_timesteps)):
            epsilon = self._linear_schedule(
                start_e,
                end_e,
                exploration_fraction * total_timesteps,
                global_step,
            )
            if random.random() < epsilon:
                actions = np.array(self.env.action_space.sample())
                mask = torch.from_numpy(self.env.get_action_mask())
                if mask.sum() > 0:
                    actions = Categorical(1 - mask).sample().cpu().numpy()
            else:
                q_values = self.q_network(
                    torch.Tensor(obs).to(self.device)
                ).reshape(1, -1)
                mask = torch.from_numpy(self.env.get_action_mask())
                if mask.sum() > 0:
                    actions = Categorical(1 - mask).sample().cpu().numpy()
                else:
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, rewards, dones, infos = self.env.step(actions)

            self.rb.add(obs, next_obs, actions, rewards, dones, infos)

            if dones:
                self.env.reset()

            self.exp_logger.log_metrics(
                {"charts/epsilon": epsilon}, global_step
            )

            if global_step > learning_starts:
                if global_step % train_frequency == 0:
                    if self.sample_strategy == "random":
                        data = self.rb.sample(batch_size)
                    elif self.sample_strategy == "linear":
                        if self.rb.full:
                            start = (
                                np.random.randint(
                                    1,
                                    self.rb.buffer_size - batch_size - 1,
                                    size=1,
                                )
                                + self.rb.pos
                            ) % self.rb.buffer_size
                        else:
                            start = np.random.randint(
                                0, self.rb.pos - batch_size - 1, size=1
                            )
                        batch_inds = np.arange(start, start + batch_size)
                        data = self.rb._get_samples(batch_inds)
                    with torch.no_grad():
                        target_max, _ = self.target_network(
                            data.next_observations
                        ).max(dim=1)
                        td_target = (
                            data.rewards.flatten()
                            + gamma * target_max * (1 - data.dones.flatten())
                        )
                    old_val = (
                        self.q_network(data.observations)
                        .gather(1, data.actions)
                        .squeeze()
                    )
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        self.exp_logger.log_metrics(
                            {"losses/td_loss": loss}, global_step
                        )
                        self.exp_logger.log_metrics(
                            {"losses/q_values": old_val.mean().item()},
                            global_step,
                        )

                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # update target network
                if global_step % target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        self.target_network.parameters(),
                        self.q_network.parameters(),
                    ):
                        target_network_param.data.copy_(
                            tau * q_network_param.data
                            + (1.0 - tau) * target_network_param.data
                        )

    def _linear_schedule(
        self, start_e: float, end_e: float, duration: int, t: int
    ):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def predict(self, state, mask: np.ndarray = None):
        q_values = self.q_network(torch.Tensor(state).to(self.device)).reshape(
            1, -1
        )
        if mask is not None:
            if mask.sum() > 0:
                q_values[:, mask] = -1e5
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions

    def run(self, time: np.datetime64) -> t.Tuple[float, float, np.datetime64]:
        state = self.env.get_state()

        action = self.predict(state, self.env.get_action_mask())
        if isinstance(action, t.Tuple):
            action = action[0]

        next_state, reward, done, info = self.env.step(action)
        self.time = self.env.time

        return None, reward, self.time, done

    def save(self, directory: str, model_name: str, *args, **kwargs):
        if not model_name.endswith(".pth"):
            model_name += ".pth"
        path = os.path.join(directory, model_name)
        torch.save(self.q_network.state_dict(), path)
        return self.q_network

    def load(self, dirpath: str, name: str):
        if not name.endswith(".pth"):
            name += ".pth"
        q_network = torch.load(os.path.join(dirpath, name))
        self.q_network.load_state_dict(q_network)


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
        self, in_features: int, out_features: int, std_init: float = 0.5
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class NoisyQNetwork(DQNQNetwork):
    def __init__(self, env, layer_config: t.List):
        super().__init__(env, layer_config)

        network = [
            nn.Linear(
                np.array(env.observation_space.shape).prod(), layer_config[0]
            )
        ]
        for idx, config in enumerate(layer_config[1:]):
            network.append(nn.ReLU())
            network.append(NoisyLinear(layer_config[idx], config))

        network.append(NoisyLinear(layer_config[-1], env.action_space.n))
        self.network = nn.Sequential(*network)


class NoisyDQN(DQN):
    def __init__(
        self,
        env: EnCortexEnv,
        seed: int,
        exp_logger,
        torch_deterministic: bool = True,
        layer_config: t.List = [128, 64, 32],
        learning_rate: float = 0.001,
        use_safety: bool = False,
        sample_strategy: str = "random",
        device: str = "cuda:0",
    ) -> None:
        super().__init__(
            env,
            seed,
            exp_logger,
            torch_deterministic,
            layer_config,
            learning_rate,
            use_safety,
            sample_strategy,
            device,
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        self.q_network = NoisyQNetwork(env, layer_config).to(device)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate
        )
        self.target_network = NoisyQNetwork(env, layer_config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            int(1e5),
            env.observation_space,
            env.action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        self.device = device
        self.exp_logger = exp_logger
        self.use_safety = use_safety

    def train(
        self,
        total_timesteps: int,
        start_e: float,
        end_e: float,
        exploration_fraction: float,
        learning_starts: int,
        train_frequency: int,
        batch_size: int,
        gamma: float,
        target_network_frequency: int,
        tau: float,
    ):
        self.env.reset()
        obs = self.env.get_state()
        for global_step in tqdm(range(total_timesteps)):
            epsilon = self._linear_schedule(
                start_e,
                end_e,
                exploration_fraction * total_timesteps,
                global_step,
            )
            if random.random() < epsilon:
                actions = np.array(self.env.action_space.sample())
                if self.use_safety:
                    mask = torch.from_numpy(self.env.get_action_mask())
                    if mask.sum() > 0:
                        actions = Categorical(1 - mask).sample().cpu().numpy()
            else:
                q_values = self.q_network(
                    torch.Tensor(obs).to(self.device)
                ).reshape(1, -1)
                if self.use_safety:
                    mask = torch.from_numpy(self.env.get_action_mask())
                    if mask.sum() > 0:
                        actions = Categorical(1 - mask).sample().cpu().numpy()
                    else:
                        actions = torch.argmax(q_values, dim=1).cpu().numpy()
                else:
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, rewards, dones, infos = self.env.step(actions)

            self.rb.add(obs, next_obs, actions, rewards, dones, infos)

            if dones:
                self.env.reset()

            self.exp_logger.log_metrics(
                {"charts/epsilon": epsilon}, global_step
            )

            if global_step > learning_starts:
                if global_step % train_frequency == 0:
                    if self.sample_strategy == "random":
                        data = self.rb.sample(batch_size)
                    elif self.sample_strategy == "linear":
                        if self.rb.full:
                            start = (
                                np.random.randint(
                                    1, self.rb.buffer_size - batch_size, size=1
                                )
                                + self.rb.pos
                            ) % self.rb.buffer_size
                        else:
                            start = np.random.randint(
                                0, self.rb.pos - batch_size, size=1
                            )
                        batch_inds = np.arange(start, start + batch_size)
                        data = self.rb._get_samples(batch_inds)
                    with torch.no_grad():
                        target_max, _ = self.target_network(
                            data.next_observations
                        ).max(dim=1)
                        td_target = (
                            data.rewards.flatten()
                            + gamma * target_max * (1 - data.dones.flatten())
                        )
                    old_val = (
                        self.q_network(data.observations)
                        .gather(1, data.actions)
                        .squeeze()
                    )
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        self.exp_logger.log_metrics(
                            {"losses/td_loss": loss}, global_step
                        )
                        self.exp_logger.log_metrics(
                            {"losses/q_values": old_val.mean().item()},
                            global_step,
                        )

                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # update target network
                if global_step % target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(
                        self.target_network.parameters(),
                        self.q_network.parameters(),
                    ):
                        target_network_param.data.copy_(
                            tau * q_network_param.data
                            + (1.0 - tau) * target_network_param.data
                        )

    def _linear_schedule(
        self, start_e: float, end_e: float, duration: int, t: int
    ):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self, env, actor_layer_config: t.List, critic_layer_config: t.List
    ):
        super().__init__()

        critic_network = [
            nn.Linear(
                env.observation_space.shape.prod(), critic_layer_config[0]
            )
        ]
        for idx, config in enumerate(critic_layer_config[1:]):
            critic_network.append(nn.ReLU())
            critic_network.append(nn.Linear(critic_layer_config[idx], config))
        self.critic = nn.Sequential(*critic_network)

        actor_network = [
            nn.Linear(env.observation_space.shape.prod(), actor_layer_config[0])
        ]
        for idx, config in enumerate(actor_layer_config[1:]):
            actor_network.append(nn.ReLU())
            actor_network.append(nn.Linear(actor_layer_config[idx], config))
        self.actor = nn.Sequential(*critic_network)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPO(Optimizer):
    pass
