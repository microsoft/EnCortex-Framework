from encortex.optimizer import Optimizer
from encortex.env import EnCortexEnv

from d3rlpy.algos import DiscreteSAC
import os
import torch


__all__ = ["DiscreteSACEnCortexRLOptimizer"]


class DiscreteSACEnCortexRLOptimizer(Optimizer):
    def __init__(
        self,
        env: EnCortexEnv,
        batch_size: int = 128,
        target_update_interval: int = 500,
    ) -> None:
        super().__init__(env)

        use_gpu = False
        if torch.cuda.device_count() > 0:
            use_gpu = True
        self.model = DiscreteSAC(
            batch_size=batch_size,
            use_gpu=use_gpu,
            target_update_interval=target_update_interval,
        )

    def train(self, iters: int):
        self.model.fit_online(self.env, n_steps=iters)

    def run(self, time):
        state = self.env.get_state()

        action = self.model.predict(state.reshape(1, -1))

        next_state, reward, done, info = self.env.step(action.reshape(-1))
        self.time = self.env.time

        return None, reward, self.time, done

    def save(self, directory: str, model_name: str, *args, **kwargs):
        self.model.save_model(os.path.join(directory, model_name))

    def load(self, dirpath: str, name: str):
        self.model.load_model(os.path.join(dirpath, name))
