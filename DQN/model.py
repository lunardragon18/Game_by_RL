import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from  stable_baselines3.common.callbacks import BaseCallback

from environment.env import VEnv

class Logging(BaseCallback):
    def __init__(self,freq,save_path,verbose =1):
        super().__init__(verbose)
        self.freq = freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path:
            os.makedirs(self.save_path,exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.freq == 0 :
            model_path = os.path.join(self.save_path,'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

TRAIN_DIR = "trained_models"
LOGS = "Logs"
callback = Logging(1000,save_path=TRAIN_DIR)
env = DummyVecEnv([lambda: Monitor(VEnv())])
model = DQN("CnnPolicy",env,tensorboard_log=LOGS,verbose =1, buffer_size=100000,learning_starts=1000)
model.learn(total_timesteps=10000)




