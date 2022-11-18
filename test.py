import sys
sys.path.append(r'C:\Users\nikii\AppData\Local\Programs\Python\Python310\Lib\site-packages')
import os
#import SMB
import gym_super_mario_bros
#import controls
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt

#create base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#simplify controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#Greyscale
env = GrayScaleObservation(env, keep_dim=True)
#wrap in dummy env
env = DummyVecEnv([lambda: env])
#stack frames
env = VecFrameStack(env, 4, channels_order='last')


#load model
model = PPO.load('./train/model_570000_steps.zip')
#start game
state = env.reset()
#loop through the game
while True:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

class TrainAndLoggingCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(TrainAndLoggingCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path
        def __init_callback(self):
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                model_path = os.path.join(self.save_path, 'model_{}_steps'.format(self.n_calls))
                self.model.save(model_path)
            return True
CHEKPOINT_DIR = './train/'
LOG_DIR = './logs/'
#setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHEKPOINT_DIR)

# #ai model started
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
# #start teaching ai model
# model.learn(total_timesteps=5000000, callback=callback)

#save_model
#model.save('best_model_so_far')
