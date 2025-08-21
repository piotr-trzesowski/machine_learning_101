import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# dependency hell
# pip install setuptools==65.5.0 "wheel<0.40.0"
# pip install gym==0.21.0
# pip install gym-super-mario-bros==7.3.0
# pip install stable-baselines3[extra]==1.6.0
# pip install 'numpy<2.0.0'

def make_env():
    def _init():
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        return env
    return _init

if __name__ == "__main__":
    env = DummyVecEnv([make_env()])
    try:
        model = PPO('CnnPolicy', env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save("ppo_mario")
    finally:
        env.close()

# Note: The above code is a simple example of how to set up and run a Super Mario Bros environment using Gymnasium.