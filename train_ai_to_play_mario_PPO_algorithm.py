# Not working - in progress

import gymnasium as gym  # <-- Use this instead of 'import gym'
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, info = env.reset()  # Note: Gymnasium returns (obs, info) instead of just obs
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)  # Gymnasium uses 'terminated' and 'truncated'
    done = terminated or truncated
    env.render()

env.close()

# Note: The above code is a simple example of how to set up and run a Super Mario Bros environment using Gymnasium.