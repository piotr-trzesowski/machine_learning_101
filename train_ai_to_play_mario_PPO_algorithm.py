import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

with gym_super_mario_bros.make('SuperMarioBros-1-1-v0') as env:
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

# Note: The above code is a simple example of how to set up and run a Super Mario Bros environment using Gymnasium.