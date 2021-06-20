import gym
from DownEnv.DownEnv import DownEnv

env = DownEnv()
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while True:
        t += 1
        env.render()
        action = env.action_space.sample()
        reward, done = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
