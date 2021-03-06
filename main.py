import gym
from DownEnv.DownEnv import DownEnv

env = DownEnv()
for i_episode in range(20):
    observation = env.reset()
    env.render()
    t = 0
    while True:
        t += 1
        env.render()
        action = env.action_space.sample()
        result, reward, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
