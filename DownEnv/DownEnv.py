import math
import gym
from time import sleep
from gym import spaces
import numpy as np
from cv2 import cv2 as cv
from .DownConst import HEIGHT, WIDTH
from .DownGame import DownGame


class DownEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        super(DownEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, 1), dtype=np.uint8)
        self.reward_range = (-math.inf, math.inf)
        self.action_space = spaces.Discrete(3)
        self.game = DownGame()

    def step(self, action):
        self.game.take_action(action)
        result, reward, done = self.game.observe()
        return result, reward, done, {}

    def reset(self):
        self.game.toggle_start()
        return np.random.random_sample((8,))

    def render(self, mode="human"):
        cv.imshow("Down game", self.game.screenshot)
        if cv.waitKey(25) & 0xFF == ord("q"):
            cv.destroyAllWindows()
