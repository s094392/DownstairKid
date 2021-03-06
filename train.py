import os
import cv2
import gym
import math
import random
import torch
import torch.nn as nn
from torchvision import transforms
import time
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter
from DownEnv.DownEnv import DownEnv
import torch.optim as optim
from dqn import DQN
from DownEnv.DownConst import WIDTH, HEIGHT
import sys


MEAN_REWARD_BOUND = 99999999999999999999


gamma = 0.99
batch_size = 64
replay_size = 1000
learning_rate = 1e-4
sync_target_frames = 20
replay_start_size = 1000

eps_start = 0.02
eps_decay = .999985
eps_min = 0.02
filename = "1220_s3"


Experience = collections.namedtuple('Experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        try:
            return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), np.array(next_states)
        except:
            import pdb
            pdb.set_trace()



class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None


        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device).float()
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # self.env.render()
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            # self._reset()

        return done_reward


def train(device):
    env = DownEnv()
    obs_shape = env.observation_space.shape
    net = DQN([1, obs_shape[0], obs_shape[1]],
              env.action_space.n).to(device)
    target_net = DQN([1, obs_shape[0], obs_shape[1]],
                     env.action_space.n).to(device)
    net.load_state_dict(torch.load("1220"))
    target_net.load_state_dict(torch.load("1220"))
    writer = SummaryWriter(comment="-" + "Down")

    buffer = ExperienceReplay(replay_size)
    agent = Agent(env, buffer)

    epsilon = eps_start

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0

    best_mean_reward = None
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, 8 , (WIDTH, HEIGHT))
    highest_reward = 0

    while True:
        frame_idx += 1
        epsilon = max(epsilon*eps_decay, eps_min)

        if agent.env.game.FSM.is_waiting():
            agent._reset()

        while agent.env.game.FSM.is_waiting():
            pass


        reward = agent.play_step(net, epsilon, device=device)
        img = agent.env.game.screenshot
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        if reward is not None:
            out.release()

            if highest_reward < reward:
                os.rename("output.mp4", f"{int(reward)}_output.mp4")
                highest_reward = reward
                print("Saved highest_reward %.3f" % reward)

            total_rewards.append(reward)

            mean_reward = np.mean(total_rewards[-100:])

            print("%d:  %d games, mean reward %.3f, current reward %.3f, highest mean reward %.3f (epsilon %.2f)" % (
                frame_idx, len(total_rewards), mean_reward, reward, best_mean_reward or 0, epsilon))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), filename)
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            out = cv2.VideoWriter('output.mp4', fourcc, 8 , (WIDTH, HEIGHT))
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < replay_start_size:
            continue

        batch = buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch
        try:
            states = states.astype('float32')
        except:
            import pdb
            pdb.set_trace()

        states_v = torch.tensor(states).to(device).float()
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = net(states_v).gather(
            1, actions_v.unsqueeze(-1)).squeeze(-1)

        next_states_v = next_states_v.float()
        next_state_values = target_net(next_states_v).max(1)[0]

        next_state_values[done_mask] = 0.0

        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_v

        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        if frame_idx % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())

    writer.close()


if __name__ == "__main__":
    train("cuda")
