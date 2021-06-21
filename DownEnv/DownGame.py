import pdb
import mss
import time
import numpy as np
from time import sleep
from torchvision import transforms
import os
import cv2
import subprocess
from xvfbwrapper import Xvfb
from pynput.keyboard import Controller, Key
from multiprocessing import Pipe, Process
from transitions import Machine, State
from .DownConst import ACTIONS, TOP, LEFT, HEIGHT, WIDTH, LOSE_COLOR, LOSE_LOCATION, PLAYER_COLOR, PLAYFORM_COLOR, PIKE_COLOR, N_PLATFORMS


def down_client(conn):
    monitor = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}

    def press_and_release(keyboard, keys, holdtime=0.1):
        for key in keys:
            keyboard.press(key)

        time.sleep(holdtime)

        for key in keys:
            keyboard.release(key)

    def start_wine():
        subprocess.Popen([r"wine", r"down.exe"])
        time.sleep(4)
        echo_arg = os.environ['DISPLAY']
        echo = subprocess.run(
            ["echo", echo_arg],
            stdout=subprocess.PIPE,
            check=True,
        )

        display_id = echo.stdout[1:-1]
        print(int(display_id))

        xdp = subprocess.run(
            ["xdotool", "search", "--name", "NS-SHAFT"],
            stdout=subprocess.PIPE,
            check=True,
        )
        window_id = xdp.stdout
        # adjust window position
        cmd = ["xdotool", "windowmove", window_id, "0", "0"]
        subprocess.run(cmd, check=True)

    def start_game(keyboard, sct):
        img = np.array(sct.grab(monitor))
        conn.send(img)

    with Xvfb(width=WIDTH, height=WIDTH):
        # create instance in current desktop
        keyboard = Controller()
        sct = mss.mss()
        # initialize game
        start_wine()
        start_game(keyboard, sct)
        # wait actions
        while True:
            press_flow = conn.recv()
            if press_flow != "UPDATE":
                for keys in press_flow:
                    press_and_release(keyboard, keys=keys)
            img = np.array(sct.grab(monitor))
            conn.send(img)


class DownGame(object):
    def _spawn_down(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=down_client, args=(child_conn,))
        p.start()
        return parent_conn, parent_conn.recv()

    def __init__(self):
        self.actions = ACTIONS
        self.parent_conn, self.screenshot = self._spawn_down()
        self.FSM = self._init_FSM()
        # print(self.FSM.is_gaming())
        print("-- client ready")

    def take_action(self, idx):
        action = self.actions[idx]
        # print(self.FSM.state, action)
        self.parent_conn.send([action])
        self.screenshot = self.parent_conn.recv()

    def toggle_start(self):
        self.parent_conn.send([(Key.alt,)])
        self.parent_conn.send([("f",)])
        self.parent_conn.send([("n",)])
        # self.parent_conn.send([(Key.enter,)])
        self.screenshot = self.parent_conn.recv()
        self.screenshot = self.parent_conn.recv()
        self.screenshot = self.parent_conn.recv()
        while True:
            self._update_screenshot()
            if not (self.screenshot[LOSE_LOCATION[0]][LOSE_LOCATION[1]] == LOSE_COLOR).all():
                break
        self.FSM.play()


    def observe(self):
        done = False

        if (self.screenshot[LOSE_LOCATION[0]][LOSE_LOCATION[1]] == LOSE_COLOR).all() and self.FSM.is_gaming():
            self.FSM.wait()
            done = True
            return np.zeros(N_PLATFORMS*3+2), -1000, done

        player = np.where(self.screenshot == PLAYER_COLOR)
        pikes = np.where(self.screenshot == PLAYFORM_COLOR)
        platforms = np.where(self.screenshot == PIKE_COLOR)

        items = list()
        # 0 is empty, 1 is pikes, 2 is platforms
        for i in range(len(pikes[0])):
            items.append((pikes[0][i], pikes[1][i], 1))

        for i in range(len(platforms[0])):
            items.append((platforms[0][i], platforms[1][i], 1))

        if len(player[0]) > 0:
            self.player_pos = player[0][0], player[1][0]

        items.sort()
        items += [[0,0,0]] * N_PLATFORMS
        items = items[:N_PLATFORMS]
        items = np.asarray(items)

        result = np.asarray([self.player_pos[0], self.player_pos[1]])

        items = items.reshape(-1)

        result = np.concatenate((result, items)).astype("float32")


        img = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2GRAY)
        img = transforms.ToTensor()(img).float()

        if done:
            return result, 0, done
        else:
            return result, 1 + self.player_pos[0], done

    def _update_screenshot(self):
        self.parent_conn.send("UPDATE")
        self.screenshot = self.parent_conn.recv()

    def _init_FSM(self):
        states = [
            State(name='waiting', on_enter=[], on_exit=[]),
            State(name='gaming', on_enter=[], on_exit=[]),
        ]
        transitions = [
            {'trigger': 'wait', 'source': 'gaming', 'dest': 'waiting'},
            {'trigger': 'play', 'source': 'waiting', 'dest': 'gaming'},
        ]
        return DownGameState(states, transitions, 'waiting')


class DownGameState(object):
    def __init__(self, states, transitions, initial_state):
        self.machine = Machine(model=self,
                               states=states,
                               transitions=transitions,
                               initial=initial_state)
