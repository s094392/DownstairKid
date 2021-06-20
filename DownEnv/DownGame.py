import pdb
import mss
import time
import numpy as np
import os
import cv2
import subprocess
from xvfbwrapper import Xvfb
from pynput.keyboard import Controller, Key
from multiprocessing import Pipe, Process
from transitions import Machine, State
from .DownConst import ACTIONS, HEIGHT, WIDTH, LOSE_COLOR


def down_client(conn):
    monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}

    def press_and_release(keyboard, keys, holdtime=0.01):
        for key in keys:
            keyboard.press(key)
        time.sleep(holdtime)

        for key in keys:
            keyboard.release(key)

    def start_wine():
        subprocess.Popen([r"wine", r"down.exe"])
        time.sleep(4)
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
        print(self.FSM.is_gaming())
        print("-- client ready")

    def take_action(self, idx):
        action = self.actions[idx]
        self.parent_conn.send([action])
        self.screenshot = self.parent_conn.recv()

    def toggle_start(self):
        self.parent_conn.send([(Key.alt,)])
        self.parent_conn.send([("f",)])
        self.parent_conn.send([("n",)])
        # self.parent_conn.send([(Key.enter,)])
        self.screenshot = self.parent_conn.recv()
        while True:
            self.take_action(2)
            if not (self.screenshot[225][470] == LOSE_COLOR).all():
                break
        self.FSM.play()

    def observe(self):
        done = False

        if (self.screenshot[225][470] == LOSE_COLOR).all() and self.FSM.is_gaming():
            self.FSM.wait()
            done = True

        img = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2GRAY)

        return img, 1, done

    def _update_screenshot(self):
        self.parent_conn.send([])
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
