import pdb
import mss
import time
import numpy as np
import os
import subprocess
# import cv2
from xvfbwrapper import Xvfb
from pynput.keyboard import Controller, Key
from multiprocessing import Pipe, Process
from DownEnv.DownConst import ACTIONS, HEIGHT, WIDTH


def down_client(conn, display_id):
    monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}

    def press_and_release(keyboard, keys, holdtime=0.01):
        for key in keys:
            keyboard.press(key)
        time.sleep(holdtime)

        for key in keys:
            keyboard.release(key)

    def start_wine():
        my_env = os.environ.copy()
        my_env["export DISPLAY"] = f":{display_id}"
        subprocess.Popen([r"wine", r"down.exe"], env=my_env)

    def start_game(keyboard, sct):
        press_and_release(keyboard, keys=(Key.enter, ))

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


class PikaGame(object):
    def _spawn_down(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=down_client, args=(child_conn, 3))
        p.start()
        return parent_conn, parent_conn.recv()

    def __init__(self):
        self.actions = ACTIONS
        self.parent_conn, self.screenshot = self._spawn_down()
        print("-- client ready")

    def take_action(self, idx):
        action = self.actions[idx]
        self.parent_conn.send([action])
        self.screenshot = self.parent_conn.recv()


if __name__ == "__main__":
    game = PikaGame()
    while True:
        game.take_action(2)
        # cv2.imshow("image", self.screenshot)
        # cv2.waitKey()
