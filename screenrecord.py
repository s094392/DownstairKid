from time import sleep
import os
import numpy as np
import mss
import cv2
from DownEnv.DownConst import ACTIONS, TOP, LEFT, HEIGHT, WIDTH

def main():
    monitor = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('record.mp4', fourcc, 60 , (WIDTH, HEIGHT))
    for i in range(10000):
        print(i, 10000)
        sct = mss.mss()
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

if __name__ == "__main__":
    main()
