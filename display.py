from time import sleep
import os
import mss
import cv2


def main():
    while True:
        sct = mss.mss()
        sct.shot()
        sleep(0.1)

if __name__ == "__main__":
    main()
