import cv2
from DownEnv.DownGame import DownGame

if __name__ == "__main__":
    game = DownGame()
    while True:
        game.take_action(2)
        cv2.imshow("image", game.screenshot)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
