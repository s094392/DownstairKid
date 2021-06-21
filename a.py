import cv2

while True:
    try:
        img = cv2.imread("monitor-1.png")
        cv2.imshow("Down game", img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
    except:
        pass


