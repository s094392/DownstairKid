from pynput.keyboard import Key

TOP = 0
LEFT = 0
WIDTH = 635
HEIGHT = 455
ORIGIN_WIDTH = 635
ORIGIN_HEIGHT = 455
ACTIONS = [
    (Key.left,),
    (Key.right,),
    ()
]

ORIGIN_LOSE_LOCATION = [225, 470]
LOSE_LOCATION = [225 - TOP, 470 - LEFT]
LOSE_COLOR = [200, 208, 212,   0]
