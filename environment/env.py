

import pydirectinput
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
from gymnasium import Env
from gymnasium.spaces import Box,Discrete
import time
import numpy as np
from mss import mss


class VEnv(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0,high=255,shape=(1,100,100),dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_capture = {"top" : 100,'left': 50, "width" :600,"height" : 300}
        self.game_over_capture = {'top' : 200 , 'left' : 300,"width" : 400,'height' : 50}

    def render(self):
        pass

    def step(self, action):

        mapping = {
            0 : "space",
            1 : "down",
            2 : "nothing"
        }

        if action != 2:
            pydirectinput.press((mapping[action]))

        done = self.get_over()
        next_state = self.next_state()

        reward = 2 if not done else  -1
        info = {}
        truncated = False
        return next_state, reward,done,truncated,info


    def close(self):
        pass
    def next_state(self):
        get = np.array(self.cap.grab(self.game_capture))[:,:,:3]
        grayscale = cv2.cvtColor(get,cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale,(100,100))
        resized = np.resize(resized,(1,100,100))
        return resized

    def get_over(self):
        done_capture = np.array(self.cap.grab(self.game_over_capture))
        done = False
        words = pytesseract.image_to_string(done_capture)[:4]
        if words == 'GAME':
            done = True
        return done

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        time.sleep(1)
        pydirectinput.press("space")
        return  self.next_state(),{}

