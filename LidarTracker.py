import math
import numpy as np
import cv2

from constants import Constants


class Tracker():
    def __init__(self, height, width):
        self.numCalls = 0
        self.height = height
        self.width = width
        self.reflections = []
        self.gridDivisor = 5
        # safes the 3 lowest reflections in a lower grid
        # if a recognition close to that is seen the probability increases
        # [numRecognitions, probability, height in Image]
        self.grid = np.zeros((width // self.gridDivisor, 1, 2), dtype = np.float)



    def increaseCnt(self):
        self.numCalls += 1

    # adds a track to the grid in image coordinates
    def addTrack(self, u, v):
        if np.abs(self.grid[u // self.gridDivisor, 0, 2] - v) < 5:
            self.grid[u, 0, 1] += 1
            self.grid[u, 0, 2] = v
        self.grid[u, 1, 0] = 1 - 0.4 ** self.grid[u, v, 1]


