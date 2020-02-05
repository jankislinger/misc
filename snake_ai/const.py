import numpy as np
import pygame

WIDTH, HEIGHT = 32, 24
PX = 10
LEN = 3
SPEED = 1000

BLACK = 0, 0, 0
GRAY = 100, 100, 100
RED = 255, 0, 0
WHITE = 255, 255, 255

CHANN = 4

ACTION_ARRS = {
    pygame.K_UP: np.array([1, 0, 0, 0], np.float32),
    pygame.K_DOWN: np.array([0, 1, 0, 0], np.float32),
    pygame.K_RIGHT: np.array([0, 0, 1, 0], np.float32),
    pygame.K_LEFT: np.array([0, 0, 0, 1], np.float32),
}

ACTIONS = list(ACTION_ARRS)
NUM_ACTIONS = len(ACTIONS)
