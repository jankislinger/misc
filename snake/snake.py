import random
from typing import Tuple, List

import numpy as np
import pygame
import scipy.special
# from tensorflow.python import keras

WIDTH, HEIGHT = 32, 24
PX = 10
LEN = 3
SPEED = 100

BLACK = 0, 0, 0
RED = 255, 0, 0
WHITE = 255, 255, 255

LR = 0.5
DF = 0.95
RANDOM_PROB = 0.05

CHANN = 4

Position = Tuple[int, int]

ACTION_ARRS = {
    None: np.array([0, 0, 0, 0], np.float32),
    pygame.K_UP: np.array([1, 0, 0, 0], np.float32),
    pygame.K_DOWN: np.array([0, 1, 0, 0], np.float32),
    pygame.K_RIGHT: np.array([0, 0, 1, 0], np.float32),
    pygame.K_LEFT: np.array([0, 0, 0, 1], np.float32),
}

ACTIONS = list(ACTION_ARRS)


class Snake:
    score: int
    head: Position
    direction: Position
    direction_key: int
    tail: List[Position]
    treat: Position
    current_run: int = 0

    def __init__(self):
        # self.model = get_model()
        self.clock = pygame.time.Clock()
        self.max_score = 0
        self.screen = pygame.display.set_mode((PX * WIDTH, PX * HEIGHT))
        self.running = True
        self.last_predicted_q = 0

        self.reset_snake()

        self.reward = 0
        self.last_board_arr = None
        self.last_action = None
        self.board_arr = self.as_array()
        self.steps = []

    def play(self):
        while self.running:
            self.update()
            self.clock.tick(SPEED)

    def reset_snake(self):
        print(f"died after {self.current_run} steps, predicted q = {self.last_predicted_q}")
        self.score = 0
        self.head = WIDTH // 2, HEIGHT // 2
        self.direction = 1, 0
        self.direction_key = pygame.K_RIGHT
        self.tail = []
        for i in range(1, LEN + 1):
            self.tail.append((self.head[0] - i * self.direction[0], self.head[1] - i * self.direction[1]))
        self.place_treat()
        self.current_run = 0

    def update(self):
        self.change_direction()
        self.move()
        self.draw()
        # self.update_model()

    def change_direction(self):
        key = self.read_key()
        if key is None:
            self.last_action = None
            return

        if key == pygame.K_LEFT and self.direction_key != pygame.K_RIGHT:
            self.direction_key = self.last_action = key
            self.direction = (-1, 0)
        elif key == pygame.K_RIGHT and self.direction_key != pygame.K_LEFT:
            self.direction_key = self.last_action = key
            self.direction = (1, 0)
        elif key == pygame.K_UP and self.direction_key != pygame.K_DOWN:
            self.direction_key = self.last_action = key
            self.direction = (0, -1)
        elif key == pygame.K_DOWN and self.direction_key != pygame.K_UP:
            self.direction_key = self.last_action = key
            self.direction = (0, 1)
        else:
            self.last_action = None

    def move(self):
        self.last_board_arr = self.board_arr

        self.tail.insert(0, self.head)
        self.head = self.head[0] + self.direction[0], self.head[1] + self.direction[1]
        reward = 0

        if self.head == self.treat:
            self.place_treat()
            reward = 1
            self.score += 1
            if self.score > self.max_score:
                print('increasing max score to', self.score)
                self.max_score = self.score
        else:
            self.tail.pop(-1)

        if not 0 <= self.head[0] < WIDTH or not 0 <= self.head[1] < HEIGHT or self.head in self.tail:
            self.reset_snake()
            reward = -10

        self.reward = reward
        self.current_run += 1
        self.board_arr = self.as_array()

    def place_treat(self):
        while True:
            treat = random.randrange(WIDTH), random.randrange(HEIGHT)
            if treat != self.head and treat not in self.tail:
                self.treat = treat
                return

    def read_key(self):
        self.save_transition()
        arrow_keys = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
        key = None
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or (event.type == pygame.QUIT):
                self.running = False
            if event.type == pygame.KEYDOWN and event.key in arrow_keys:
                key = event.key
        return key

    def read_key_ai(self):
        self.save_transition()

        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or (event.type == pygame.QUIT):
                self.running = False

        state = np.repeat(self.board_arr[np.newaxis, :, :, :], len(ACTIONS), axis=0)
        actions = np.stack([ACTION_ARRS[a] for a in ACTIONS])
        pred = self.model.predict([state, actions])[:, 0]
        probs = scipy.special.softmax(pred)
        k = np.random.choice(len(probs), p=probs)
        self.last_predicted_q = pred[k]
        return ACTIONS[k]

    def save_transition(self):
        if self.last_board_arr is None:
            return

        self.steps.append({
            'state': self.last_board_arr,
            'action': self.last_action,
            'reward': self.reward,
            'new_state': self.as_array(),
        })

    def draw(self):
        self.screen.fill(WHITE)
        for x, y in [self.head] + self.tail:
            pygame.draw.rect(self.screen, BLACK, (PX * x, PX * y, PX, PX))
        pygame.draw.rect(self.screen, RED, (PX * self.treat[0], PX * self.treat[1], PX, PX))
        pygame.display.set_caption(f"Snake {self.score} / {self.max_score}")
        pygame.display.flip()

    def as_array(self):
        body_array = np.zeros((WIDTH, HEIGHT))
        head_array = np.zeros((WIDTH, HEIGHT))
        treat_array = np.zeros((WIDTH, HEIGHT))
        frame_array = np.zeros((WIDTH, HEIGHT))

        head_array[self.head] = 1
        body_array[self.head] = 1

        for pos in self.tail:
            body_array[pos] = 1

        treat_array[self.treat] = 1

        for x in range(WIDTH):
            frame_array[x, 0] = 1
            frame_array[x, HEIGHT - 1] = 1

        for y in range(HEIGHT):
            frame_array[0, y] = 1
            frame_array[WIDTH - 1, y] = 1

        return np.stack([body_array, head_array, treat_array, frame_array], axis=2)

    def update_model(self):
        n = len(self.steps)
        if n < 1000:
            return

        new_states = np.stack([s['new_state'] for s in self.steps])
        new_states = np.repeat(new_states, 5, axis=0)
        reward = np.array([s['reward'] for s in self.steps])
        actions = np.repeat(np.stack([ACTION_ARRS[a] for a in ACTIONS]), n, axis=0)

        q = self.model.predict([new_states, actions])
        q_max = np.amax(q.reshape((n, 5)), axis=1)
        q_max = np.where(reward >= 0, q_max, 0)

        states = np.stack([s['state'] for s in self.steps])
        actions = np.stack([ACTION_ARRS[s['action']] for s in self.steps])

        old_value = self.model.predict([states, actions]).reshape((-1,))
        new_value = (1 - LR) * old_value + LR * (reward + DF * q_max)
        for i in range(3):
            print(self.model.train_on_batch([states, actions], new_value))

        self.steps = []


def get_model():
    x = input_board = keras.layers.Input((WIDTH, HEIGHT, CHANN))
    for _ in range(2):
        x = keras.layers.Conv2D(4, kernel_size=5)(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)

    input_actions = keras.layers.Input((4,))
    x = keras.layers.Concatenate()([x, input_actions])

    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(1)(x)

    output = x

    model = keras.models.Model([input_board, input_actions], output)
    model.compile(
        optimizer='adam',
        loss=keras.losses.MSE,
        metrics=[keras.metrics.accuracy])

    model.summary()
    return model


def main():
    pygame.init()
    snake = Snake()
    snake.play()
    pygame.quit()


if __name__ == '__main__':
    main()
