import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from snake_ai.const import WIDTH, HEIGHT, CHANN, NUM_ACTIONS


class Agent:
    model: keras.models.Model

    def __init__(self, model, mem_size=1000):
        self.model = model
        self.eps = 0.2
        self.action_space = list(range(5))
        self.mem_size = mem_size

        self.idx = 0

        self.state_mem = np.zeros((mem_size, WIDTH, HEIGHT, CHANN))
        self.new_state_mem = np.zeros((mem_size, WIDTH, HEIGHT, CHANN))
        self.action_mem = np.zeros((mem_size,), dtype=int)
        self.reward_mem = np.zeros((mem_size,))
        self.terminal_mem = np.zeros((mem_size,))

        self.last_state = None
        self.last_action = None

        self.screen = None

    def get_action(self, state, reward, terminal, interactive):
        if self.last_state is not None and self.last_action is not None:
            self.store_transition(self.last_state, self.last_action, reward, state, terminal)
        self.last_state = state
        self.last_action, q = self.choose_action(state, interactive)
        return self.last_action, q

    def store_transition(self, state, action, reward, new_state, terminal):
        idx = self.idx % self.mem_size

        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.new_state_mem[idx] = new_state
        self.terminal_mem[idx] = float(terminal)

        self.idx += 1
        if self.idx % self.mem_size == 0:
            self.update_model(self.idx // self.mem_size)
            self.reset_memory()
            self.eps *= 0.999
            print(f"new epsilon: {self.eps}")

    def reset_memory(self):
        self.state_mem[:] = 0.0
        self.action_mem[:] = 0
        self.reward_mem[:] = 0.0
        self.new_state_mem[:] = 0.0
        self.terminal_mem[:] = 0.0

    def choose_action(self, state, interactive):
        if random.random() < self.eps:
            return random.randrange(NUM_ACTIONS), None

        pred = self.model.predict(state[np.newaxis, ...])[0]
        # if self.screen is not None:
        #     for i, (a, q) in enumerate(zip(['UP', 'DN', 'RT', 'LT'], pred)):
        #         msg = f"{a}: {q:.3f}"
        #         font = pygame.font.Font('freesansbold.ttf', 9)
        #         text = font.render(msg, True, BLACK)
        #         rect = text.get_rect()
        #         rect.center = rect.center[0], 2 + rect.center[1] + 10 * i
        #         self.screen.blit(text, rect)
        #     pygame.display.update()
        #
        #     # wait = True
        #     # while wait:
        #     #     for ev in pygame.event.get(pygame.KEYDOWN):
        #     #         if ev.key == pygame.K_SPACE:
        #     #             wait = False
        #     # input()

        return np.argmax(pred), np.max(pred)

    def update_model(self, epoch):
        print('updating model')
        gamma = 0.99

        unique, counts = np.unique(self.reward_mem, return_counts=True)
        print(np.array([unique, counts]))

        q_max = np.max(self.model.predict(self.new_state_mem), axis=1)
        learned_value = self.reward_mem + gamma * q_max * (1 - self.terminal_mem)

        q = self.model.predict(self.state_mem)
        q[np.arange(self.mem_size), self.action_mem] = learned_value

        losses = []
        for _ in range(5):
            loss = self.model.train_on_batch(self.state_mem, q)
            losses.append(str(loss))
        print('losses:', ','.join(losses))

        save_model(self.model, epoch)

    def set_screen(self, screen):
        self.screen = screen


def get_model():
    x = input_board = tf.keras.layers.Input((WIDTH, HEIGHT, CHANN))
    for filters, kernel_size in [(16, 5), (16, 3), (32, 3)]:
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(NUM_ACTIONS)(x)

    output = x

    model = keras.models.Model(input_board, output)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MSE,
        metrics=[])

    model.summary()
    return model


def load_model(epoch):
    model = get_model()
    model.load_weights(f"snake_weights_{epoch:05d}.h5")
    return model


def save_model(model, epoch):
    model.save_weights(f"snake_weights_{epoch:05d}.h5")
