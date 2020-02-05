import random
from typing import List, Optional, Tuple

from snake_ai.agent import Agent, load_model
from snake_ai.const import *

Position = Tuple[int, int]


class Snake:
    agent: Optional[Agent]

    score: int
    head: Position
    direction: Position
    direction_key: int
    tail: List[Position]
    treat: Position
    current_run: int = 0

    def __init__(self, agent=None, use_screen=True):
        self.agent = agent

        self.clock = pygame.time.Clock()
        self.max_score = 0
        self.running = True
        self.last_predicted_q = 0
        self.interactive = True

        self.reset_snake()

        self.terminal = True
        self.reward = 0
        self.q = 0.0

        if use_screen:
            self.screen = pygame.display.set_mode((PX * WIDTH, PX * HEIGHT))

        if self.agent is not None and use_screen:
            self.agent.set_screen(self.screen)

    def play(self):
        while self.running:
            # self.clock.tick(SPEED)
            self.update()
            # time.sleep(1 / SPEED)

    def reset_snake(self):
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
        q = self.change_direction()
        if q is not None:
            self.q = q
        self.move()
        self.draw(self.q)

    def change_direction(self):
        if self.agent is not None:
            i, q = self.agent.get_action(self.state, self.reward, self.terminal, self.interactive)
            key = ACTIONS[i]
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_i:
                    self.interactive = not self.interactive
        else:
            q = None
            key = self.read_key()
            if key is None:
                return

        if key == pygame.K_LEFT and self.direction_key != pygame.K_RIGHT:
            self.direction_key = key
            self.direction = (-1, 0)
        elif key == pygame.K_RIGHT and self.direction_key != pygame.K_LEFT:
            self.direction_key = key
            self.direction = (1, 0)
        elif key == pygame.K_UP and self.direction_key != pygame.K_DOWN:
            self.direction_key = key
            self.direction = (0, -1)
        elif key == pygame.K_DOWN and self.direction_key != pygame.K_UP:
            self.direction_key = key
            self.direction = (0, 1)

        return q

    def move(self):
        self.tail.insert(0, self.head)
        self.head = self.head[0] + self.direction[0], self.head[1] + self.direction[1]
        self.reward = 0

        if self.head == self.treat:
            self.place_treat()
            self.score += 1
            self.reward = 1
            if self.score > self.max_score:
                print('increasing max score to', self.score)
                self.max_score = self.score
        else:
            self.tail.pop(-1)

        if not 0 <= self.head[0] < WIDTH or not 0 <= self.head[1] < HEIGHT or self.head in self.tail:
            self.terminal = True
            self.reward = -10
            self.reset_snake()
        else:
            self.terminal = False

        self.current_run += 1

    def place_treat(self):
        def rnd_treat():
            return random.randrange(WIDTH), random.randrange(HEIGHT)

        def is_valid(treat):
            return treat != self.head and treat not in self.tail

        def dist(treat):
            return sum(abs(x - y) for x, y in zip(self.head, treat))

        try:
            self.treat = min((t for t in (rnd_treat() for _ in range(100)) if is_valid(t)), key=dist)
        except ValueError:
            self.treat = rnd_treat()

    def read_key(self):
        if pygame.event.get(pygame.QUIT):
            self.running = False
            return

        arrow_keys = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
        key = None
        events = pygame.event.get(pygame.KEYDOWN)
        print('num events:', len(events))
        for event in events:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key in arrow_keys:
                print(event)
                key = event.key

        new_events = pygame.event.get(pygame.KEYDOWN)
        if new_events:
            print('new events:', len(new_events))
        return key

    def draw(self, q):
        self.screen.fill(WHITE)
        head_x, head_y = self.head
        pygame.draw.rect(self.screen, BLACK, (PX * head_x, PX * head_y, PX, PX))
        for x, y in self.tail:
            pygame.draw.rect(self.screen, GRAY, (PX * x, PX * y, PX, PX))
        pygame.draw.rect(self.screen, RED, (PX * self.treat[0], PX * self.treat[1], PX, PX))
        pygame.display.set_caption(f"Snake {self.score} / {self.max_score} | q = {q:.3f}")
        pygame.display.flip()

    @property
    def state(self):
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


def main():
    pygame.init()
    snake = Snake(Agent(load_model(0)))
    snake.play()
    pygame.quit()


if __name__ == '__main__':
    main()
