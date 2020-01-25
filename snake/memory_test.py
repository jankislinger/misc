import pygame

WIDTH, HEIGHT = 32, 24
PX = 10
LEN = 3
SPEED = 100

BLACK = 0, 0, 0
RED = 255, 0, 0
WHITE = 255, 255, 255

pygame.init()
screen = pygame.display.set_mode((PX * WIDTH, PX * HEIGHT))
clock = pygame.time.Clock()

head = 10, 15
tail = [(10 + i, 15 - i) for i in range(6)]
i = 0

while True:
    i += 1
    screen.fill(WHITE)
    for x, y in [head] + tail:
        pygame.draw.rect(screen, BLACK, (PX * x, PX * y, PX, PX))
    pygame.draw.rect(screen, RED, (PX * (i % WIDTH), PX * 3, PX, PX))
    pygame.display.set_caption("Snake")
    pygame.display.flip()

    clock.tick(SPEED)
