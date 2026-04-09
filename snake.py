import numpy as np
import pygame
from random import randint
from enum import Enum
from pygame import Vector2

TILE_SIZE = 20
WIDTH = 1280
HEIGHT = 720

COLORS = {
    "background": (20, 20, 20),
    "head": (18, 176, 60),
    "body": (25, 145, 57),
    "apple": (222, 27, 46),
    "border": (105, 105, 105)
}

borders = []

for i in range(WIDTH//TILE_SIZE//4):
    for j in range(HEIGHT//TILE_SIZE//4):
        if randint(0, 100)/ 100 > 0.95:
            borders.append(pygame.Rect(i*TILE_SIZE*4, j*TILE_SIZE*4, TILE_SIZE*4, TILE_SIZE*4))


class Directions(Enum):
    FORWARD = [1, 0, 0]
    LEFT = [0, 1, 0]
    RIGHT = [0, 0, 1]

class Snake:
    def __init__(self, px: int, py: int):
        self.body = [pygame.Rect(px + TILE_SIZE * i, py, TILE_SIZE, TILE_SIZE) for i in range(3)]
        self.facing = (-1, 0)

    def get_head(self):
        return self.body[0].copy()

    def get_next_head(self, facing):
        next_head = self.get_head()
        next_head.x = next_head.x + facing[0] * TILE_SIZE
        next_head.y = next_head.y + facing[1] * TILE_SIZE
        return next_head

    def get_new_facing(self, direction: list[int]):
        # directions
        # [1, 0, 0] STRAIGHT
        # [0, 1, 0] LEFT
        # [0, 0, 1] RIGHT

        #            LEFT      UP       RIGHT   DOWN
        clock_wise = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        idx = clock_wise.index(self.facing)

        # straight
        if np.array_equal(direction, [1, 0, 0]):
            new_facing = clock_wise[idx]

        # ccw rotation
        elif np.array_equal(direction, [0, 1, 0]):
            new_idx = (idx + 3) % 4
            new_facing = clock_wise[new_idx]
        # cw rotation
        else:
            new_idx = (idx - 3) % 4
            new_facing = clock_wise[new_idx]

        return new_facing

    def update_body_position(self):
        new_head = self.get_next_head(self.facing)

        self.body.insert(0, new_head)

        self.body.pop()

    def update_body_position_and_grow(self):
        new_head = self.get_next_head(self.facing)

        self.body.insert(0, new_head)

    def collision(self, head) -> bool:
        if head.right > WIDTH or head.left < 0 or head.bottom > HEIGHT or head.top < 0:
            return True

        if any(part.colliderect(head) for part in self.body[1:]):
            return True

        if any(part.colliderect(head) for part in borders):
            return True

        return False

    def draw(self, screen):
        for i, part in enumerate(self.body):
            color = COLORS["head"] if i == 0 else COLORS["body"]
            pygame.draw.rect(screen, color, part)

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.inactive_moves = 0
        self.made_move = False
        self.snake = Snake(WIDTH // 2, HEIGHT // 2)
        self.apple: pygame.Rect = None
        self.SNAKE_MOVE_EVENT = pygame.USEREVENT + 1
        self.spawn_apple()
        self.score = 0
        self.max_distance = Vector2(0, 0).distance_to(Vector2(WIDTH, HEIGHT))
        self.last_distance_to_apple = self.max_distance
        pygame.time.set_timer(self.SNAKE_MOVE_EVENT, 100)

    def spawn_apple(self) -> pygame.Rect:
        ax = randint(0, (WIDTH // TILE_SIZE) - 1) * TILE_SIZE
        ay = randint(0, (HEIGHT // TILE_SIZE) - 1) * TILE_SIZE
        apple_rect = pygame.Rect(ax, ay, TILE_SIZE, TILE_SIZE)

        if any(apple_rect.colliderect(border) for border in borders) or apple_rect.colliderect(self.snake.get_head()):
            self.spawn_apple()

        self.apple = apple_rect

    def restart(self):
        self.snake = Snake(WIDTH // 2, HEIGHT // 2)
        self.spawn_apple()
        self.score = 0

    def distance_to_apple(self):
        head = self.snake.get_head()
        head_vec = Vector2(head.x, head.y)
        return Vector2(self.apple.x, self.apple.y).distance_to(head_vec)

    def get_game_state(self):
        # facing each direction is there danger in the next move?
        state = []
        for direction in Directions:
            # forward, left, right
            facing = self.snake.get_new_facing(direction.value)
            next_head = self.snake.get_next_head(facing)
            state.append(1 if self.snake.collision(next_head) else 0)

        head = self.snake.body[0]
        apple = self.apple
        # apple is above head
        state.append(apple.bottom <= head.top)
        # apple is on the left
        state.append(apple.right <= head.left)
        # apple is on the right
        state.append(apple.left >= head.right)
        # apple is below
        state.append(apple.top >= head.bottom)

        state.append(self.snake.facing == (0, -1))  # up
        state.append(self.snake.facing == (0, 1))  # down
        state.append(self.snake.facing == (-1, 0))  # left
        state.append(self.snake.facing == (1, 0))  # right

        return state

    def play_step(self, direction):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.snake.facing = self.snake.get_new_facing(direction)

        next_head = self.snake.get_next_head(self.snake.facing)

        reward = 0
        game_over = 0

        if self.inactive_moves > 250:
            game_over = 1
            reward = -5
            self.inactive_moves = 0
            print("starved")
            self.restart()

        distance_to_apple = self.distance_to_apple()
        if distance_to_apple < self.last_distance_to_apple:
            reward += 0.05
        else:
            reward -= 0.02

        self.last_distance_to_apple = distance_to_apple

        if next_head.colliderect(self.apple):
            self.snake.update_body_position_and_grow()
            self.spawn_apple()
            self.score += 1
            reward = 100
            print("ate!")
            self.inactive_moves = 0
        else:
            self.snake.update_body_position()
            self.inactive_moves += 1

        if self.snake.collision(self.snake.body[0]):
            print("bumped its head")
            self.inactive_moves = 0
            game_over = 1
            reward = -10
            self.restart()

        self.update_ui()

        return reward, game_over, self.score

    def update_ui(self):
        self.screen.fill("black")

        self.snake.draw(self.screen)
        pygame.draw.rect(self.screen, COLORS["apple"], self.apple)
        for border in borders:
            pygame.draw.rect(self.screen, COLORS["border"], border)
        pygame.display.flip()