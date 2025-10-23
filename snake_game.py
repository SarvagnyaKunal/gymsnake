import pygame
import random
import time
import sys
class SnakeEnv:
    def __init__(self, width=750, height=750, delta=30, fps=240):
        pygame.init()
        self.width = width
        self.height = height
        self.delta = delta
        self.fps = fps

        self.playSurface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game")
        self.fpsController = pygame.time.Clock()

        # Colors
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.brown = pygame.Color(165, 42, 42)

        self.reset()

    def reset(self):
        # Ensure grid alignment and valid spawn
        grid_w = self.width // self.delta
        grid_h = self.height // self.delta
        # Snake head at least 2 cells from left edge
        snake_x = random.randrange(2, grid_w - 1)
        snake_y = random.randrange(1, grid_h - 1)
        self.snakePos = [snake_x * self.delta, snake_y * self.delta]
        self.snakeBody = [list(self.snakePos),
                         [self.snakePos[0] - self.delta, self.snakePos[1]],
                         [self.snakePos[0] - 2 * self.delta, self.snakePos[1]]]
        # Food not on snake   
        while True:
            food_x = random.randrange(1, grid_w - 1)
            food_y = random.randrange(1, grid_h - 1)
            self.foodPos = [food_x * self.delta, food_y * self.delta]
            if self.foodPos not in self.snakeBody:
                break
        self.foodSpawn = True
        self.direction = 'RIGHT'
        self.changeto = ''
        self.score = 0
        self.done = False
        self.prev_distance = abs(self.snakePos[0] - self.foodPos[0]) + abs(self.snakePos[1] - self.foodPos[1])
        return self.get_state()


    def step(self, action):
        # Map action int to direction
        action_map = {0:'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}
        self.changeto = action_map[action]

        # Validate direction
        if self.changeto == 'RIGHT' and self.direction != 'LEFT':
            self.direction = self.changeto
        if self.changeto == 'LEFT' and self.direction != 'RIGHT':
            self.direction = self.changeto
        if self.changeto == 'UP' and self.direction != 'DOWN':
            self.direction = self.changeto
        if self.changeto == 'DOWN' and self.direction != 'UP':
            self.direction = self.changeto

        old_dist = abs(self.snakePos[0] - self.foodPos[0]) + abs(self.snakePos[1] - self.foodPos[1])

        # Update snake position
        if self.direction == 'RIGHT':
            self.snakePos[0] += self.delta
        if self.direction == 'LEFT':
            self.snakePos[0] -= self.delta
        if self.direction == 'DOWN':
            self.snakePos[1] += self.delta
        if self.direction == 'UP':
            self.snakePos[1] -= self.delta

        # Snake body mechanism
        self.snakeBody.insert(0, list(self.snakePos))
        if self.snakePos == self.foodPos:
            self.foodSpawn = False
            self.score += 1
        else:
            self.snakeBody.pop()
        if not self.foodSpawn:
            while True:
                self.foodPos = [random.randrange(1, self.width // self.delta) * self.delta,
                                random.randrange(1, self.height // self.delta) * self.delta]
                if self.foodPos not in self.snakeBody:
                    break
            self.foodSpawn = True

        # Check death
        if (self.snakePos[0] >= self.width or self.snakePos[0] < 0 or
            self.snakePos[1] >= self.height or self.snakePos[1] < 0):
            self.done = True

        for block in self.snakeBody[1:]:
            if self.snakePos == block:
                self.done = True

        # Reward logic
        dist = abs(self.snakePos[0] - self.foodPos[0]) + abs(self.snakePos[1] - self.foodPos[1])
        reward = 0  # Default reward
        if self.snakePos == self.foodPos:
            reward = +25.0   # strong reward for eating
            self.score += 1
            self.prev_distance = dist
        elif self.done:
            reward = -15.0   # strong penalty for dying

        else:
            if dist < self.prev_distance:
                reward = +0.5   # moved closer to apple
            elif dist > self.prev_distance:
                reward = -0.5   # moved away
            

            self.prev_distance = dist

        return self.get_state(), reward, self.done

    def get_state(self):
        # Danger detection: check if moving in each direction would cause death
        danger_up = 0
        danger_down = 0
        danger_left = 0
        danger_right = 0
        
        # Check wall collisions
        if self.snakePos[1] - self.delta < 0:  # Top wall
            danger_up = 1
        if self.snakePos[1] + self.delta >= self.height:  # Bottom wall
            danger_down = 1
        if self.snakePos[0] - self.delta < 0:  # Left wall
            danger_left = 1
        if self.snakePos[0] + self.delta >= self.width:  # Right wall
            danger_right = 1
        
        # Check body collisions
        next_up = [self.snakePos[0], self.snakePos[1] - self.delta]
        next_down = [self.snakePos[0], self.snakePos[1] + self.delta]
        next_left = [self.snakePos[0] - self.delta, self.snakePos[1]]
        next_right = [self.snakePos[0] + self.delta, self.snakePos[1]]
        
        if next_up in self.snakeBody:
            danger_up = 1
        if next_down in self.snakeBody:
            danger_down = 1
        if next_left in self.snakeBody:
            danger_left = 1
        if next_right in self.snakeBody:
            danger_right = 1
        
        # State vector: head_x, head_y, food_x, food_y, direction_onehot, danger_detection
        dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        dir_onehot = [0, 0, 0, 0]
        dir_onehot[dirs.index(self.direction)] = 1
        return [self.snakePos[0], self.snakePos[1],
                self.foodPos[0], self.foodPos[1]] + dir_onehot + [danger_up, danger_down, danger_left, danger_right]

    def render(self):
        self.playSurface.fill(self.white)
        for pos in self.snakeBody:
            pygame.draw.rect(self.playSurface, self.green, pygame.Rect(pos[0], pos[1], self.delta, self.delta))
        pygame.draw.rect(self.playSurface, self.brown, pygame.Rect(self.foodPos[0], self.foodPos[1], self.delta, self.delta))
        pygame.display.flip()
        self.fpsController.tick(self.fps)
