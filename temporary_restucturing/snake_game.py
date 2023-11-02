import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import pygame
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80


class SnakeGame:
    '''
    Class for the snake game. Enviroment comes will be fully initialized
    on 
    '''
    
    def __init__(self, board_size=(10, 10), frames=4, start_length=3,
                 display_game=False, seed=None, max_time_rate=100):
        '''
        Initializer of the snake game. The explanation of features
        can be seen below.

        Parameters
        ----------

        board_size : tuple/int, optional
            The board size of the game
        frames : int, optional
            The total number of sequential frames return in each
            state
        start_length : int, optional
            The starting length of the snake
        display_game : Bool, optional
            Choose wether or not to display the game, useful for
            debugging during training
        seed : int, optional
            Used for the randomness of spawning apples
        max_time_rate : int, optional
            Coefficient used to determine the max frames allowed
            within a certain game state. The maximum allowed frames
            for a given game is max_iter_rate * len(snake)
        '''
        self.board_size = board_size
        self.frames = frames
        self.start_length = start_length
        self.seed = seed
        self.max_time_rate = max_time_rate

        # Initialize the display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        '''
        Reset's the state of the game. Reset is automatically called on the
        initialization of the entire snake class. Function reset board and
        snake in deterministic fashion, but will randomly initialize the
        apple based on seed.
        '''
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def get_game_state(self):
        '''
        Will return the current state of the game. A 3 dimensional tensor
        is return as a numpy array has the dimensions in the form of
        (frames, height, width)

        Returns
        -------
        context: Numpy array
            State of the game
        '''
        context = np.zeros(((self.h // BLOCK_SIZE) + 2, (self.w // BLOCK_SIZE) + 2), dtype=int)

        # Create boarders
        context[0, :] = -10
        context[-1, :] = -10
        context[:, 0] = -10
        context[:, -1] = -10

        # Insert the snake into the context
        for body_point in self.snake:
            context[int((body_point.y - BLOCK_SIZE )// BLOCK_SIZE) + 2, int((body_point.x - BLOCK_SIZE) // BLOCK_SIZE) + 2] = -1

        # Inset the food into the context
        context[int((self.food.y - BLOCK_SIZE) // BLOCK_SIZE) + 2, int((self.food.x - BLOCK_SIZE) // BLOCK_SIZE) + 2] = 1

        return context

    def _place_food(self):
        '''
        Function will place food in the game based on the seed provided. If
        no seed is provided then the game with randomly place a peice of food.
        '''
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        '''
        Will

        Returns
        -------
        reward : int
            The reward gained from taking the supplied action
        game_over : Bool
            Boolean indication wether the game has ended or not. Will only
            be raised when the snake collides with itself or the wall.
        score: int
            The current score of the game. Equivalent to the total number
            of apples the snake has consummed.
        '''
        # Update the current frame iteration
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            self.snake.pop()
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        '''
        Function check the head of the snake and determines if it has collided
        with its body or the boarders of the game.

        Returns
        -------

        collision: Bool
            True is snake collided with something
        '''
        # If temporary function call of is_collision is used then pt -> point
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        '''
        Function which visually displays the game using pygame.
        '''
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        '''
        Function which moves the snake based on a certain action. Will
        update the entire context of the game and shift in a new frame
        '''
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
