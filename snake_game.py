import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np

# Initializing pygame
pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (200, 200, 200)
GREY = (128, 128, 128)
RED = (200, 0, 0)
GREEN1 = (0, 158, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80


class SnakeGame:
    '''
    Class for the snake game. Enviroment comes will be fully initialized
    on constuction.

    Note: The game is still playable after collision, so it is important
    to call the reset function in order to fully restart the game after
    each iteration.

    Attributes
    ----------
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
        start_length : int, optional <- Not Setup
            The starting length of the snake
        display_game : Bool, optional <- Not Setup
            Choose wether or not to display the game, useful for
            debugging during training
        seed : int, optional <- Not Setup
            Used for the randomness of spawning apples
        max_time_rate : int, optional
            Coefficient used to determine the max frames allowed
            within a certain game state. The maximum allowed frames
            for a given game is max_iter_rate * len(snake)
        '''
        # Passed Parameters
        self.board_size = board_size
        self.frames = frames
        self.start_length = start_length
        self.display_game = display_game
        self.seed = seed
        self.max_time_rate = max_time_rate

        # Generated Parameters
        self.state = deque(maxlen=frames)

        # Initialize the display
        if self.display_game:
            WINDOW = ((board_size[1]+2)*BLOCK_SIZE,
                      (board_size[0]+2)*BLOCK_SIZE)
            self.display = pygame.display.set_mode(WINDOW)
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
        self.head = Point(self.board_size[1] * BLOCK_SIZE / 2,
                          self.board_size[0] * BLOCK_SIZE / 2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.state.clear()

        # Init the game state with starting frames
        while len(self.state) != self.frames:
            self.state.append(self._get_game_frame())

    def _get_game_frame(self):
        '''
        Will return the current frame of the game. A 2 dimensional tensor
        is return as a numpy array has the dimensions in the form of
        (height, width)

        Returns
        -------
        frame: Numpy array
            Current frame of the game
        '''
        frame = np.zeros(((self.board_size[0]) + 2,
                          (self.board_size[1]) + 2), dtype=int)

        # Insert boarders
        frame[0, :] = -10
        frame[-1, :] = -10
        frame[:, 0] = -10
        frame[:, -1] = -10

        # Insert the snake into the context
        for body_point in self.snake:
            frame[int((body_point.y - BLOCK_SIZE) // BLOCK_SIZE) + 2,
                  int((body_point.x - BLOCK_SIZE) // BLOCK_SIZE) + 2] = -1

        # Inset the food into the context
        frame[int((self.food.y - BLOCK_SIZE) // BLOCK_SIZE) + 2,
              int((self.food.x - BLOCK_SIZE) // BLOCK_SIZE) + 2] = 1

        return frame

    def get_game_state(self):
        '''
        Will return the current state of the game which is stored in the
        state buffer self.state. The return will be a numpy array in the
        shape (frames, frame_height, frame_width)

        Returns
        -------
        context : Numpy array
            Current State of the game
        '''
        context = np.zeros((self.frames,
                           (self.board_size[0]) + 2,
                           (self.board_size[1]) + 2), dtype=int)

        for i, frame in enumerate(self.state):
            context[i] = frame

        return context

    def _place_food(self):
        '''
        Function will place food in the game based on the seed provided. If
        no seed is provided then the game with randomly place a peice of food.
        '''
        x = random.randint(0, (self.board_size[1] - 1))*BLOCK_SIZE
        y = random.randint(0, (self.board_size[0] - 1))*BLOCK_SIZE
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
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        max_frame_threshold = self.max_time_rate * len(self.snake)
        if self.is_collision() or self.frame_iteration > max_frame_threshold:
            self.snake.pop()
            self.state.append(self._get_game_frame())
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
        self.state.append(self._get_game_frame())
        if self.display_game:
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
        if pt.x > (self.board_size[1] - 1) * BLOCK_SIZE or pt.x < 0 or \
                pt.y > (self.board_size[0] - 1) * BLOCK_SIZE or pt.y < 0:
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

        # Place grid
        for idx_1 in range(self.board_size[1] + 2):
            for idx_2 in range(self.board_size[0] + 2):
                pygame.draw.rect(self.display,
                                 WHITE,
                                 pygame.Rect(idx_1 * BLOCK_SIZE,
                                             idx_2 * BLOCK_SIZE,
                                             BLOCK_SIZE,
                                             BLOCK_SIZE))

                pygame.draw.rect(self.display,
                                 BLACK,
                                 pygame.Rect(idx_1 * BLOCK_SIZE,
                                             idx_2 * BLOCK_SIZE,
                                             BLOCK_SIZE - 1,
                                             BLOCK_SIZE - 1))

        # Place Boarders
        for idx in range(self.board_size[0] + 2):
            pygame.draw.rect(self.display,
                             GREY,
                             pygame.Rect(0,
                                         idx * BLOCK_SIZE,
                                         BLOCK_SIZE - 1,
                                         BLOCK_SIZE - 1))

            pygame.draw.rect(self.display,
                             GREY,
                             pygame.Rect((self.board_size[1] + 1) * BLOCK_SIZE,
                                         idx * BLOCK_SIZE,
                                         BLOCK_SIZE - 1,
                                         BLOCK_SIZE - 1))

        for idx in range(self.board_size[1] + 2):
            pygame.draw.rect(self.display,
                             GREY,
                             pygame.Rect(idx * BLOCK_SIZE,
                                         0,
                                         BLOCK_SIZE - 1,
                                         BLOCK_SIZE - 1))

            pygame.draw.rect(self.display,
                             GREY,
                             pygame.Rect(idx * BLOCK_SIZE,
                                         (self.board_size[0] + 1) * BLOCK_SIZE,
                                         BLOCK_SIZE - 1,
                                         BLOCK_SIZE - 1))
        # Place Snake
        for pt in self.snake:
            pygame.draw.rect(self.display,
                             GREEN1,
                             pygame.Rect(pt.x + BLOCK_SIZE,
                                         pt.y + BLOCK_SIZE,
                                         BLOCK_SIZE - 1,
                                         BLOCK_SIZE - 1))

        # Place Head
        pygame.draw.rect(self.display,
                         GREEN2,
                         pygame.Rect(self.head.x + BLOCK_SIZE,
                                     self.head.y + BLOCK_SIZE,
                                     BLOCK_SIZE - 1,
                                     BLOCK_SIZE - 1))

        # Place Food
        pygame.draw.rect(self.display,
                         RED,
                         pygame.Rect(self.food.x + BLOCK_SIZE,
                                     self.food.y + BLOCK_SIZE,
                                     BLOCK_SIZE - 1,
                                     BLOCK_SIZE - 1))

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


if __name__ == "__main__":
    # Use this only for testing
    game = SnakeGame(board_size=(6, 4))

    while (True):
        game.play_step([0, 1, 0])
