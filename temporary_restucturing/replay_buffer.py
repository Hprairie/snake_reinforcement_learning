import random
from collections import deque


class replay_buffer:
    def __init__(self, batch_size, max_memory, seed=None):
        self.buffer = deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.seed = seed

    def remember(self, state, action, reward, next_state, game_over):
        self.replay_buffer.append((state, action, reward,
                                   next_state, game_over))

    def get_batch(self):
        if self.seed:
            random.seed(self.seed)

        if len(self.buffer) > self.batch_size:
            sample = random.sample(self.buffer, self.batch_size)
        else:
            sample = self.buffer

        state, action, reward, next_state, game_over = zip(*sample)

        return state, action, reward, next_state, game_over
