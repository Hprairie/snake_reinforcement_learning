import random
from collections import deque


class UniformBuffer:
    def __init__(self, buffer_parameters) -> None:
        self.buffer = deque(maxlen=buffer_parameters['buffer_size'])
        self.batch_size = buffer_parameters['batch_size']

    def add_sample(self, sample):
        self.buffer.append(sample)

    def get_batch(self):
        if len(self.buffer) > self.batch_size:
            mini_sample = random.sample(self.buffer, self.batch_size)
        else:
            mini_sample = self.buffer

        return zip(*mini_sample)


class PrioritizedReplayBuffer:
    def __init__(self) -> None:
        pass

    def add_sample(self):
        pass

    def get_batch(self):
        pass


def determine_buffer(buffer_parameters):
    if buffer_parameters['name'] == 'Uniform':
        strategy = UniformBuffer(buffer_parameters)
    else:
        raise Exception('Unknown Buffer passed. Check naming convention.')

    return strategy
