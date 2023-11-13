import random
import numpy as np
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
    def __init__(self, buffer_parameters) -> None:
        self.buffer = [0 for _ in range(buffer_parameters['buffer_size'])]
        self.priority_sum = [0 for _ in range(2 * buffer_parameters['buffer_size'])]
        self.priority_min = [float('inf') for _ in range(2 * buffer_parameters['buffer_size'])]
        self.next_idx = 0
        self.max_priority = 1
        self.size = 0
        self.capacity = buffer_parameters['buffer_size']

        self.batch_size = buffer_parameters['batch_size']

        self.alpha = buffer_parameters['alpha']
        self.beta = buffer_parameters['beta']
        self.beta_max = buffer_parameters['beta_max']
        self.beta_growth = buffer_parameters['beta_growth']

    def _set_priority_sum(self, index, priority):
        index += self.capacity
        self.priority_sum[index] = priority

        # Update the Segmentation Tree
        while index >= 2:
            index = int(index // 2)
            self.priority_sum[index] = self.priority_sum[index*2] + self.priority_sum[index*2+1]

    def _set_priority_min(self, index, priority):
        index += self.capacity
        self.priority_min[index] = priority

        # Update the Segmentation Tree
        while index >= 2:
            index = int(index // 2)
            self.priority_min[index] = min(self.priority_min[2*index], self.priority_min[2*index+1])

    def append(self, sample):
        # Add sample to buffer and update size and index trackers
        index = self.next_idx
        self.buffer[index] = sample
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # Update the sample sum and min segment trees
        self._set_priority_sum(index, self.max_priority ** self.alpha)
        self._set_priority_min(index, self.max_priority ** self.alpha)

    def get_batch(self):
        # Create Distribution and Indexed List
        distribution = np.array(self.priority_sum[self.capacity:self.capacity + self.size]) / self.priority_sum[1]
        index_list = list(range(self.size))

        # Sample from the list (index, element)
        if self.size < self.batch_size:
            batch_size = self.size
        else:
            batch_size = self.batch_size

        indexs = np.random.choice(index_list, p=distribution, size=batch_size)
        samples = [self.buffer[index] for index in indexs]

        # Compute the Importance Sampling Weight
        max_weight = (self.size * self.priority_min[1]) ** (-self.beta)
        weights = (self.size * distribution[indexs]) ** (-self.beta) / max_weight

        return indexs, samples, weights

    def update_priorities(self, indexs, priorities):
        for index, priority in zip(indexs, priorities):
            self.max_priority = max(self.max_priority, priority)
            self._set_priority_min(index, priority ** self.alpha)
            self._set_priority_sum(index, priority ** self.alpha)
    
    def update_hyperparameters(self):
        if self.beta < self.beta_max:
            self.beta = min(self.beta_max, self.beta + self.beta_growth)


def determine_buffer(buffer_parameters):
    if buffer_parameters['name'] == 'Uniform':
        strategy = UniformBuffer(buffer_parameters)
    if buffer_parameters['name'] == 'Priority':
        strategy = PrioritizedReplayBuffer(buffer_parameters)
    else:
        raise Exception('Unknown Buffer passed. Check naming convention.')

    return strategy
