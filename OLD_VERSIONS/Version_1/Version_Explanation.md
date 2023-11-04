# Version 1 of Using Deep Q Networks for Reinforcement Learning

The beginning model use a limited context of the imformation directly around the head, along with information about where the apple is in relation to the head.

The model uses a two neural network (current and target models), $\epsilon$ decay for randomized exploration, and a replay buffer. This implementation mimics Mnih et. all 2015 where they introduce DQN, and create a CNN for atari 2600 games.

The major difference between this model and their model is that a smaller CNN is used as a smaller context window is given to the game. Due to the inherent grid nature of snake, it is much simpler to encode the state of the board as 2-dimensional tensor.

However the biggest problem with using a 2-dimensional tensor is that it is hard to indicate to the network which direction the snake is going. In other games Mnih et. all were able to overcome with by having a state include the previous 3 frames in order to indicate direction. I have followed a smilar approach.