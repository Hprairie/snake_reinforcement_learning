# Common Reinforcement Learning algorithms implemented on Snake

Common reinforcement learning algorithms are applied to the game Snake. Some of my favorite runs are displayed below.


## Code Structure

There are 4 main partitions of the projects which allow for using JSON files to essentially pick and choose different algorithms, models, exploration strategies, and board states within the code. I will outline the structure for each of these parts along with their encapsulation. I will also explain how to add additional algorithms, models, exploration strategies, and board states if interested.

<center>

| Agent            | Description | Policy Type | Paper |
|------------------|-------------|-------------|-------|
| DQN              |             |             |       |
| Double DQN       |             |             |       |
| Prioritized DDQN |             |             |       |
| Dueling PDDQN    |             |             |       |
| MultiStep DPDDQN |             |             |       |
| NoiseyNet        |             |             |       |
| Distributional QL|             |             |       |
| RainbowDQN       |             |             |       |
| Actor2Critic     |             |             |       |
| Actor3Critic     |             |             |       |

</center>

### Models

Whenever a model is needed within the project the Model() class in model.py is called, which parses the JSON parameters of the model version passed in at runtime. When constructing the model appends blocks to a sequential model in PyTorch, which enables everything about an experiment to be encapsulated in the JSON file. For example, when creating a CNN model for a DQN, we will assign the topology of the network to the keyword `model`.

```JSON
    "model":{
        "Conv2D":{
            "in_channels" : 4,
            "out_channels" : 32,
            "kernel_size" : [2,2],
            "stride" : 1
        },
        "ReLU" : {},
        "Conv2D_1":{
            "in_channels" : 32,
            "out_channels" : 32,
            "kernel_size" : [2,2],
            "stride" : 1
        },
        "ReLU_2" : {},
        "Flatten":{},
        "Dense":{
            "in_features" : 1152,
            "out_features" : 128
        },
        "ReLU_3" : {},
        "Dense2":{
            "in_features" : 128,
            "out_features" : 3
        }
    }
```

Order is important in the JSON file and runtime errors will be thrown if the model parameters within the JSON file are not compatible with the `game_state` parameters within the JSON file.


## Loading Pre-Trained Models


## How to run experiments

### JSON Parameters

## Personal Experiments


## How 


## Acknowledgments

I would like to acknowledge [this repository](https://github.com/patrickloeber/snake-ai-pytorch) which I originally used to get started with reinforcement learning. The basic structure of the snake game was lifted from the repository, however many things were changed in order to improve the display of the game and also to expand the context window to the entire board frame, plus additional board frames. Furthermore, the structure of this codebase was heavily inspired by [this repository](https://github.com/DragonWarrior15/snake-rl), where JSON files were parsed in order to control the model parameters. I do want to note that 98% of this code was still written by me and inspiration was just lifted from these sources.

### Important Papers used for this Project
