import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import gym
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import tqdm

ENVIRONMENT = gym.make('Breakout-v0').unwrapped
ENVIRONMENT.reset()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print("Is python : {}".format(is_ipython))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(DEVICE))


AMOUNT_OF_ACTIONS = ENVIRONMENT.action_space.n
print("Number of actions : {}".format(AMOUNT_OF_ACTIONS))







START_REPLAY_MEMORY = 5000
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
INPUT_SIZE = 84
AMOUNT_OF_EPISODES = 50000
HELPER_UPDATE = 100
DISCOUNT_FACTOR = 0.9
PRINT_EVERY = 100

EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 100


REWARD_DEDUCTOR_IN_CASE_OF_LOOSE_POINTS = 0
REWARD_MULTIPLICATOR_FOR_GETTING_POINTS = 50
REWARD_FOR_STAYING_ALIVE = 0
