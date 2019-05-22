#!/usr/bin/env python
# coding: utf-8

# # Functional Encryption - Classification and information leakage
#  
# We would like to have a measure of the leakage with cryptology notions. In particular, we will look at the notion of [advantage](https://en.wikipedia.org/wiki/Advantage_(cryptography)) in the distinction task.
# 
# > An adversary's advantage is a measure of how successfully it can attack a cryptographic algorithm, by distinguishing it from an idealized version of that type of algorithm -- Wikipedia
# 
# 
# Let $Q$ be the real quadratic network, and let $Q_{ideal}$ be the real one with no collateral leakage (ie accuracy on the collateral task would be 50%). The adversary $F$ is a probabilistic algorithm given $Q$ or $Q_{ideal}$ as input and which outputs 1 or 0 (depending on the font). $F$'s job is to distinguish $Q$ from $Q_{ideal}$ based on making queries given data samples. We say: $Adv(F)=|\Pr[F(Q)=1]-\Pr[F(Q_{ideal})=1]|$
# 
# Because the adversary is sharp, it will fix a digit and try to distinguish on this one the font used. We chose $6$ because we are fair and wanted a digit of average difficulty.
# 
# So to be clear: when building the resistance, we must provide equal resistance against all kind of digits from the couple of fonts selected, but the adversary can be specilized in a single digit. Therefore the problem is a bit unbalanced and we expect worse results compared to the Part 6. Therefore, we already start with longer sabotage and attack phases to get a more realistic view of what we should expect.

# # 8. Collateral Learning assessed with Advantage
# 

# We will use the code directly from the repo, to make the notebook more readable. Functions are similar to those presented earlier.

# In[1]:


# Allow to load packages from parent
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))


# In[2]:


import random
import pickle
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils

import learn
from learn import show_results
# We now import from the collateral module
from learn import collateral


# Hyperparameters:

# PRIVATE_OUTPUT_SIZE = [2, 3, 4, 5, 6, 7]
# ALPHA = [0.5, 1, 1.7, 3]
# LEARNING_RATE = [0.0005, 0.001, 0.002, 0.005, 0.01]
# MOMENTUM = [0.1, 0.2, 0.5, 0.9]

PRIVATE_OUTPUT_SIZE = list(range(1, 11))
LEARNING_RATE = [0.002]
MOMENTUM = [0.5]
ALPHA = [1.7]
CROSS_VALIDATION = list(range(7))

HYPER_PARAMETERS = [MOMENTUM, LEARNING_RATE, ALPHA, PRIVATE_OUTPUT_SIZE, CROSS_VALIDATION]

# In[5]:


N_CHARS = 10
N_FONTS = 2


# ## 8.1 Loading $Q$ with resistance

# In[6]:


class CollateralNet(nn.Module):
    def __init__(self, private_output_size):
        super(CollateralNet, self).__init__()
        self.proj1 = nn.Linear(784, 40)
        self.diag1 = nn.Linear(40, private_output_size, bias=False)

        # --- FFN for characters
        self.lin1 = nn.Linear(private_output_size, 32)
        self.lin2 = nn.Linear(32, N_CHARS)

        # --- Junction
        self.jct = nn.Linear(private_output_size, 784)

        # --- CNN for families
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, N_FONTS)

        # FFN for families
        self.lin3 = nn.Linear(private_output_size, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 16)
        self.lin6 = nn.Linear(16, 8)
        self.lin7 = nn.Linear(8, N_CHARS)

    def quad(self, x):
        # --- Quadratic
        x = x.view(-1, 784)
        x = self.proj1(x)
        x = x * x
        x = self.diag1(x)
        return x

    def char_net(self, x):
        # --- FFN
        x = F.relu(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def font_net(self, x):
        # --- Junction
        x = self.jct(x)
        x = x.view(-1, 1, 28, 28)

        # --- CNN
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_char(self, x):
        x = self.quad(x)
        x = self.char_net(x)
        return F.log_softmax(x, dim=1)

    def forward_font(self, x):
        x = self.quad(x)
        x = self.font_net(x)
        return F.log_softmax(x, dim=1)

    def forward_adv_font(self, x):
        x = self.quad(x)
        # --- FFN
        x = F.relu(x)
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))
        x = self.lin7(x)
        return F.log_softmax(x, dim=1)
    
    # We add the ability to freeze some layers to ensure that the collateral task does
    # not modify the quadratic net
    
    def get_params(self, net):
        """Select the params for a given part of the net"""
        if net == 'quad':
            layers = [self.proj1, self.diag1]
        elif net == 'char':
            layers = [self.lin1, self.lin2]
        elif net == 'font':
            layers = [self.jct, self.fc1, self.fc2, self.conv1, self.conv2]
        else:
            raise AttributeError(f'{net} type not recognized')
        params = [p for layer in layers for p in layer.parameters()]
        return params

    def freeze(self, net):
        """Freeze a part of the net"""
        net_params = self.get_params(net)
        for param in net_params:
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze the net"""
        for param in self.parameters():
            param.requires_grad = True


# In[7]:


def build_resistance(model, args, cv, alpha=0):
    """
    Perform a dual learning phase with sabotage
    """
    
    train_loader, test_loader = collateral.get_data_loaders(args, cv=cv)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    test_perfs_char = []
    test_perfs_font = []
    
    for epoch in range(1, args.epochs + args.sabotage_epochs + args.new_adversary_epochs + 1):
        initial_phase = epoch <= args.epochs
        if initial_phase:
            print("(initial phase)")
        perturbate = epoch > args.epochs and epoch <= args.epochs + args.sabotage_epochs
        if perturbate:
            print("(perturbate)")
        recover = epoch > args.epochs + args.sabotage_epochs
        if recover:
            print("(recover)")
        collateral.train(
            args, model, train_loader, optimizer, epoch, alpha, 
            initial_phase, perturbate, recover, True
        )
        test_perf_char, test_perf_font = collateral.test(args, model, test_loader, recover and True)
        test_perfs_char.append(test_perf_char)
        test_perfs_font.append(test_perf_font)

    return test_perfs_char, test_perfs_font


# We could use the same $Q$ than in part 6, but we could like a better resistance so we rebuild a new one.

def main(model_signature):
    momentum, lr, alpha, private_output_size, cv = model_signature

    class Parser:
        """Parameters for the training"""

        def __init__(self):
            self.epochs = 10
            self.sabotage_epochs = 50
            self.new_adversary_epochs = 15
            self.lr = lr
            self.momentum = momentum
            self.test_batch_size = 1000
            self.batch_size = 64
            self.log_interval = 300

    args = Parser()
    signature = '_'.join(map(str, model_signature))
    print('Computing model...', model_signature)

    path = f"models/quadconvnet_{signature}_par2.pt"
    model = CollateralNet(int(private_output_size))
    results = {}

    test_perfs_char_perturbate, test_perfs_font_perturbate = build_resistance(model, args, cv, alpha=alpha)
    results[f"Main task {signature}"] = test_perfs_char_perturbate
    results[f"Collateral task {signature}"] = test_perfs_font_perturbate
    print(test_perfs_char_perturbate)
    print(test_perfs_font_perturbate)
    torch.save(model.state_dict(), path)
    return results


all_config = list(itertools.product(*HYPER_PARAMETERS))

print('Computing', len(all_config), 'configurations')

results = []
for i, config in enumerate(all_config):
    print(i, '/', len(all_config))
    config_results = main(config)
    results.append(config_results)

with open('results/results_par2.pickle', 'wb') as f:
    pickle.dump(results, f)
