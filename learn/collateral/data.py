import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils

import pickle


def get_data_loaders(args, font1=None, font2=None, cv=6):
    torch.manual_seed(1)

    train_dataset, test_dataset = get_datasets(font1, font2, cv)

    train_loader = utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = utils.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True
    )
    return train_loader, test_loader


def get_datasets(font1, font2, cv=6):
    data = load_resistance_data(font1, font2, cv)
    train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family = data
    # Merge the target datasets
    train_target = list(zip(train_target_char, train_target_family))
    test_target = list(zip(test_target_char, test_target_family))

    # We use here the slightly modified version of this function
    train_dataset = build_tensor_dataset(train_data, train_target)
    test_dataset = build_tensor_dataset(test_data, test_target)

    return train_dataset, test_dataset


def load_resistance_data(font1, font2, cv=6):
    print('cv=', cv)

    base_name = f'dataset/character_dataset_{font1}_{font2}' \
        if font1 is not None and font2 is not None \
        else f'dataset/character_dataset'

    train_data, train_target_char, train_target_family = [], [], []

    data_files = [f'{base_name}_train{i}.pkl' for i in range(6)] + [f'{base_name}_test.pkl']

    assert 0 <= cv <= len(data_files)

    for i, data_file in enumerate(data_files):
        with open(data_file, 'rb') as input_file:
            # Build the training set
            if i != cv:
                training_set = pickle.load(input_file)
                train_data_i, train_target_char_i, train_target_family_i = training_set
                train_data += train_data_i
                train_target_char += train_target_char_i
                train_target_family += train_target_family_i
            # The testing set
            else:
                testing_set = pickle.load(input_file)
                test_data, test_target_char, test_target_family = testing_set

    print('Training set', len(train_data), 'items')
    print('Testing set ', len(test_data), 'items')

    data = train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family

    return data


def get_collateral_data_loaders(args, font1, font2, letter, cv=6):
    torch.manual_seed(1)

    train_dataset, test_dataset = get_collateral_datasets(font1, font2, letter, cv)

    train_loader = utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = utils.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True
    )
    return train_loader, test_loader


def get_collateral_datasets(font1, font2, letter, cv=6):
    data = load_collateral_data(font1, font2, letter, cv)
    train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family = data
    # Merge the target datasets
    train_target = list(zip(train_target_char, train_target_family))
    test_target = list(zip(test_target_char, test_target_family))

    # We use here the slightly modified version of this function
    train_dataset = build_tensor_dataset(train_data, train_target)
    test_dataset = build_tensor_dataset(test_data, test_target)

    return train_dataset, test_dataset


def load_collateral_data(font1, font2, letter, cv=6):
    print('cv=', cv)
    base_name = f'dataset/character_dataset'
    train_data, train_target_char, train_target_family = [], [], []

    data_files = [f'{base_name}_{font1}_{font2}_{letter}_train{i}.pkl' for i in range(6)] \
                 + [f'{base_name}_{font1}_{font2}_{letter}_test.pkl']

    assert 0 <= cv <= len(data_files)

    for i, data_file in enumerate(data_files):
        with open(data_file, 'rb') as input_file:
            # Build the training set
            if i != cv:
                training_set = pickle.load(input_file)
                train_data_i, train_target_char_i, train_target_family_i = training_set
                train_data += train_data_i
                train_target_char += train_target_char_i
                train_target_family += train_target_family_i
            # The testing set
            else:
                testing_set = pickle.load(input_file)
                test_data, test_target_char, test_target_family = testing_set

    print('Training set', len(train_data), 'items')
    print('Testing set ', len(test_data), 'items')

    data = train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family

    return data


def build_tensor_dataset(data, target):
    """Utility function to cast our data into a normalized torch TensorDataset"""
    normed_data = [(d - d.mean()) / d.std() for d in data]
    normed_data = torch.stack([torch.Tensor(d).reshape(1, 28, 28) for d in normed_data])
    target = torch.LongTensor([[i[0][0], i[1][0]] for i in target])
    tensor_dataset = utils.TensorDataset(normed_data, target)
    return tensor_dataset