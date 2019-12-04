import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as utils


class Parser:
    """Parameters for the training"""

    def __init__(self):
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.test_batch_size = 1000
        self.batch_size = 64
        self.log_interval = 300


def build_tensor_dataset(data, target):
    """Utility function to cast our data into a normalized torch TensorDataset"""
    normed_data = [(d - d.mean()) / d.std() for d in data]
    normed_data = torch.stack([torch.Tensor(d).reshape(1, 28, 28) for d in normed_data])
    target = torch.LongTensor([i[0] for i in target])
    tensor_dataset = utils.TensorDataset(normed_data, target)
    return tensor_dataset


def get_test_loader(args):
    """
    Convenient function to make notebook more readable
    :return: the test loader
    """
    data = load_data()
    _, _, _, test_data, test_target_char, _ = data
    test_dataset = build_tensor_dataset(test_data, test_target_char)
    test_loader = utils.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True
    )
    return test_loader


def preprocess_data(data):
    """
    Inverse color and normalize like MNIST
    """
    mean, std = 0.1307, 0.3081
    data = 255 - np.array(data)
    data = (data - data.mean()) / data.std() * std + mean
    return data


def load_data():
    """
    Load the training and testing sets
    """

    train_data, train_target_char, train_target_family = [], [], []

    # The training set
    for i in range(6):
        with open(f"dataset/character_dataset_train{i}.pkl", "rb") as input_file:
            training_set = pickle.load(input_file)
            train_data_i, train_target_char_i, train_target_family_i = training_set
            train_data += train_data_i
            train_target_char += train_target_char_i
            train_target_family += train_target_family_i

    # The testing set
    with open(f"dataset/character_dataset_test.pkl", "rb") as input_file:
        testing_set = pickle.load(input_file)
        test_data, test_target_char, test_target_family = testing_set

    print("Training set", len(train_data), "items")
    print("Testing set ", len(test_data), "items")

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    data = (
        train_data,
        train_target_char,
        train_target_family,
        test_data,
        test_target_char,
        test_target_family,
    )

    return data
