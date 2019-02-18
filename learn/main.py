import pickle

import torch
import torch.optim as optim
import torch.utils.data as utils

from .train import train
from .test import test

from .models import QuadNet, CNN


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
    test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
    return test_loader


def load_data():

    train_data, train_target_char, train_target_family = [], [], []

    # The training set
    for i in range(6):
        with open(f'dataset/character_dataset_train{i}.pkl', 'rb') as input_file:
            training_set = pickle.load(input_file)
            train_data_i, train_target_char_i, train_target_family_i = training_set
            train_data += train_data_i
            train_target_char += train_target_char_i
            train_target_family += train_target_family_i

    # The testing set
    with open(f'dataset/character_dataset_test.pkl', 'rb') as input_file:
        testing_set = pickle.load(input_file)
        test_data, test_target_char, test_target_family = testing_set

    print('Training set', len(train_data), 'items')
    print('Testing set ', len(test_data), 'items')

    data = train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family

    return data


def main(model_type, args=None, data=None, model=None, optimizer=None, task='char', reg_l2=False, epochs=None, return_model=False, return_pred_label=False):
    """
    Perform a learning phase

    You can provide your data and model, but by default it will load it by himself.

    You can choose to learn on:
    task=(char|family)
    model_type=(quad|cnn)
    """

    torch.manual_seed(1)
    if args is None:
        args = Parser()
        if epochs is not None:
            args.epochs = epochs
    else:
        assert epochs is None

    if data is None:
        data = load_data()
        train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family = data

    target_types = ['family', 'char']
    assert task in target_types
    print('Learning on', task, 'with', model_type, 'and reg_l2' if reg_l2 else '')
    if task == 'family':
        train_target = train_target_family
        test_target = test_target_family
        output_size = 5
    elif task == 'char':
        train_target = train_target_char
        test_target = test_target_char
        output_size = 26

    train_dataset = build_tensor_dataset(train_data, train_target)
    test_dataset = build_tensor_dataset(test_data, test_target)

    train_loader = utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = utils.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True
    )
    assert model_type in ['quad', 'cnn']

    # You can provide your own model
    if model is None:
        if model_type == 'quad':
            model = QuadNet(output_size=output_size)
        elif model_type == 'cnn':
            model = CNN(output_size=output_size)

    if optimizer is None:
        # optimizer = optim.Adam(model.parameters(), lr=0.0006)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test_perfs = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch, model_type)
        if epoch <= args.epochs:
            if return_pred_label:
                test_perf, pred_labels = test(args, model, test_loader, return_pred_label)
            else:
                test_perf = test(args, model, test_loader)
            test_perfs.append(test_perf)

    returns = [test_perfs]
    if return_model:
        returns.append(model)
    if return_pred_label:
        returns.append(pred_labels)

    if len(returns) > 1:
        return tuple(returns)
    else:
        return returns[0]
