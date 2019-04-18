import pickle

import torch
import torch.optim as optim
import torch.utils.data as utils

from .train import train
from .test import test

from .models import QuadNet, CNN

from learn.utils import Parser, load_data, build_tensor_dataset

def main(model_type='quad', args=None, model=None, optimizer=None, task='char', reg_l2=False, reg=0.01, epochs=None, return_model=False, return_pred_label=False):
    """
    Perform a learning phase

    You can provide your data and model, but by default it will load it by himself.

    You can choose to learn on:
    task=(char|font)
    model_type=(quad|cnn)
    """

    torch.manual_seed(1)
    if args is None:
        args = Parser()
        if epochs is not None:
            args.epochs = epochs
    else:
        assert epochs is None

    data = load_data()
    train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family = data

    target_types = ['font', 'char']
    assert task in target_types
    print('Learning on', task, 'and reg_l2' if reg_l2 else '')
    if task == 'font':
        train_target = train_target_family
        test_target = test_target_family
    elif task == 'char':
        train_target = train_target_char
        test_target = test_target_char

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

    if optimizer is None:
        # optimizer = optim.Adam(model.parameters(), lr=0.0006)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    test_perfs = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch, model_type, reg_l2, reg)
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
