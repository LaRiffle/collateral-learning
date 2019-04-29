import torch
import torch.optim as optim
import torch.utils.data as utils

from learn import build_tensor_dataset, load_data
from learn.collateral_OLD import collateral_train, collateral_test


def collateral_task(args, model, collateral_model, prec_frac):

    # Load data
    data = load_data()
    train_data, train_target_char, train_target_family, test_data, test_target_char, test_target_family = data

    # setting = the family recognition task
    train_dataset = build_tensor_dataset(train_data, train_target_family)
    test_dataset = build_tensor_dataset(test_data, test_target_family)

    train_loader = utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = utils.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True
    )

    adversarial_optimizer = optim.SGD(collateral_model.parameters(), lr=args.lr, momentum=args.momentum)

    test_perfs = []
    for epoch in range(1, args.epochs + 1):
        collateral_train(args, collateral_model, model, train_loader, adversarial_optimizer, epoch, prec_frac)
        acc = collateral_test(collateral_model, model, test_loader, prec_frac)
        test_perfs.append(acc)

    return test_perfs

