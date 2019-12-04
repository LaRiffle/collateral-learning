import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def test(args, model, test_loader, new_adversary):
    model.eval()
    correct_char = 0
    correct_font = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Split the two targets
            target_char = target[:, 0]
            target_font = target[:, 1]

            # Char evaluation
            if not new_adversary:
                output = model.forward_char(data)
                pred = output.argmax(1, keepdim=True)
                correct_char += pred.eq(target_char.view_as(pred)).sum().item()

            # Font evaluation
            if not new_adversary:
                output = model.forward_font(data)
            else:
                output = model.forward_adv_font(data)
            pred = output.argmax(1, keepdim=True)
            correct_font += pred.eq(target_font.view_as(pred)).sum().item()

    acc_char = 100.0 * correct_char / len(test_loader.dataset)
    acc_font = 100.0 * correct_font / len(test_loader.dataset)
    print(
        "\nTest set: Accuracy Char : {}/{} ({:.2f}%)\n          Accuracy Font : {}/{} ({:.2f}%)".format(
            correct_char,
            len(test_loader.dataset),
            acc_char,
            correct_font,
            len(test_loader.dataset),
            acc_font,
        )
    )

    return acc_char, acc_font
