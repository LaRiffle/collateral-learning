import torch
import torch.nn.functional as F


def collateral_test(collateral_model, model, test_loader, prec_frac):
    collateral_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data.fix_precision_(precision_fractional=prec_frac) # Convert to Fixed precision
            data = model.transform(data)  # Just do the private part of the forward
            data = data.float_precision()  # Convert back to Float
            output = collateral_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))

    return acc
