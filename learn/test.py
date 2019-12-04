import torch
import torch.nn.functional as F


def test(args, model, test_loader, return_pred_label=False):
    model.eval()
    test_loss = 0
    correct = 0
    pred_labels = None
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                1, keepdim=True
            )  # get the index of the max log-probability

            # Store all the pred, label pairs to draw the confusion matrix
            pred_labels_batch = torch.stack((pred, target.view_as(pred))).view(
                2, args.test_batch_size
            )
            if pred_labels is None:
                pred_labels = pred_labels_batch
            else:
                pred_labels = torch.cat((pred_labels, pred_labels_batch), dim=1).view(
                    2, -1
                )

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc
        )
    )

    if return_pred_label:
        return acc, pred_labels.transpose(0, 1)
    else:
        return acc
