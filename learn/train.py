import torch.nn.functional as F


def train(args, model, train_loader, optimizer, epoch, model_type, reg_l2=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if model_type == 'quad' and reg_l2:
            reg = 0.005
            regularization = reg * (
                model.proj1.bias.norm() ** 2 +
                model.proj1.weight.norm() ** 2 +
                model.diag1.bias.norm() ** 2 +
                model.diag1.weight.norm() ** 2
            )
            loss = loss + regularization

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))