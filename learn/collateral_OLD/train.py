import torch.nn.functional as F


def collateral_train(args, collateral_model, model, train_loader, adv_optimizer, epoch, prec_frac):
    collateral_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data.fix_precision_(precision_fractional=prec_frac)  # Convert to Fixed precision
        data = model.transform(data)  # Just do the private part of the forward
        data = data.float_precision()  # Convert back to float
        # Really start the ususal training with the collateral_model
        adv_optimizer.zero_grad()
        output = collateral_model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        adv_optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
