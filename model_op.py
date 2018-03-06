import logging
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def train(model, train_loader, valid_loader, epochs, optim, lr, momentum, cuda):
    """
    the phase of training.

    :param model: the model object to train
    :param train_loader: the data loader of train_dataset
    :param valid_loader: the data loader of valid_dataset
    :param epochs: how many epochs to train
    :param optim: which optimization algorithm used in training
    :param lr: learning rate
    :param momentum: momentum in SGD optimization
    :param cuda: whether the model runs on GPU via CUDA

    :return: list of history losses
    """
    logger = logging.getLogger()
    logger.info('start training {0}: epochs={1}'.format(model.name, epochs))

    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if cuda:
        model = model.cuda()

    epoch_size = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(epochs):
        loss_sum = 0
        hit = 0
        total = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = Variable(x), Variable(y)
            if cuda:
                x, y = x.cuda(), y.cuda()
            y_ = model(x)
            pred = torch.max(F.softmax(y_), 1)[1]

            hit += torch.sum(pred == y).data[0]
            total += y.data.size()[0]

            optimizer.zero_grad()
            loss = criterion(y_, y)
            loss_sum += loss  # calculate average loss in an epoch
            loss.backward()
            optimizer.step()

        loss_avg = loss_sum.data[0] / epoch_size
        losses.append(loss_avg)

        train_acc = hit / total
        valid_acc = validate(model, valid_loader, cuda)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        logger.info('epoch #{0}: train_loss={1:.5} | train_acc={2:.5%} | valid_acc={3:.5%}'
                    .format(epoch + 1, loss_avg, train_acc, valid_acc))

    return losses, train_accs, valid_accs


def validate(model, loader, cuda):
    """
    the phase of validation.

    :param model: the model object to be validated
    :param loader: the data loader of valid_dataset
    :param cuda: whether the model runs on on GPU via CUDA

    :return: accuracy
    """
    if cuda:
        model = model.cuda()

    hit = 0
    total = 0
    for step, (x, y) in enumerate(loader):
        x, y = Variable(x), Variable(y)
        if cuda:
            x, y = x.cuda(), y.cuda()
        y_ = torch.max(F.softmax(model(x)), 1)[1]
        hit += torch.sum(y == y_).data[0]
        total += y.data.size()[0]

    return hit / total


def load_param(model, path):
    """
    load parameters from a pytorch parameter file to a existed model.

    :param model: the existed model object
    :param path: the path of the parameters file

    :return: None
    """
    model.load_state_dict(torch.load(path))


def save_param(model, path):
    """
    save the parameters of a existed model to a pytorch parameter file.

    :param model: the existed model object
    :param path: the path of the parameters file

    :return: None
    """
    torch.save(model.state_dict(), path)
