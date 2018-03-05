import logging
import torch
from torch.autograd import Variable


def train(model, loader, epochs, optim, lr, momentum, cuda):
    logger = logging.getLogger()
    logger.info('start training {0}: epochs={1}'.format(model.name, epochs))

    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if cuda:
        model = model.cuda()

    epoch_size = len(loader)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        logger.info('epoch #{0}'.format(epoch))
        loss_sum = 0
        for step, (x, y) in enumerate(loader):
            x, y = Variable(x), Variable(y)
            if cuda:
                x, y = x.cuda(), y.cuda()
            y_ = model(x)
            optimizer.zero_grad()
            loss = criterion(y_, y)
            loss_sum += loss    # calculate average loss in an epoch
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                logger.info('\tstep #{0}: loss={1}'.format(step+1, loss.data[0]))
        loss_avg = loss_sum.data[0] / epoch_size
        logger.info('epoch #{0}: avg_loss={1}'.format(epoch+1, loss_avg))
        losses.append(loss_avg)
    return losses


def reference(model, args):
    pass


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

