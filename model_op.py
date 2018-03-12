import logging
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def train(model, train_loader, valid_loader, epochs, optim, lr, momentum, cuda, lr_decay_interval):
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
    :param lr_decay_interval: lr decayed interval epoch

    :return: list of history losses
    """
    logger = logging.getLogger()
    logger.info('start training {0}: epochs={1}'.format(model.name, epochs))

    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr, weight_decay=0.0005)
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
        adjust_learning_rate(optimizer, lr, epoch, lr_decay_interval)   # lr decayed

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

            # l2_reg = None
            # for w in model.parameters():
            #     if l2_reg is None:
            #         l2_reg = w.norm(2)
            #     else:
            #         l2_reg += w.norm(2)
            #
            # loss = criterion(y_, y) + l2_reg * l2_lambda
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

    return losses, (train_accs, valid_accs)


def adjust_learning_rate(optimizer, init_lr, cur_epoch, interval_epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every x epochs.

    :param init_lr: initial learning rate
    :param optimizer: the optimizer object
    :param cur_epoch: current epoch
    :param interval_epoch: lr decayed interval epoch
    """
    lr = init_lr * (0.1 ** (cur_epoch // interval_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(model, loader, cuda, final_test=False, category_size=24):
    """
    the phase of validation.

    :param category_size:
    :param final_test:
    :param model: the model object to be validated
    :param loader: the data loader of valid_dataset
    :param cuda: whether the model runs on on GPU via CUDA

    :return: accuracy
    """
    if cuda:
        model = model.cuda()

    if final_test:
        distribution_mat = torch.LongTensor(category_size, category_size).fill_(1)
    hit = 0
    total = 0
    for step, (x, y) in enumerate(loader):
        x, y = Variable(x), Variable(y)
        if cuda:
            x, y = x.cuda(), y.cuda()
        y_ = torch.max(F.softmax(model(x)), 1)[1]
        hit += torch.sum(y == y_).data[0]
        total += y.data.size()[0]

        if final_test:
            pred_stat(y_.data, y.data, distribution_mat)

    if final_test:
        distribution_mat = distribution_mat.float()
        for ax in range(category_size):
                distribution_mat[ax, :] /= torch.sum(distribution_mat[ax, :])
        return distribution_mat

    return hit / total


def pred_stat(pred, fact, mat):
    """

    :param pred: prediction value
    :param fact: factual value
    :param mat: distribution matrix

    :return:
    """
    for y_, y in zip(pred, fact):
        mat[y_][y] += 1


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
