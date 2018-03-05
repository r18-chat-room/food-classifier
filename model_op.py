import torch


def train(model, args):
    pass


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

