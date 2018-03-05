import torch


def train(model, args):
    pass


def reference(model, args):
    pass


def load_param(model, path):
    model.load_state_dict(torch.load(path))


def save_param(model, path):
    torch.save(model.state_dict(), path)

