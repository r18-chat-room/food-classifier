import argparse


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__()
        self.add_argument('phase', type=str, choices=['train', 'inference'],
                          help='the phase of model')

        self.add_argument('--model', type=str, default=None,
                          help='the name of model')
        self.add_argument('--model-path', type=str, default=None,
                          help='the path of pre-trained model parameters (.pkl)'),
        self.add_argument('--data-path', type=str, default='./data/',
                          help='the path of dataset used for training')
        self.add_argument('--img-size', type=int, default=224,
                          help='the size of images in dataset')
        self.add_argument('--category-size', type=int, default=24,
                          help='the number of categories')

        self.add_argument('--lr', dest='learning_rate', type=float, default=0.0003,
                          help='learning rate when training')
        self.add_argument('--momentum', type=float, default=0.95,
                          help='momentum in SGD optimization')
        self.add_argument('--batch-size', type=int, default=8,
                          help='how many samples per batch to load')
        self.add_argument('--epoch', type=int, default=80,
                          help='how many epochs to train')
        self.add_argument('--valid', dest='valid_pct', type=float, default=0.2,
                          help='percentage split of dataset used for validation')

        self.add_argument('--gpu', action='store_true', default=False,
                          help='whether the model runs on GPU via CUDA')
        # self.add_argument('')

