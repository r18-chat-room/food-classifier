import logging
import sys
import matplotlib as mpl
import numpy as np
import time
import model_op as op

from dataset import get_categories, get_train_valid_loader
from parser import Parser
from model import model_list

# Logging Settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s'))
logger.addHandler(handler)

parser = Parser()
categories = get_categories('./data')
args, _ = parser.parse_known_args(sys.argv[1:])

mpl.use('Agg')
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.figsize'] = (8, 6)
np.random.seed(int(time.time()))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    net = model_list[args.model](args.category_size)
    train_loader, valid_loader = get_train_valid_loader(
        path=args.data_path, scale_size=args.img_size, batch_size=args.batch_size,
        valid_size=args.valid_pct, num_workers=args.num_workers, shuffle=True,
    )

    if args.model_path is not None:
        op.load_param(net, args.model_path)
        logger.info('load success from {0}'.format(args.model_path))

    if args.op == 'info':
        print(net)
    elif args.op == 'train':
        losses = op.train(net, train_loader,
                          epochs=args.epoch, lr=args.lr,
                          momentum=args.momentum, optim=args.optim,
                          cuda=args.gpu)
        if args.saving_model_path is not None:
            op.save_param(net, path=args.saving_model_path)
            logger.info('save success at {0}'.format(args.saving_model_path))
        if args.loss_curve_path is not None:
            plt.plot(losses)
            plt.savefig(args.loss_curve_path)
            logger.info('save loss curve at {0}'.format(args.loss_curve_path))
    elif args.op == 'inference':
        pass
