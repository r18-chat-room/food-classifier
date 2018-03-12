#!/usr/bin/python3
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
    logger.info(args)
    logger.info('*'*10)
    logger.info(categories)

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
        losses, accuracies = op.train(net, train_loader, valid_loader,
                                      epochs=args.epoch, lr=args.lr,
                                      momentum=args.momentum, optim=args.optim,
                                      cuda=args.gpu)
        if args.saving_model_path is not None:
            op.save_param(net, path=args.saving_model_path)
            logger.info('save success at {0}'.format(args.saving_model_path))
        if args.loss_curve_path is not None:
            plt.plot(losses)
            plt.savefig(args.loss_curve_path)
            plt.clf()
            logger.info('save loss curve at {0}'.format(args.loss_curve_path))
        if args.acc_curve_path is not None:
            plt.plot(accuracies[0], label='train_acc')  # train_acc
            plt.plot(accuracies[1], label='valid_acc')  # valid_acc
            plt.legend()
            plt.savefig(args.acc_curve_path)
            plt.clf()
            logger.info('save accuracy curve at {0}'.format(args.acc_curve_path))
        if args.distribution_path is not None:
            distribution_mat = op.validate(
                net, valid_loader, cuda=args.gpu, final_test=True, category_size=args.category_size
            )
            print(distribution_mat)
            plt.xlim(0, args.category_size)
            plt.ylim(0, args.category_size)
            plt.pcolor(distribution_mat.numpy(), cmap=mpl.cm.Blues)
            plt.savefig(args.distribution_path)
            plt.clf()
            logger.info('save distribution head map at {0}'.format(args.distribution_path))

    elif args.op == 'inference':
        pass
