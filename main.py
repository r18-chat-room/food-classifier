import logging
import sys
import matplotlib as mpl
import numpy as np
import time
import model_op as op

from dataset import get_categories
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

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.figsize'] = (8, 6)
np.random.seed(int(time.time()))


if __name__ == '__main__':
    net = model_list[args.model](args.category_size)

    if args.model_path is not None:
        op.load_param(net, args.model_path)
        logger.info('load success from {0}'.format(args.model_path))

    if args.op == 'info':
        print(net)
    elif args.op == 'train':
        pass
    elif args.op == 'inference':
        pass

