import sys
import matplotlib as mpl
import numpy as np
import time

from dataset import get_categories
from parser import Parser


parser = Parser()
categories = get_categories('./data')
args, _ = parser.parse_known_args(sys.argv[1:])

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.figsize'] = (8, 6)
np.random.seed(time.time())


if __name__ == '__main__':
    pass
