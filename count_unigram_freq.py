# -*- coding: utf-8 -*-
"""
counting unigram frequency
"""
import sys
from collections import defaultdict

from logzero import logger


def main(fi):
    logger.info('start counting')
    d = defaultdict(int)
    for line in fi:
        tokens = line.strip().split()
        for token in tokens:
            d[token] += 1
    logger.info('finish counting')

    logger.info('printing to stdout')
    for token, freq in sorted(d.items(), key=lambda x: x[1], reverse=True):
        print('{}\t{}'.format(token, freq))
    logger.info('done')


if __name__ == "__main__":
    main(sys.stdin)
