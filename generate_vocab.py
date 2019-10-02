# -*- coding: utf-8 -*-
"""

"""
import sys

from logzero import logger


def main(fi):
    vocab = set()
    logger.info('building vocabs')
    for line in fi:
        tokens = line.strip().split(' ')
        for token in tokens:
            vocab.add(token)
    logger.info('done')
    for token in vocab:
        print(token)


if __name__ == "__main__":
    main(sys.stdin)
