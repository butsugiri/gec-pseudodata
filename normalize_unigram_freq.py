# -*- coding: utf-8 -*-
"""
一定頻度以下は同じ頻度のvocabとして扱うことにする
"""
import argparse
import sys

from logzero import logger


def main(fi, norm):
    sum_freq = 0
    for line in fi:
        token, freq = line.strip().split('\t')
        normalized_freq = max(int(freq) // norm, 1)
        sum_freq += normalized_freq
        print('{}\t{}'.format(token, normalized_freq))
    logger.info('sum_freq: {}'.format(sum_freq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hogehoge')
    parser.add_argument('--norm', default=300, type=int, help='write here')
    args = parser.parse_args()
    main(sys.stdin, args.norm)
