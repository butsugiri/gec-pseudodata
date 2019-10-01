# -*- coding: utf-8 -*-
"""
pre-processing large monolingual corpus with several heuristics
"""
import argparse
import os
import re
import string
from pathlib import Path

from logzero import logger

SYMBOLS = set(string.punctuation)
ASCII_CHARS = set(string.printable)


def get_args():
    parser = argparse.ArgumentParser(description='my script')
    parser.add_argument('--input', '-i', default=None, help='files to read, if empty, stdin is used')
    parser.add_argument('--output', '-o', required=True, type=os.path.abspath,
                        help='path to output dir')
    args = parser.parse_args()
    return args


def remove_long_sent(line, threshold=80):
    tokens = line.split(' ')
    if len(tokens) > threshold:
        return None
    else:
        return line


def remove_short_sent(line, threshold=2):
    tokens = line.split(' ')
    if len(tokens) <= threshold:
        return None
    else:
        return line


def remove_too_many_puncts(line, thresh_ratio=0.20):
    tokens = line.split(' ')
    n_puncs = len([t for t in tokens if t in SYMBOLS])
    n_total = len(tokens)
    ratio = (n_puncs / n_total)
    if ratio >= thresh_ratio and n_total >= 10:
        return None
    else:
        return line


def remove_nonascii_chars(line):
    filterred = ''.join(c for c in line if c in ASCII_CHARS)
    if len(filterred) < len(line):
        return None
    else:
        return line


def remove_consecutive_whitespace(line):
    if re.search(r'\s{3,}', line):
        return None
    else:
        return line


def remove_too_many_digits_sentence(line):
    total_tokens = len(line.split())
    match = re.findall(r'\s\d[\d,\/]*\s', line)
    if not match:
        return line
    else:
        n_digit_tokens = len(match)
        if n_digit_tokens / total_tokens > 0.10:
            return None
        else:
            return line


def main(args):
    dest = Path(args.output, *Path(args.input).parts[-1:])
    logger.info('Processing: {}'.format(args.input))

    with open(args.input, 'r') as fi, open(dest, 'w') as fo:
        for line in fi:
            line = line.strip()

            # remove if line is too long
            if line:
                line = remove_long_sent(line)

            # remove if line is too short
            if line:
                line = remove_short_sent(line)

            # remove if certain ratio of tokens are symbols
            if line:
                line = remove_too_many_puncts(line)

            # remove if line contains non-ascii characters
            if line:
                line = remove_nonascii_chars(line)

            # remove if consecutive spaces exist (probably the numerical table)
            if line:
                line = remove_consecutive_whitespace(line)

            # remove if too many digits exist
            if line:
                line = remove_too_many_digits_sentence(line)

            if line:
                line = line + '\n'
                # fo.write(line.encode('utf-8'))
                fo.write(line)


if __name__ == "__main__":
    args = get_args()
    main(args)
