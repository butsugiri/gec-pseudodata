# -*- coding: utf-8 -*-
"""

"""
import argparse
import gzip
import os
from pathlib import Path

import spacy
from blingfire import text_to_sentences
from logzero import logger

SPACY_MODEL = 'en_core_web_sm'


def get_args():
    parser = argparse.ArgumentParser(description='sentence split and tokenize large corpus')
    parser.add_argument('--input', '-i', required=True, type=os.path.abspath,
                        help='path to input file')
    parser.add_argument('--output', '-o', required=True, type=os.path.abspath,
                        help='path to output dir')
    args = parser.parse_args()
    return args


def ssplit(text):
    sentences = text_to_sentences(text.strip()).split('\n')
    return sentences


def tokenize(text, nlp):
    doc = nlp(text.strip(), disable=['parser', 'tagger', 'ner'])
    tokens = [str(token) for token in doc]
    return tokens


def main(args):
    nlp = spacy.load(SPACY_MODEL)

    logger.info('Processing: {}'.format(args.input))
    dest = Path(args.output, *Path(args.input).parts[-2:])
    with gzip.open(args.input, 'rt') as fi, \
            gzip.open(dest, 'wb') as fo:
        for line in fi:
            sentences = ssplit(line)
            for sent in sentences:
                tokens = tokenize(sent, nlp)
                out = ' '.join(tokens) + '\n'
                fo.write(out.encode('utf-8'))


if __name__ == "__main__":
    args = get_args()
    main(args)
