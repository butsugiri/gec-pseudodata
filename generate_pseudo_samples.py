#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich
"""
Use insertion/deletion instead of the phrase table approach
"""

from __future__ import unicode_literals

import argparse
import codecs
import os
import random
import re
import sys
from collections import defaultdict
# hack for python2/3 compatibility
from io import open
from logzero import logger

argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=os.path.abspath,
        # metavar='PATH',
        help="Input text (default: standard input).")
    parser.add_argument(
        '--dfile', '-d', type=argparse.FileType('r'), default=sys.stdin,
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--threshold', '-t', type=int, default=0,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--seed', '-s', type=int, default=1, metavar='SEED',
        help='Stop if no symbol pair has frequency >= SEED (default: %(default)s))')
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    parser.add_argument(
        '--prob_mask', '-pm', type=float, default=0.5,
        help="probability to use mask")

    parser.add_argument(
        '--prob_orig', '-po', type=float, default=0.2,
        help="probability to use original token")

    parser.add_argument(
        '--unigram_freq', '-uf', type=os.path.abspath,
        help="verbose mode.")

    parser.add_argument(
        '--single_mistake', '-sm', type=int, choices=[0, 1],
        help="if we want single mistake...")

    parser.add_argument(
        '--use_insertion', '-ui', type=int, choices=[0, 1], default=1,
        help="generate error by insertion?")

    parser.add_argument(
        '--use_deletion', '-ud', type=int, choices=[0, 1], default=1,
        help="generate error by deletion?")

    return parser


def get_vocabulary(fobj, threshold):
    """Read text and return dictionary that encodes vocabulary
    """
    p_dict = dict()
    add_c = 0
    for line in fobj:
        phrase = line.strip('\r\n ').split(' ||| ')

        src_list = phrase[0].split(' ')
        trg_list = phrase[1].split(' ')
        if len(src_list) == 1 or len(trg_list) == 1:  # 長さが1のものは使わない
            continue
        elif len(src_list) == len(trg_list) and len(trg_list) > 1 and (
                src_list[0] == trg_list[0] or src_list[-1] == trg_list[-1]):  # 長さが同じ場合は，先頭か末尾が同じなら許容する
            pass
        elif not (src_list[0] == trg_list[0] and src_list[-1] == trg_list[-1]):  # （長さが違う場合は）先頭と末尾が同じ場合だけ許容
            continue

        p_src = phrase[0].strip('\r\n ')  # .split()
        p_trg = phrase[1].strip('\r\n ')  # .split()
        count = int(phrase[-1])
        if p_trg not in p_dict:
            p_dict[p_trg] = []
        if not (count < threshold):
            p_dict[p_trg].append((p_src, count))
            add_c += 1
        p = ""
        for w in trg_list[::-1]:
            p = w + " " + p if p != "" else w
            if p not in p_dict:
                p_dict[p] = []
    sys.stderr.write('vocab Done len={} add_c={}\n'.format(len(p_dict), add_c))
    return p_dict


def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word) - 1 and old_word[i + 1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i - 1:i + 1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word) - 2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                        nex = old_word[i + 1:i + 3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i - 1:i + 1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word) - 1 and word[i + 1] != new_pair:
                nex = word[i:i + 2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1


def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # bpe structure of pair frequencies
    stats = defaultdict(int)

    # index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes


def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item, freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def main(dict_file, infile, outfile, threshold, index2word, word_index_list, r_seed=1, verbose=False, is_dict=False,
         prob_mask=0.3, prob_orig=0.2, args=None):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    sys.stderr.write('random seed: {}\n'.format(r_seed))
    random.seed(r_seed)
    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    # outfile.write('#version: 0.2\n')

    # vocab = get_vocabulary(dict_file, threshold)
    # for z in vocab:
    #     sys.stdout.write('vocab[{}] ||| {} ||| {}\n'.format(z, len(vocab[z]), vocab[z]))

    proceed = 0
    skip = 0
    # with open(infile, 'r') as fi:
    for c, line in enumerate(sys.stdin):  # 入力分の読み込み
        wlist = line.strip('\r\n ').split(' ')
        proceed += 1

        output_list = []
        for i in range(1):  # 複数の候補を作る場合はここの数を修正する
            maxlen = len(wlist)
            cnt = 0
            t_out = ""
            # sys.stdout.write('#START {}->{}: {}\n'.format(i,maxlen, t_out))
            # print('original: ', wlist)
            # print()
            while cnt < maxlen:
                rnd = random.random()
                # print('focus: {}'.format(wlist[cnt]))
                if rnd < prob_orig:  # 空リストのばあいは必ずスキップ or 確率0.3以下で元の単語を選択
                    # t_out += '**' + wlist[cnt] + ' ' # 出力の文字列
                    t_out += wlist[cnt] + ' '  # 出力の文字列
                    # sys.stdout.write('# {} | {}\n'.format(cnt, t_out))
                    # print('keep orig: {}'.format(wlist[cnt]))
                    cnt += 1
                elif rnd < prob_mask:  # 空リストのばあいは必ずスキップ or 確率0.3以下で元の単語を選択
                    t_out += '| '  # 出力の文字列
                    # print('masking: {}'.format(wlist[cnt]))
                    cnt += 1
                else:
                    rnd2 = random.random()
                    if rnd2 < 0.5:  # insert
                        t_out += wlist[cnt] + ' '
                        if args.use_insertion:
                            index = random.choice(word_index_list)
                            t_out += index2word[index] + ' '
                        cnt += 1
                        # sys.stderr.write('insert: {}\n'.format(index2word[index]))
                    else:  # delete
                        # sys.stderr.write('delete: {}\n'.format(wlist[cnt]))
                        if not args.use_deletion:
                            t_out += wlist[cnt] + ' '
                        cnt += 1
                # print(t_out)
            if t_out.strip('\r\n ') == line.strip('\r\n '):
                # sys.stdout.write('#SAME {}||| {}'.format(t_out, line))
                padsize = random.randrange(1, 9)
                pad = ""
                for i in range(padsize):
                    pad += '| '
                output_list.append('{}||| {}{}'.format(t_out, pad, line))
            else:
                output_list.append('{}||| {}'.format(t_out, line))
            # print(output_list[0])
            # exit()
        # sys.stdout.write('{}\n'.format( len(output_list)))
        sys.stdout.write('{}'.format(random.choice(output_list)))
    sys.stderr.write('# {} {}\n'.format(proceed, skip))
    return 0


def single_mistake(dict_file, infile, outfile, threshold, index2word, word_index_list, r_seed=1, verbose=False, is_dict=False,
         prob_mask=0.3, prob_orig=0.2):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    sys.stderr.write('random seed: {}\n'.format(r_seed))
    random.seed(r_seed)
    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    # outfile.write('#version: 0.2\n')

    # vocab = get_vocabulary(dict_file, threshold)
    # for z in vocab:
    #     sys.stdout.write('vocab[{}] ||| {} ||| {}\n'.format(z, len(vocab[z]), vocab[z]))

    proceed = 0
    skip = 0
    # with open(infile, 'r') as fi:
    for c, line in enumerate(sys.stdin):  # 入力分の読み込み
        wlist = line.strip('\r\n ').split(' ')
        proceed += 1

        output_list = []
        for i in range(1):  # 複数の候補を作る場合はここの数を修正する
            maxlen = len(wlist)
            cnt = 0
            t_out = ""
            # sys.stdout.write('#START {}->{}: {}\n'.format(i,maxlen, t_out))
            # print('original: ', wlist)
            # print()
            mistake_idx = random.choice(list(range(maxlen)))
            while cnt < maxlen:
                if mistake_idx == cnt:
                    # print('i am making mistake', cnt, mistake_idx)
                    rnd = random.random()
                    # print(rnd)
                    # print('focus: {}'.format(wlist[cnt]))
                    if rnd < prob_orig:  # 空リストのばあいは必ずスキップ or 確率0.3以下で元の単語を選択
                        # t_out += '**' + wlist[cnt] + ' ' # 出力の文字列
                        t_out += wlist[cnt] + ' '  # 出力の文字列
                        # sys.stdout.write('# {} | {}\n'.format(cnt, t_out))
                        # print('keep orig: {}'.format(wlist[cnt]))
                        cnt += 1
                    elif rnd < prob_mask:  # 空リストのばあいは必ずスキップ or 確率0.3以下で元の単語を選択
                        t_out += '| '  # 出力の文字列
                        # print('masking: {}'.format(wlist[cnt]))
                        cnt += 1
                    else:
                        rnd2 = random.random()
                        # print(rnd2)
                        if rnd2 < 0.5:  # insert
                            t_out += wlist[cnt] + ' '
                            index = random.choice(word_index_list)
                            t_out += index2word[index] + ' '
                            cnt += 1
                            # sys.stdout.write('insert: {}\n'.format(index2word[index]))
                        else:  # delete
                            # sys.stdout.write('delete: {}\n'.format(wlist[cnt]))
                            cnt += 1
                else:
                    # print('keep original', wlist[cnt])
                    # print(t_out)
                    t_out += wlist[cnt] + ' '
                    cnt += 1
                # print(t_out)
            if t_out.strip('\r\n ') == line.strip('\r\n '):
                # sys.stdout.write('#SAME {}||| {}'.format(t_out, line))
                padsize = random.randrange(1, 9)
                pad = ""
                for i in range(padsize):
                    pad += '| '
                output_list.append('{}||| {}{}'.format(t_out, pad, line))
            else:
                output_list.append('{}||| {}'.format(t_out, line))
            # print(output_list[0])
            # exit()
        # sys.stdout.write('{}\n'.format( len(output_list)))
        sys.stdout.write('{}'.format(random.choice(output_list)))
    sys.stderr.write('# {} {}\n'.format(proceed, skip))
    return 0


def read_unigram_freq(path_to_unigram_freq):
    index2word = {}
    word_index_list = []
    with open(path_to_unigram_freq, 'r') as fi:
        for n, line in enumerate(fi):
            token, freq = line.strip().split('\t')
            index2word[n] = token
            word_index_list += [n] * int(freq)
    return index2word, word_index_list


if __name__ == '__main__':
    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    # if args.input.name != '<stdin>':
    #     pass
        # args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    # word_index_listには，単語のindexが頻度個だけ並んでいる
    logger.info('loading unigram frequency...')
    index2word, word_index_list = read_unigram_freq(args.unigram_freq)
    logger.info('index2word contains {} words'.format(len(index2word)))
    logger.info('word_index_list sample: {}'.format(word_index_list[:1000]))
    logger.info('word_index_list sample: {}'.format(word_index_list[323235:323335]))

    # assert args.prob_orig < args.prob_mask
    if args.single_mistake:
        logger.info('Making single mistake in single sequence')
        single_mistake(
            dict_file=args.dfile,
            infile=args.input,
            outfile=args.output,
            threshold=args.threshold,
            r_seed=args.seed,
            verbose=args.verbose,
            prob_orig=args.prob_orig,
            prob_mask=args.prob_mask,
            index2word=index2word,
            word_index_list=word_index_list
        )
    else:
        logger.info('Making mistake in each token')
        main(
            dict_file=args.dfile,
            infile=args.input,
            outfile=args.output,
            threshold=args.threshold,
            r_seed=args.seed,
            verbose=args.verbose,
            prob_orig=args.prob_orig,
            prob_mask=args.prob_mask,
            index2word=index2word,
            word_index_list=word_index_list,
            args=args
        )
